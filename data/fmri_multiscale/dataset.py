import json
import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _NpzCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> Dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        arr = np.load(path)
        data = {
            "voxel_ts": arr["voxel_ts"],
            "parcel_ts": arr["parcel_ts"],
            "network_ts": arr["network_ts"],
        }
        self._cache[path] = data
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return data


class _TripleArrayCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def get(self, voxel_path: str, parcel_path: str, network_path: str) -> Dict[str, np.ndarray]:
        key = f"{voxel_path}|{parcel_path}|{network_path}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = {
            # Use read-only memmap to avoid loading full subject arrays into RAM at cache insert.
            "voxel_ts": np.load(voxel_path, mmap_mode="r"),
            "parcel_ts": np.load(parcel_path, mmap_mode="r"),
            "network_ts": np.load(network_path, mmap_mode="r"),
        }
        self._cache[key] = data
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return data


class FMRIMultiScaleWindowDataset(Dataset):
    """
    Returns per-sample dict with:
      - voxel_window: [T, V]
      - bottom_tokens: [N0, T, K]
      - middle_tokens: [T, P]
      - top_tokens: [T, N]
      - token_parcels: [N0]
      - token_networks: [N0]
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        layout_dir: Optional[str] = None,
        window_size: int = 32,
        stride: int = 16,
        cache_items: int = 8,
        random_sampling: bool = False,
        max_windows_per_subject: Optional[int] = None,
    ) -> None:
        if cache_dir is None and index_path is None:
            raise ValueError("Either cache_dir or index_path must be provided.")
        self.cache_dir = cache_dir
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.random_sampling = bool(random_sampling)
        self._npz_cache = _NpzCache(max_items=cache_items)
        self._triple_cache = _TripleArrayCache(max_items=cache_items)

        if index_path is None:
            index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, object]] = json.load(f)
        if layout_dir is None:
            if cache_dir is None:
                raise ValueError("layout_dir must be provided when cache_dir is None")
            layout_dir = cache_dir
        layout = np.load(os.path.join(layout_dir, "layout.npz"))
        self.chunk_indices = layout["chunk_indices"].astype(np.int64)
        self.chunk_parcels = layout["chunk_parcels"].astype(np.int64)
        self.parcel_to_network = layout["parcel_to_network"].astype(np.int64)
        self.chunk_networks = self.parcel_to_network[self.chunk_parcels]
        self.n_parcels = int(self.parcel_to_network.shape[0])
        self.n_networks = int(np.max(self.parcel_to_network)) + 1

        self.windows: List[Tuple[int, int]] = []
        for ridx, rec in enumerate(self.records):
            t = int(rec["timepoints"])
            if t < self.window_size:
                continue
            starts = list(range(0, t - self.window_size + 1, self.stride))
            if max_windows_per_subject is not None and len(starts) > max_windows_per_subject:
                starts = starts[:max_windows_per_subject]
            self.windows.extend((ridx, s) for s in starts)

        self._subject_window_spans: List[Tuple[int, int]] = []
        for ridx, rec in enumerate(self.records):
            t = int(rec["timepoints"])
            if t < self.window_size:
                self._subject_window_spans.append((0, 0))
            else:
                max_start = t - self.window_size
                self._subject_window_spans.append((0, max_start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.random_sampling:
            rec_idx = random.randrange(len(self.records))
            rec = self.records[rec_idx]
            low, high = self._subject_window_spans[rec_idx]
            if high <= low:
                start = 0
            else:
                # step-constrained random starts for consistency with stride behavior
                candidates = list(range(low, high + 1, self.stride))
                start = random.choice(candidates) if candidates else 0
        else:
            rec_idx, start = self.windows[idx]
            rec = self.records[rec_idx]

        if "path" in rec:
            data = self._npz_cache.get(str(rec["path"]))
        else:
            data = self._triple_cache.get(
                str(rec["voxel_path"]),
                str(rec["parcel_path"]),
                str(rec["network_path"]),
            )
        stop = start + self.window_size

        voxel = data["voxel_ts"][start:stop]  # [T,V]
        parcel = data["parcel_ts"][start:stop]  # [T,P]
        network = data["network_ts"][start:stop]  # [T,N]

        # Convert memmap-backed slices to writable float32 arrays before torch.from_numpy.
        voxel = np.array(voxel, dtype=np.float32, copy=True)
        parcel = np.array(parcel, dtype=np.float32, copy=True)
        network = np.array(network, dtype=np.float32, copy=True)

        # Gather fixed-size sub-parcel tokens with zero padding for -1 entries.
        v = voxel.shape[1]
        padded = np.concatenate([voxel, np.zeros((voxel.shape[0], 1), dtype=voxel.dtype)], axis=1)
        gather = np.where(self.chunk_indices < 0, v, self.chunk_indices)
        bottom = padded[:, gather]  # [T,N0,K]
        bottom = np.transpose(bottom, (1, 0, 2))  # [N0,T,K]

        return {
            "voxel_window": torch.from_numpy(voxel),
            "bottom_tokens": torch.from_numpy(bottom).float(),
            "middle_tokens": torch.from_numpy(parcel),
            "top_tokens": torch.from_numpy(network),
            "chunk_indices": torch.from_numpy(self.chunk_indices).long(),
            "token_parcels": torch.from_numpy(self.chunk_parcels).long(),
            "token_networks": torch.from_numpy(self.chunk_networks).long(),
            "parcel_to_network": torch.from_numpy(self.parcel_to_network).long(),
            "subject_index": torch.tensor(rec_idx, dtype=torch.long),
        }


def make_dataloader(
    cache_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    layout_dir: Optional[str] = None,
    batch_size: int = 2,
    window_size: int = 32,
    stride: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    random_sampling: bool = False,
    max_windows_per_subject: Optional[int] = None,
) -> DataLoader:
    ds = FMRIMultiScaleWindowDataset(
        cache_dir=cache_dir,
        index_path=index_path,
        layout_dir=layout_dir,
        window_size=window_size,
        stride=stride,
        random_sampling=random_sampling,
        max_windows_per_subject=max_windows_per_subject,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
