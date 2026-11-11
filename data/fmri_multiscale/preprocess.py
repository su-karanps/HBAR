import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .atlas import build_hard_assignments, spatial_chunk_tokens


def _subject_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(sub-[A-Za-z0-9]+)", base)
    if m:
        return m.group(1)
    stem = base.replace(".nii.gz", "").replace(".nii", "")
    return stem


def _zscore_per_voxel(x_tv: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x_tv.mean(axis=0, keepdims=True)
    std = x_tv.std(axis=0, keepdims=True)
    return (x_tv - mean) / (std + eps)


def _aggregate_mean(ts_tv: np.ndarray, ids_v: np.ndarray, n_ids: int) -> np.ndarray:
    t, _ = ts_tv.shape
    out = np.zeros((t, n_ids), dtype=np.float32)
    counts = np.bincount(ids_v, minlength=n_ids).astype(np.float32)
    counts[counts == 0] = 1.0
    for i in range(t):
        out[i] = np.bincount(ids_v, weights=ts_tv[i], minlength=n_ids).astype(np.float32) / counts
    return out


@dataclass
class LayoutMeta:
    n_voxels: int
    n_bottom_tokens: int
    chunk_size: int
    n_parcels: int
    n_networks: int
    network_mapping_source: str
    voxel_storage_dtype: str


class MultiScalePreprocessor:
    def __init__(
        self,
        output_dir: str,
        gm_mask_path: Optional[str] = None,
        difumo_maps_path: Optional[str] = None,
        difumo_labels_csv_path: Optional[str] = None,
        n_parcels: int = 1024,
        chunk_size: int = 16,
        voxel_storage_dtype: str = "float32",
        layout_dir: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.samples_dir = os.path.join(output_dir, "samples")
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        if voxel_storage_dtype not in {"float32", "float16"}:
            raise ValueError("voxel_storage_dtype must be one of: float32, float16")
        if layout_dir is not None:
            self.layout = self._load_layout(layout_dir)
            loaded_dtype = self.layout["meta"].voxel_storage_dtype
            if loaded_dtype not in {"float32", "float16"}:
                raise ValueError(f"Invalid voxel_storage_dtype in layout meta: {loaded_dtype}")
            self.voxel_storage_dtype = loaded_dtype
        else:
            if gm_mask_path is None or difumo_labels_csv_path is None:
                raise ValueError("gm_mask_path and difumo_labels_csv_path are required when layout_dir is not provided")
            gm_mask_img = nib.load(gm_mask_path)
            difumo_img = nib.load(difumo_maps_path) if difumo_maps_path else None

            self.voxel_storage_dtype = voxel_storage_dtype
            assignments = build_hard_assignments(
                gm_mask_img=gm_mask_img,
                difumo_labels_csv_path=difumo_labels_csv_path,
                difumo_maps_img=difumo_img,
                n_parcels=n_parcels,
            )
            chunk_indices, chunk_parcels = spatial_chunk_tokens(
                voxel_coords=assignments.voxel_coords,
                voxel_to_parcel=assignments.voxel_to_parcel,
                chunk_size=chunk_size,
                n_parcels=n_parcels,
            )

            self.layout = {
                "voxel_mask_flat_indices": assignments.voxel_mask_flat_indices,
                "voxel_coords": assignments.voxel_coords,
                "voxel_to_parcel": assignments.voxel_to_parcel,
                "parcel_to_network": assignments.parcel_to_network,
                "chunk_indices": chunk_indices,
                "chunk_parcels": chunk_parcels,
                "meta": LayoutMeta(
                    n_voxels=int(assignments.voxel_mask_flat_indices.size),
                    n_bottom_tokens=int(chunk_indices.shape[0]),
                    chunk_size=int(chunk_size),
                    n_parcels=int(n_parcels),
                    n_networks=int(assignments.n_networks),
                    network_mapping_source=f"difumo_labels_csv_yeo18:{os.path.abspath(difumo_labels_csv_path)}",
                    voxel_storage_dtype=voxel_storage_dtype,
                ),
            }
            self._save_layout()
            self._save_layout_qc_plot()

    def _load_layout(self, layout_dir: str) -> Dict[str, object]:
        layout_npz = np.load(os.path.join(layout_dir, "layout.npz"))
        with open(os.path.join(layout_dir, "layout_meta.json"), "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = LayoutMeta(**meta_dict)
        return {
            "voxel_mask_flat_indices": layout_npz["voxel_mask_flat_indices"],
            "voxel_coords": layout_npz["voxel_coords"],
            "voxel_to_parcel": layout_npz["voxel_to_parcel"],
            "parcel_to_network": layout_npz["parcel_to_network"],
            "chunk_indices": layout_npz["chunk_indices"],
            "chunk_parcels": layout_npz["chunk_parcels"],
            "meta": meta,
        }

    def _save_layout(self) -> None:
        np.savez_compressed(
            os.path.join(self.output_dir, "layout.npz"),
            voxel_mask_flat_indices=self.layout["voxel_mask_flat_indices"],
            voxel_coords=self.layout["voxel_coords"],
            voxel_to_parcel=self.layout["voxel_to_parcel"],
            parcel_to_network=self.layout["parcel_to_network"],
            chunk_indices=self.layout["chunk_indices"],
            chunk_parcels=self.layout["chunk_parcels"],
        )
        with open(os.path.join(self.output_dir, "layout_meta.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.layout["meta"]), f, indent=2)

    def _save_layout_qc_plot(self) -> None:
        chunk_parcels = self.layout["chunk_parcels"]
        if chunk_parcels.size == 0:
            return
        counts = np.bincount(chunk_parcels, minlength=self.layout["meta"].n_parcels)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(counts[counts > 0], bins=40)
        axes[0].set_title("Bottom tokens per parcel")
        axes[0].set_xlabel("tokens")
        axes[0].set_ylabel("parcel count")
        voxel_to_parcel = self.layout["voxel_to_parcel"]
        voxel_counts = np.bincount(voxel_to_parcel, minlength=self.layout["meta"].n_parcels)
        axes[1].hist(voxel_counts[voxel_counts > 0], bins=40)
        axes[1].set_title("Voxel count per parcel")
        axes[1].set_xlabel("voxels")
        axes[1].set_ylabel("parcel count")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "layout_qc.png"), dpi=150)
        plt.close(fig)

    def _save_subject_qc_plot(
        self,
        subject_id: str,
        voxel_ts: np.ndarray,
        parcel_ts: np.ndarray,
        network_ts: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        t_show = min(150, voxel_ts.shape[0])
        axes[0].plot(voxel_ts[:t_show, :5], linewidth=1.0)
        axes[0].set_title("Voxel scale (5 examples)")
        axes[1].plot(parcel_ts[:t_show, :5], linewidth=1.0)
        axes[1].set_title("Parcel scale (5 examples)")
        axes[2].plot(network_ts[:t_show, :5], linewidth=1.0)
        axes[2].set_title("Network scale (5 examples)")
        for ax in axes:
            ax.set_xlabel("TR")
            ax.set_ylabel("Signal")
        fig.suptitle(f"Multi-scale QC: {subject_id}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, f"{subject_id}_multiscale_qc.png"), dpi=150)
        plt.close(fig)

    def process_volume(self, fmri_path: str, zscore_voxels: bool = True) -> Dict[str, object]:
        subject_id = _subject_id_from_path(fmri_path)
        img = nib.load(fmri_path)
        data = img.get_fdata()
        if data.ndim != 4:
            raise ValueError(f"Expected 4D fMRI data, got shape={data.shape} for {fmri_path}")

        t = data.shape[-1]
        x_tv = data.reshape(-1, t).T[:, self.layout["voxel_mask_flat_indices"]].astype(np.float32)
        if zscore_voxels:
            x_tv = _zscore_per_voxel(x_tv)

        parcel_ts = _aggregate_mean(
            ts_tv=x_tv,
            ids_v=self.layout["voxel_to_parcel"],
            n_ids=self.layout["meta"].n_parcels,
        )
        # Networks average the parcel series as they are finally stored, so with float16
        # storage they are computed from the float16-rounded parcel series.
        network_ts = _aggregate_mean(
            ts_tv=parcel_ts.astype(self.voxel_storage_dtype).astype(np.float32),
            ids_v=self.layout["parcel_to_network"],
            n_ids=self.layout["meta"].n_networks,
        )

        out_file = os.path.join(self.samples_dir, f"{subject_id}.npz")
        if self.voxel_storage_dtype == "float16":
            voxel_ts_out = x_tv.astype(np.float16)
        else:
            voxel_ts_out = x_tv
        np.savez_compressed(
            out_file,
            voxel_ts=voxel_ts_out,
            parcel_ts=parcel_ts,
            network_ts=network_ts,
        )
        self._save_subject_qc_plot(subject_id, x_tv, parcel_ts, network_ts)
        return {
            "subject_id": subject_id,
            "path": out_file,
            "timepoints": int(t),
        }

    def process_many(self, fmri_paths: List[str], zscore_voxels: bool = True) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for path in tqdm(fmri_paths, desc="Preprocessing fMRI volumes"):
            records.append(self.process_volume(path, zscore_voxels=zscore_voxels))

        index_path = os.path.join(self.output_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return records
