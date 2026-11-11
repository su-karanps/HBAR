#!/usr/bin/env python3
"""
Encode the full fMRI window dataset with a trained tokenizer; save discrete token
ids (CSR), optional pooled post-quant vectors, and wall-clock timing.

Multi-GPU usage (via torchrun):
    torchrun --nproc_per_node=4 precompute_fMRI_tokenization.py [args...]

Each rank processes a disjoint round-robin shard of the dataset.  Rank 0 merges
all shards into the final output files after a distributed barrier.

Single-GPU / CPU usage (unchanged):
    python precompute_fMRI_tokenization.py [args...]
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm import tqdm as _tqdm_cls
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset
from downstream.hcp_metadata import infer_subject_id_from_record
from downstream.fmri_token_codec import TokenSequenceMeta, discretize_model, pooled_representation
from train_fmri import _build_model, _load_config


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_distributed() -> tuple[int, int, int]:
    """
    Initialize torch.distributed if running under torchrun.
    Returns (rank, world_size, local_rank).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def _merge_shards(
    out_dir: str,
    world_size: int,
    tag: str,
    save_pooled: bool,
    ds0: FMRIMultiScaleWindowDataset,
) -> tuple[int, TokenSequenceMeta]:
    """
    Rank 0 only: merge per-rank shard files into final output arrays.
    Returns (total_windows, tmeta_from_shard0).
    """
    all_rec: List[np.ndarray] = []
    all_wix: List[np.ndarray] = []
    all_len: List[np.ndarray] = []
    all_shard: List[np.ndarray] = []
    all_row: List[np.ndarray] = []
    all_pooled: List[np.ndarray] = []
    tmeta_dict: Optional[Dict[str, Any]] = None
    n_skip_total = 0
    shard_flats: List[np.ndarray] = []
    shard_indptrs: List[np.ndarray] = []

    for r in range(world_size):
        shard_dir = os.path.join(out_dir, f"_shard_{r}")
        flat = np.load(os.path.join(shard_dir, f"token_ids_{tag}.npy"))
        indptr = np.load(os.path.join(shard_dir, f"token_indptr_{tag}.npy"))
        rec = np.load(os.path.join(shard_dir, f"rec_idx_{tag}.npy"))
        wix = np.load(os.path.join(shard_dir, f"window_index_{tag}.npy"))
        shard_flats.append(flat)
        shard_indptrs.append(indptr)

        n_win = len(rec)
        all_rec.append(rec)
        all_wix.append(wix)
        all_len.append(np.diff(indptr).astype(np.int64, copy=False))
        all_shard.append(np.full(n_win, r, dtype=np.int32))
        all_row.append(np.arange(n_win, dtype=np.int32))

        if save_pooled:
            p = os.path.join(shard_dir, f"pooled_{tag}.npy")
            if os.path.exists(p):
                all_pooled.append(np.load(p))

        meta_path = os.path.join(shard_dir, f"meta_{tag}.json")
        if tmeta_dict is None and os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            tmeta_dict = m.get("discrete_meta", {})
            n_skip_total += int(m.get("batches_with_errors", 0))

    rec_cat = np.concatenate(all_rec).astype(np.int32, copy=False)
    wix_cat = np.concatenate(all_wix).astype(np.int32, copy=False)
    len_cat = np.concatenate(all_len).astype(np.int64, copy=False)
    shard_cat = np.concatenate(all_shard).astype(np.int32, copy=False)
    row_cat = np.concatenate(all_row).astype(np.int32, copy=False)

    # Sort by original window index for deterministic global ordering.
    order = np.argsort(wix_cat, kind="stable")
    rec_sorted = rec_cat[order]
    wix_sorted = wix_cat[order]
    len_sorted = len_cat[order]
    shard_sorted = shard_cat[order]
    row_sorted = row_cat[order]

    indptr_final = np.concatenate([[np.int64(0)], np.cumsum(len_sorted, dtype=np.int64)])
    flat_final = np.empty(int(indptr_final[-1]), dtype=np.int32)
    write = 0
    for s, rr, ln in zip(shard_sorted.tolist(), row_sorted.tolist(), len_sorted.tolist()):
        if ln <= 0:
            continue
        a = int(shard_indptrs[s][rr])
        b = int(shard_indptrs[s][rr + 1])
        flat_final[write : write + ln] = shard_flats[s][a:b]
        write += ln

    w_starts = np.array(
        [int(ds0.windows[i][1]) if i < len(ds0.windows) else 0 for i in wix_sorted],
        dtype=np.int32,
    )

    np.save(os.path.join(out_dir, f"token_ids_{tag}.npy"), flat_final.astype(np.int32))
    np.save(os.path.join(out_dir, f"token_indptr_{tag}.npy"), indptr_final)
    np.save(os.path.join(out_dir, f"rec_idx_{tag}.npy"), rec_sorted)
    np.save(os.path.join(out_dir, f"win_start_{tag}.npy"), w_starts)
    np.save(os.path.join(out_dir, f"window_index_{tag}.npy"), wix_sorted)

    if save_pooled and all_pooled:
        pooled_cat = np.concatenate(all_pooled, axis=0)
        if len(pooled_cat) == len(order):
            pooled_sorted = pooled_cat[order]
        else:
            pooled_sorted = pooled_cat
        np.save(os.path.join(out_dir, f"pooled_{tag}.npy"), pooled_sorted)

    # Cleanup shard directories.
    for r in range(world_size):
        shutil.rmtree(os.path.join(out_dir, f"_shard_{r}"), ignore_errors=True)

    tmeta = TokenSequenceMeta(
        model_type=str(tmeta_dict.get("model_type", "")),
        seq_len=int(tmeta_dict.get("seq_len", 0)),
        vocab_size=int(tmeta_dict.get("vocab_size", 0)),
        layout=tmeta_dict.get("layout"),
    ) if tmeta_dict else TokenSequenceMeta("", 0, 0, None)

    return int(len(rec_sorted)), tmeta


def _merge_pooled_only_shards(out_dir: str, world_size: int, tag: str) -> int:
    """Rank 0 only: merge pooled vectors from per-rank shards."""
    all_wix: List[np.ndarray] = []
    all_pooled: List[np.ndarray] = []

    for r in range(world_size):
        shard_dir = os.path.join(out_dir, f"_shard_{r}")
        wix = np.load(os.path.join(shard_dir, f"window_index_{tag}.npy"))
        pooled_path = os.path.join(shard_dir, f"pooled_{tag}.npy")
        if not os.path.isfile(pooled_path):
            raise FileNotFoundError(f"Missing pooled shard file: {pooled_path}")
        p = np.load(pooled_path)
        if p.shape[0] != wix.shape[0]:
            raise ValueError(
                f"Pooled/windows mismatch in shard {r}: "
                f"pooled={p.shape[0]} vs windows={wix.shape[0]}"
            )
        all_wix.append(wix.astype(np.int32, copy=False))
        all_pooled.append(p.astype(np.float32, copy=False))

    wix_cat = np.concatenate(all_wix).astype(np.int32, copy=False)
    pooled_cat = np.concatenate(all_pooled, axis=0).astype(np.float32, copy=False)
    order = np.argsort(wix_cat, kind="stable")
    pooled_sorted = pooled_cat[order]
    np.save(os.path.join(out_dir, f"pooled_{tag}.npy"), pooled_sorted)

    for r in range(world_size):
        shutil.rmtree(os.path.join(out_dir, f"_shard_{r}"), ignore_errors=True)
    return int(len(order))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _ckpt_load(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        n = k
        if n.startswith("module."):
            n = n[7:]
        if n.startswith("_orig_mod."):
            n = n[10:]
        cleaned[n] = v
    m, u = model.load_state_dict(cleaned, strict=False)
    if m:
        print(f"[ckpt] missing {len(m)} (show 3): {m[:3]}")
    if u:
        print(f"[ckpt] unexpected {len(u)} (show 3): {u[:3]}")


def _sliced_dict(out: Dict[str, Any], j: int) -> Dict[str, torch.Tensor]:
    """Per-batch-item slice. Skip scalars and non-batched tensors."""
    o: Dict[str, torch.Tensor] = {}
    for k, v in out.items():
        if not torch.is_tensor(v) or v.dim() == 0:
            continue
        if v.shape[0] <= j:
            continue
        o[k] = v[j : j + 1]
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0

    p = argparse.ArgumentParser()
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--data_config",
        type=str,
        default=os.path.join(REPO, "configs/fmri/data_probe.yaml"),
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for model inference (bfloat16 recommended for A100/H100).",
    )
    p.add_argument("--save_pooled", action="store_true")
    p.add_argument(
        "--pooled_only",
        action="store_true",
        help="Compute/save pooled features only; do not write token_ids/token_indptr.",
    )
    p.add_argument("--tag", type=str, default="")
    p.add_argument(
        "--bottom_groups",
        type=str,
        default="",
        help="npq_hier_residual only: comma-separated bit-group sizes for the bottom "
        "level (e.g. '10,9'), used instead of the model's group_size.",
    )
    args = p.parse_args()
    bottom_groups: Optional[List[int]] = (
        [int(g) for g in args.bottom_groups.split(",") if g.strip()] if args.bottom_groups else None
    )
    if args.pooled_only:
        args.save_pooled = True

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)

    # Resolve device.
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # dtype / autocast setup
    use_amp = args.dtype in ("bfloat16", "float16") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx: Any = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else contextlib.nullcontext()
    )

    mcfg = _load_config(args.model_config)
    dcfg = _load_config(args.data_config)
    mtype = str(mcfg["model"]["type"])
    data = dcfg["data"]

    ds0 = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=str(data["index_path"]),
        layout_dir=str(data["layout_dir"]),
        window_size=int(data.get("window_size", 32)),
        stride=int(data.get("stride", 16)),
        cache_items=int(data.get("cache_items", 4)),
        max_windows_per_subject=data.get("max_windows_per_subject"),
    )
    with open(data["index_path"], "r", encoding="utf-8") as f:
        recs: List[Dict[str, Any]] = json.load(f)
    n_rec = len(recs)
    subj_per_rec = [infer_subject_id_from_record(r) for r in recs]

    b0 = {k: v.unsqueeze(0) for k, v in ds0[0].items() if torch.is_tensor(v)}
    model = _build_model(mcfg, b0).to(device)
    _ckpt_load(model, args.checkpoint)
    model.eval()

    # Check whether this model supports encode_only.
    _supports_encode_only = "encode_only" in inspect.signature(model.forward).parameters

    # Shard the dataset across ranks (round-robin).
    rank_indices = list(range(rank, len(ds0), world_size))
    n_rank_windows = len(rank_indices)
    n_batches = (n_rank_windows + args.batch_size - 1) // args.batch_size

    print(
        f"[rank {rank}/{world_size}] device={device} dtype={args.dtype} "
        f"windows={n_rank_windows}/{len(ds0)} batches≈{n_batches} batch_size={args.batch_size}"
    )

    rank_subset = Subset(_Wrap(ds0), rank_indices)
    wdl = DataLoader(
        rank_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # Token ids are streamed into a preallocated .npy memmap (one row per window)
    # instead of being accumulated in RAM: the hierarchical dumps are ~48 GB and
    # list-accumulate + concatenate needs twice that.
    shard_dir = os.path.join(args.out_dir, f"_shard_{rank}") if world_size > 1 else args.out_dir
    os.makedirs(shard_dir, exist_ok=True)
    tag = args.tag or "a"
    tok_path = os.path.join(shard_dir, f"token_ids_{tag}.npy")
    tok_mm: Optional[np.memmap] = None
    tok_write = 0
    row_lens: List[int] = []
    rec_list: List[int] = []
    wix: List[int] = []
    pooled: List[List[float]] = []
    tmeta: TokenSequenceMeta = TokenSequenceMeta(mtype, 0, 0, None)
    w_count = 0
    n_skip = 0

    t_wall0 = time.perf_counter()
    t_last_log = t_wall0
    LOG_INTERVAL_SEC = 60.0  # print a progress line at most once per minute per rank

    # Wrap with tqdm when connected to a TTY; fall back to timed line-based logging.
    _is_tty = sys.stdout.isatty()
    if _HAVE_TQDM and _is_tty:
        _iter = _tqdm_cls(
            wdl,
            desc=f"rank {rank}",
            unit="batch",
            total=n_batches,
            dynamic_ncols=True,
            position=rank,
            leave=True,
        )
    else:
        _iter = wdl

    def _log_progress(batch_idx: int, t_now: float) -> None:
        elapsed = t_now - t_wall0
        wps = w_count / max(elapsed, 1e-6)
        remaining = (n_rank_windows - w_count) / max(wps, 1e-6)
        mem_str = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            res = torch.cuda.memory_reserved(device) / 1024**3
            mem_str = f" gpu_mem={alloc:.1f}/{res:.1f}GiB"
        print(
            f"[rank {rank} | batch {batch_idx}/{n_batches} | "
            f"win {w_count}/{n_rank_windows} | "
            f"elapsed {elapsed:.0f}s | "
            f"w/s {wps:.1f} | "
            f"ETA {remaining:.0f}s{mem_str}]",
            flush=True,
        )

    for batch_idx, batch in enumerate(_iter):
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if b["bottom_tokens"].shape[0] == 0:
            continue
        try:
            with autocast_ctx:
                if _supports_encode_only:
                    out = model(b, encode_only=True)
                else:
                    out = model(b)
        except Exception as e:  # pragma: no cover
            print(f"[rank {rank}][skip batch @ {w_count}] {e}")
            n_skip += 1
            w_count += int(b["bottom_tokens"].shape[0])
            continue
        toks_t: Optional[torch.Tensor] = None
        if not args.pooled_only:
            try:
                toks_t, _v, tmeta = discretize_model(model, mtype, b, out, bottom_groups=bottom_groups)
            except Exception as e:  # pragma: no cover
                print(f"[rank {rank}][discretize fail] {e}")
                raise
        bs = b["bottom_tokens"].shape[0]
        for j in range(bs):
            if toks_t is not None:
                row = toks_t[j].to(torch.int64).cpu().numpy().astype(np.int32).ravel()
                if tok_mm is None:
                    tok_mm = np.lib.format.open_memmap(
                        tok_path, mode="w+", dtype=np.int32, shape=(n_rank_windows * int(row.shape[0]),)
                    )
                if tok_write + int(row.shape[0]) > tok_mm.shape[0]:
                    raise RuntimeError("token rows are not a constant length; cannot stream to memmap")
                tok_mm[tok_write : tok_write + int(row.shape[0])] = row
                tok_write += int(row.shape[0])
                row_lens.append(int(row.shape[0]))
            ri = int(b["subject_index"][j].item())
            rec_list.append(ri)
            if args.save_pooled:
                pvo = _sliced_dict(out, j)
                vec = pooled_representation(pvo, mtype)
                pooled.append(vec.squeeze(0).float().cpu().numpy().tolist())
            wix.append(int(b["window_index"][j].item()))
        w_count += bs

        # Update tqdm postfix or emit a timed log line.
        t_now = time.perf_counter()
        if _HAVE_TQDM and _is_tty:
            elapsed = t_now - t_wall0
            wps = w_count / max(elapsed, 1e-6)
            _iter.set_postfix({"w/s": f"{wps:.1f}", "skip": n_skip}, refresh=False)  # type: ignore[union-attr]
        elif t_now - t_last_log >= LOG_INTERVAL_SEC:
            _log_progress(batch_idx + 1, t_now)
            t_last_log = t_now

    t_wall1 = time.perf_counter()

    # Final progress summary for this rank.
    elapsed_total = t_wall1 - t_wall0
    _log_progress(n_batches, t_wall1)

    # Save this rank's shard.
    if not args.pooled_only:
        if tok_mm is not None:
            tok_mm.flush()
            n_alloc = int(tok_mm.shape[0])
            del tok_mm
            if tok_write != n_alloc:  # only if batches were skipped: trim the file
                np.save(tok_path, np.load(tok_path, mmap_mode="r")[:tok_write])
        else:
            np.save(tok_path, np.zeros((0,), dtype=np.int32))
        lens_arr = np.array(row_lens, dtype=np.int64) if row_lens else np.zeros((0,), dtype=np.int64)
        indp = np.concatenate([[np.int64(0)], np.cumsum(lens_arr, dtype=np.int64)])
        np.save(os.path.join(shard_dir, f"token_indptr_{tag}.npy"), indp)
    np.save(os.path.join(shard_dir, f"rec_idx_{tag}.npy"), np.array(rec_list, dtype=np.int32))
    np.save(os.path.join(shard_dir, f"window_index_{tag}.npy"), np.array(wix, dtype=np.int32))

    if args.save_pooled and pooled:
        np.save(os.path.join(shard_dir, f"pooled_{tag}.npy"), np.array(pooled, dtype=np.float32))

    if not args.pooled_only:
        # Write a per-shard meta for rank 0 to read during merge.
        shard_meta: Dict[str, Any] = {
            "model_type": mtype,
            "wall_time_sec": float(elapsed_total),
            "n_windows": w_count,
            "batches_with_errors": n_skip,
            "discrete_meta": {
                **tmeta.__dict__,
                "model_type": tmeta.model_type,
                "seq_len": tmeta.seq_len,
                "vocab_size": tmeta.vocab_size,
            },
        }
        with open(os.path.join(shard_dir, f"meta_{tag}.json"), "w") as f:
            json.dump(shard_meta, f, indent=2)

    print(
        f"[rank {rank}] done  windows={w_count} "
        f"wall_s={elapsed_total:.1f}  "
        f"w/s={w_count / max(elapsed_total, 1e-6):.2f}"
    )

    # Sync all ranks then let rank 0 merge.
    _barrier(world_size)

    if is_main and world_size > 1:
        print("[rank 0] merging shards...")
        if args.pooled_only:
            total_windows = _merge_pooled_only_shards(args.out_dir, world_size, tag)
        else:
            total_windows, tmeta = _merge_shards(args.out_dir, world_size, tag, args.save_pooled, ds0)
    elif world_size == 1:
        if not args.pooled_only:
            # Single-rank: also save win_start which wasn't saved in the shard loop above.
            w_starts = np.array(
                [int(ds0.windows[i][1]) if i < len(ds0.windows) else 0 for i in wix],
                dtype=np.int32,
            )
            np.save(os.path.join(args.out_dir, f"win_start_{tag}.npy"), w_starts)
        total_windows = w_count

    if is_main:
        t_total = time.perf_counter() - t_wall0
        if args.pooled_only:
            print(
                f"Done (pooled_only). windows={total_windows} wall_s={t_total:.1f} "
                f"w/s={total_windows / max(1e-6, t_total):.2f} -> {args.out_dir}"
            )
        else:
            meta: Dict[str, Any] = {
                "name": args.tag,
                "model_type": mtype,
                "model_config": os.path.abspath(args.model_config),
                "checkpoint": os.path.abspath(args.checkpoint),
                "data_config": os.path.abspath(args.data_config),
                "index_path": data["index_path"],
                "n_record_files": n_rec,
                "n_windows": total_windows,
                "seq_len_per_window": tmeta.seq_len,
                "vocab_size": tmeta.vocab_size,
                "discrete_meta": tmeta.__dict__,
                "wall_time_sec": float(t_total),
                "windows_per_sec": float(total_windows / max(1e-6, t_total)),
                "batches_with_errors": n_skip,
                "world_size": world_size,
                "dtype": args.dtype,
                "bottom_groups": bottom_groups,
                "subject_id_per_record": [
                    subj_per_rec[i] if i < len(subj_per_rec) else "?" for i in range(n_rec)
                ],
            }
            with open(os.path.join(args.out_dir, f"meta_{args.tag or 'a'}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(
                f"Done. windows={total_windows} wall_s={t_total:.1f} "
                f"w/s={meta['windows_per_sec']:.2f} -> {args.out_dir}"
            )

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


class _Wrap(Dataset):
    def __init__(self, base: FMRIMultiScaleWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        d = self.base[i]
        d = dict(d)
        d["window_index"] = torch.tensor(i, dtype=torch.long)
        return d


if __name__ == "__main__":
    main()
