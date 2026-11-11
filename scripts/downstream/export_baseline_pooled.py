#!/usr/bin/env python3
"""
Pooled *raw* or engineered summary features (no tokenizer), for `linear_probe_hcp.py`.

- difumo: time-mean of `middle_tokens` (DiFuMo parcel BOLD), 1024-D per window.
- yeo: time-mean of `top_tokens` (network-level), 17-D per window.
- fc: upper-triangle of parcel–parcel Pearson correlation per window, then
      IncrementalPCA to `--fc_pca_dim` (default 256).
- pca: flatten `middle_tokens` [T*P] per window, IncrementalPCA to `--pca_dim`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset
from data.fmri_multiscale.splits import load_split, resolve_split_path, window_indices_for_split
from downstream.hcp_metadata import infer_subject_id_from_record
from train_fmri import _load_config


def _fc_upper_tri_vec(x_tp: np.ndarray) -> np.ndarray:
    """x_tp: [T, P] float — Pearson corr [P,P], upper triangle (no diag), row-major."""
    t, p = x_tp.shape
    if t < 2:
        return np.zeros(p * (p - 1) // 2, dtype=np.float32)
    # constant columns -> nan in corrcoef; replace nan with 0
    r = np.corrcoef(x_tp, rowvar=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(p, k=1)
    return r[iu].astype(np.float32)


def _export_difumo_yeo(
    wdl: DataLoader,
    ds0: FMRIMultiScaleWindowDataset,
    name: str,
    out_dir: str,
    data: Dict[str, Any],
    subj: List[str],
    data_config_abs: str,
) -> None:
    t0 = time.perf_counter()
    feats: List[List[float]] = []
    wix: List[int] = []
    rec: List[int] = []
    for b in wdl:
        wix.extend(b["window_index"].cpu().numpy().astype(np.int32).tolist())
        rec.extend(b["subject_index"].cpu().numpy().astype(np.int32).tolist())
        m = b["middle_tokens" if name == "difumo" else "top_tokens"].float()
        z = m.mean(dim=1).cpu().numpy()
        for row in z:
            feats.append(row.astype(np.float32).tolist())
    t1 = time.perf_counter()
    X = np.array(feats, dtype=np.float32)
    w_st = [int(ds0.windows[i][1]) for i in wix]
    tag = name
    np.save(os.path.join(out_dir, f"pooled_{tag}.npy"), X)
    np.save(os.path.join(out_dir, f"rec_idx_{tag}.npy"), np.array(rec, np.int32))
    np.save(os.path.join(out_dir, f"win_start_{tag}.npy"), np.array(w_st, np.int32))
    np.save(os.path.join(out_dir, f"window_index_{tag}.npy"), np.array(wix, np.int32))
    meta = {
        "name": tag,
        "fdim": int(X.shape[1]) if X.size else 0,
        "n_windows": int(X.shape[0]) if X.size else 0,
        "wall_time_sec": float(t1 - t0),
        "data_config": data_config_abs,
        "index_path": data["index_path"],
        "subject_id_per_record": subj,
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {tag} {X.shape} in {t1 - t0:.1f}s -> {out_dir}")


def _export_fc_two_pass(
    wdl: DataLoader,
    ds0: FMRIMultiScaleWindowDataset,
    out_dir: str,
    data: Dict[str, Any],
    subj: List[str],
    fc_pca_dim: int,
    data_config_abs: str,
    fit_indices: Optional[List[int]] = None,
) -> None:
    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError as e:
        raise SystemExit("export_baseline_pooled --name fc requires scikit-learn") from e

    t0 = time.perf_counter()
    d0 = ds0[0]["middle_tokens"]
    t_win, p = int(d0.shape[0]), int(d0.shape[1])
    d_fc = p * (p - 1) // 2
    n_comp = min(fc_pca_dim, d_fc)

    ipca = IncrementalPCA(n_components=n_comp, batch_size=min(256, max(32, n_comp)))

    wdl_fit = DataLoader(
        wdl.dataset if fit_indices is None else Subset(wdl.dataset, fit_indices),
        batch_size=wdl.batch_size,
        shuffle=False,
        num_workers=wdl.num_workers,
    )
    buf: List[np.ndarray] = []

    def _flush_fc(force: bool = False) -> None:
        nonlocal buf
        while len(buf) >= n_comp:
            chunk = np.stack(buf[:n_comp], axis=0)
            buf = buf[n_comp:]
            ipca.partial_fit(chunk)
        if force and buf:
            arr = np.stack(buf, axis=0)
            buf.clear()
            if arr.shape[0] < n_comp:
                pad = np.tile(arr[-1:], (n_comp - arr.shape[0], 1))
                arr = np.vstack([arr, pad])
            ipca.partial_fit(arr)

    for b in wdl_fit:
        m = b["middle_tokens"].float().cpu().numpy()
        for i in range(m.shape[0]):
            buf.append(_fc_upper_tri_vec(m[i]))
            _flush_fc(force=False)
    _flush_fc(force=True)

    # Pass 2: transform + metadata
    feats: List[np.ndarray] = []
    wix: List[int] = []
    rec: List[int] = []
    wdl2 = DataLoader(
        wdl.dataset,
        batch_size=wdl.batch_size,
        shuffle=False,
        num_workers=wdl.num_workers,
    )
    for b in wdl2:
        wix.extend(b["window_index"].cpu().numpy().astype(np.int32).tolist())
        rec.extend(b["subject_index"].cpu().numpy().astype(np.int32).tolist())
        m = b["middle_tokens"].float().cpu().numpy()
        rows = np.stack([_fc_upper_tri_vec(m[i]) for i in range(m.shape[0])], axis=0)
        feats.append(ipca.transform(rows))
    X = np.concatenate(feats, axis=0).astype(np.float32)
    t1 = time.perf_counter()
    w_st = [int(ds0.windows[i][1]) for i in wix]
    tag = "fc"
    np.save(os.path.join(out_dir, f"pooled_{tag}.npy"), X)
    np.save(os.path.join(out_dir, f"rec_idx_{tag}.npy"), np.array(rec, np.int32))
    np.save(os.path.join(out_dir, f"win_start_{tag}.npy"), np.array(w_st, np.int32))
    np.save(os.path.join(out_dir, f"window_index_{tag}.npy"), np.array(wix, np.int32))
    meta = {
        "name": tag,
        "fdim": int(X.shape[1]) if X.size else 0,
        "n_windows": int(X.shape[0]) if X.size else 0,
        "wall_time_sec": float(t1 - t0),
        "data_config": data_config_abs,
        "index_path": data["index_path"],
        "subject_id_per_record": subj,
        "fc_pca_dim": n_comp,
        "fc_pca_fit_windows": len(wdl.dataset) if fit_indices is None else len(fit_indices),
        "fc_raw_dim": d_fc,
        "n_parcels": p,
        "t_window": t_win,
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {tag} {X.shape} (FC upper-tri + PCA->{n_comp}) in {t1 - t0:.1f}s -> {out_dir}")


def _export_pca_two_pass(
    wdl: DataLoader,
    ds0: FMRIMultiScaleWindowDataset,
    out_dir: str,
    data: Dict[str, Any],
    subj: List[str],
    pca_dim: int,
    data_config_abs: str,
    fit_indices: Optional[List[int]] = None,
) -> None:
    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError as e:
        raise SystemExit("export_baseline_pooled --name pca requires scikit-learn") from e

    t0 = time.perf_counter()
    d0 = ds0[0]["middle_tokens"]
    t_win, p = int(d0.shape[0]), int(d0.shape[1])
    d_flat = int(t_win * p)
    n_comp = min(pca_dim, d_flat)
    ipca = IncrementalPCA(n_components=n_comp, batch_size=min(256, max(32, n_comp)))

    wdl_fit = DataLoader(
        wdl.dataset if fit_indices is None else Subset(wdl.dataset, fit_indices),
        batch_size=wdl.batch_size,
        shuffle=False,
        num_workers=wdl.num_workers,
    )
    buf_rows: List[np.ndarray] = []

    def _flush_pca(force: bool = False) -> None:
        nonlocal buf_rows
        while len(buf_rows) >= n_comp:
            chunk = np.stack(buf_rows[:n_comp], axis=0)
            buf_rows = buf_rows[n_comp:]
            ipca.partial_fit(chunk)
        if force and buf_rows:
            arr = np.stack(buf_rows, axis=0)
            buf_rows.clear()
            if arr.shape[0] < n_comp:
                pad = np.tile(arr[-1:], (n_comp - arr.shape[0], 1))
                arr = np.vstack([arr, pad])
            ipca.partial_fit(arr)

    for b in wdl_fit:
        m = b["middle_tokens"].float().cpu().numpy()
        flat = m.reshape(m.shape[0], -1).astype(np.float32)
        for i in range(flat.shape[0]):
            buf_rows.append(flat[i])
            _flush_pca(force=False)
    _flush_pca(force=True)

    feats: List[np.ndarray] = []
    wix: List[int] = []
    rec: List[int] = []
    wdl2 = DataLoader(
        wdl.dataset,
        batch_size=wdl.batch_size,
        shuffle=False,
        num_workers=wdl.num_workers,
    )
    for b in wdl2:
        wix.extend(b["window_index"].cpu().numpy().astype(np.int32).tolist())
        rec.extend(b["subject_index"].cpu().numpy().astype(np.int32).tolist())
        m = b["middle_tokens"].float().cpu().numpy()
        flat = m.reshape(m.shape[0], -1).astype(np.float32)
        feats.append(ipca.transform(flat))
    X = np.concatenate(feats, axis=0).astype(np.float32)
    t1 = time.perf_counter()
    w_st = [int(ds0.windows[i][1]) for i in wix]
    tag = "pca"
    np.save(os.path.join(out_dir, f"pooled_{tag}.npy"), X)
    np.save(os.path.join(out_dir, f"rec_idx_{tag}.npy"), np.array(rec, np.int32))
    np.save(os.path.join(out_dir, f"win_start_{tag}.npy"), np.array(w_st, np.int32))
    np.save(os.path.join(out_dir, f"window_index_{tag}.npy"), np.array(wix, np.int32))
    meta = {
        "name": tag,
        "fdim": int(X.shape[1]) if X.size else 0,
        "n_windows": int(X.shape[0]) if X.size else 0,
        "wall_time_sec": float(t1 - t0),
        "data_config": data_config_abs,
        "index_path": data["index_path"],
        "subject_id_per_record": subj,
        "pca_dim": n_comp,
        "pca_input_dim": d_flat,
        "pca_fit_windows": len(wdl.dataset) if fit_indices is None else len(fit_indices),
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {tag} {X.shape} (parcel BOLD flattened + PCA->{n_comp}) in {t1 - t0:.1f}s -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_config",
        type=str,
        default=os.path.join(REPO, "configs/fmri/data_probe.yaml"),
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--name",
        type=str,
        default="difumo",
        choices=["difumo", "yeo", "fc", "pca"],
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fc_pca_dim", type=int, default=256, help="FC upper-tri PCA output dim (fc mode).")
    p.add_argument("--pca_dim", type=int, default=256, help="Parcel-flatten PCA output dim (pca mode).")
    p.add_argument(
        "--fit_split",
        type=str,
        default=None,
        help="fc/pca modes: fit the PCA only on windows of this split (e.g. train), read from data.subject_split_json. Default: all windows.",
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dcfg = _load_config(args.data_config)
    data = dcfg["data"]
    data = dict(data)
    data_cfg_abs = os.path.abspath(args.data_config)

    ds0 = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=str(data["index_path"]),
        layout_dir=str(data["layout_dir"]),
        window_size=int(data.get("window_size", 32)),
        stride=int(data.get("stride", 16)),
        cache_items=8,
        max_windows_per_subject=data.get("max_windows_per_subject"),
    )
    with open(data["index_path"], "r", encoding="utf-8") as f:
        recs: List[Dict[str, Any]] = json.load(f)
    subj = [infer_subject_id_from_record(r) for r in recs]

    class WDS(Dataset):
        def __init__(self) -> None:
            self.base = ds0

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            d = dict(self.base[i])
            d["window_index"] = torch.tensor(i, dtype=torch.long)
            return d

    wdl = DataLoader(WDS(), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.name in ("difumo", "yeo"):
        _export_difumo_yeo(wdl, ds0, args.name, args.out_dir, data, subj, data_cfg_abs)
    else:
        fit_idx = None
        if args.fit_split:
            fit_idx = window_indices_for_split(ds0, load_split(resolve_split_path(dcfg)), args.fit_split).tolist()
            print(f"[{args.name}] PCA fit on split {args.fit_split!r}: {len(fit_idx)} of {len(ds0)} windows")
        if args.name == "fc":
            _export_fc_two_pass(wdl, ds0, args.out_dir, data, subj, args.fc_pca_dim, data_cfg_abs, fit_idx)
        elif args.name == "pca":
            _export_pca_two_pass(wdl, ds0, args.out_dir, data, subj, args.pca_dim, data_cfg_abs, fit_idx)
        else:
            raise SystemExit(f"unknown --name {args.name!r}")


if __name__ == "__main__":
    main()
