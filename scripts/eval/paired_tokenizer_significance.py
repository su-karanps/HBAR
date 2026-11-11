#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset
from data.fmri_multiscale.splits import load_split, resolve_split_path, window_indices_for_split
from downstream.hcp_metadata import infer_subject_id_from_record
from train_fmri import _build_model, _load_config


def _shared_1d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 1 else x[0]


def _pool_chunk_series_to_parcel(chunk_series_bnt: torch.Tensor, chunk_parcels_n: torch.Tensor, n_parcels: int) -> torch.Tensor:
    b, _, t = chunk_series_bnt.shape
    ids = chunk_parcels_n.to(device=chunk_series_bnt.device, dtype=torch.long)
    out = torch.zeros((b, t, n_parcels), device=chunk_series_bnt.device, dtype=chunk_series_bnt.dtype)
    idx = ids.view(1, 1, -1).expand(b, t, -1)
    out.scatter_add_(dim=2, index=idx, src=chunk_series_bnt.permute(0, 2, 1))
    cnt = torch.bincount(ids, minlength=n_parcels).to(device=out.device, dtype=out.dtype).clamp_min(1.0)
    return out / cnt.view(1, 1, -1)


def _pool_parcel_to_network(parcel_btp: torch.Tensor, parcel_to_network: torch.Tensor, n_networks: int) -> torch.Tensor:
    b, t, p = parcel_btp.shape
    ids = parcel_to_network.to(device=parcel_btp.device, dtype=torch.long)
    out = torch.zeros((b, t, n_networks), device=parcel_btp.device, dtype=parcel_btp.dtype)
    idx = ids.view(1, 1, p).expand(b, t, p)
    out.scatter_add_(dim=2, index=idx, src=parcel_btp)
    cnt = torch.bincount(ids, minlength=n_networks).to(device=out.device, dtype=out.dtype).clamp_min(1.0)
    return out / cnt.view(1, 1, -1)


def _batch_mse(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    d = (pred - true).reshape(pred.shape[0], -1)
    return d.pow(2).mean(dim=1).detach().cpu().numpy().astype(np.float64)


def _batch_corr(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1).float()
    t = true.reshape(true.shape[0], -1).float()
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.norm(p, dim=1) * torch.norm(t, dim=1)
    corr = num / torch.clamp(den, min=1e-8)
    return corr.detach().cpu().numpy().astype(np.float64)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        name = k
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        cleaned[name] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[ckpt] warning: missing keys {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[ckpt] warning: unexpected keys {len(unexpected)} (first 5: {unexpected[:5]})")


def _bootstrap_ci_mean(delta: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = delta.shape[0]
    draws = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(np.mean(delta[idx]))
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lo, hi


def _permutation_signflip_p(delta: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(delta)))
    n = delta.shape[0]
    ge = 1  # add-one smoothing
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n, replace=True)
        m = abs(float(np.mean(delta * signs)))
        if m >= obs:
            ge += 1
    return ge / float(n_perm + 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired significance test for tokenizer reconstruction metrics.")
    p.add_argument("--data_config", type=str, required=True)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Subject split to test on. Anything but 'all' requires data.subject_split_json.",
    )
    p.add_argument("--model_a_name", type=str, required=True)
    p.add_argument("--model_a_config", type=str, required=True)
    p.add_argument("--model_a_ckpt", type=str, required=True)
    p.add_argument("--model_b_name", type=str, required=True)
    p.add_argument("--model_b_config", type=str, required=True)
    p.add_argument("--model_b_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dcfg = _load_config(a.data_config)
    ds = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=dcfg["data"]["index_path"],
        layout_dir=dcfg["data"]["layout_dir"],
        window_size=int(dcfg["data"].get("window_size", 32)),
        stride=int(dcfg["data"].get("stride", 16)),
        cache_items=int(dcfg["data"].get("cache_items", 8)),
        random_sampling=bool(dcfg["data"].get("random_sampling", False)),
        max_windows_per_subject=dcfg["data"].get("max_windows_per_subject"),
    )
    if a.split == "all":
        eval_set: Any = ds
        window_ids = list(range(len(ds)))
        split_desc = "all subjects (INCLUDES TRAINING DATA)"
    else:
        split = load_split(resolve_split_path(dcfg))
        window_ids = window_indices_for_split(ds, split, a.split).tolist()
        # Windows are enumerated subject-by-subject, so truncating at --max_batches
        # would sample only the first few subjects. Shuffle so the paired test
        # spreads its budget across every subject in the split.
        np.random.default_rng(a.seed).shuffle(window_ids)
        eval_set = Subset(ds, window_ids)
        split_desc = f"{a.split}: {len(split[a.split])} subjects / {len(window_ids)} windows"
    print(f"[split] {split_desc}")
    loader = DataLoader(
        eval_set, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True
    )
    sample = next(iter(loader))
    dev = torch.device(a.device)

    cfg_a = _load_config(a.model_a_config)
    cfg_b = _load_config(a.model_b_config)
    model_a = _build_model(cfg_a, sample).to(dev).eval()
    model_b = _build_model(cfg_b, sample).to(dev).eval()
    _load_checkpoint(model_a, a.model_a_ckpt)
    _load_checkpoint(model_b, a.model_b_ckpt)

    with open(dcfg["data"]["index_path"], "r", encoding="utf-8") as f:
        records = json.load(f)
    subj_by_record = [infer_subject_id_from_record(r) for r in records]

    metrics = ["voxel_mse", "voxel_corr", "parcel_mse", "parcel_corr", "network_mse", "network_corr"]
    per_subj_a: Dict[str, Dict[str, List[float]]] = {}
    per_subj_b: Dict[str, Dict[str, List[float]]] = {}

    w0 = 0
    for bi, batch in enumerate(loader):
        if bi >= a.max_batches:
            break
        bsz = int(batch["voxel_window"].shape[0])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev, non_blocking=True)

        with torch.no_grad():
            out_a = model_a(batch)
            out_b = model_b(batch)

        def _batch_metrics(out: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
            if "parcel_ts_recon" in out:
                # npq_parcel only supports parcel reconstruction.
                m: Dict[str, np.ndarray] = {k: np.full((bsz,), np.nan, dtype=np.float64) for k in metrics}
                p_true = batch["middle_tokens"].float()
                p_pred = out["parcel_ts_recon"]
                m["parcel_mse"] = _batch_mse(p_pred, p_true)
                m["parcel_corr"] = _batch_corr(p_pred, p_true)
                return m

            v_true = batch["voxel_window"].float()
            v_pred = out["voxel_recon"].float()
            chunk_scalar = out["chunk_recon"].float().mean(dim=-1)
            chunk_parcels = _shared_1d(batch["token_parcels"])
            p_pred = _pool_chunk_series_to_parcel(chunk_scalar, chunk_parcels, n_parcels=batch["middle_tokens"].shape[-1])
            p_true = batch["middle_tokens"].float()
            parcel_to_network = _shared_1d(batch["parcel_to_network"])
            n_pred = _pool_parcel_to_network(p_pred, parcel_to_network, n_networks=batch["top_tokens"].shape[-1])
            n_true = batch["top_tokens"].float()

            return {
                "voxel_mse": _batch_mse(v_pred, v_true),
                "voxel_corr": _batch_corr(v_pred, v_true),
                "parcel_mse": _batch_mse(p_pred, p_true),
                "parcel_corr": _batch_corr(p_pred, p_true),
                "network_mse": _batch_mse(n_pred, n_true),
                "network_corr": _batch_corr(n_pred, n_true),
            }

        bm_a = _batch_metrics(out_a)
        bm_b = _batch_metrics(out_b)

        for i in range(bsz):
            wi = w0 + i
            rec_idx = int(ds.windows[window_ids[wi]][0])  # type: ignore[index]
            sid = str(subj_by_record[rec_idx])
            if sid not in per_subj_a:
                per_subj_a[sid] = {k: [] for k in metrics}
                per_subj_b[sid] = {k: [] for k in metrics}
            for k in metrics:
                per_subj_a[sid][k].append(float(bm_a[k][i]))
                per_subj_b[sid][k].append(float(bm_b[k][i]))
        w0 += bsz

    common_subjects = sorted(set(per_subj_a.keys()).intersection(per_subj_b.keys()))
    if not common_subjects:
        raise SystemExit("No overlapping subjects found for paired comparison.")

    out: Dict[str, Any] = {
        "model_a": {"name": a.model_a_name, "config": a.model_a_config, "ckpt": a.model_a_ckpt},
        "model_b": {"name": a.model_b_name, "config": a.model_b_config, "ckpt": a.model_b_ckpt},
        "split": {"name": a.split, "description": split_desc},
        "n_subjects": len(common_subjects),
        "n_windows_processed": int(w0),
        "max_batches": int(a.max_batches),
        "tests": {},
    }

    higher_better = {"voxel_corr", "parcel_corr", "network_corr"}
    per_subject_dump: Dict[str, Any] = {}
    for j, k in enumerate(metrics):
        vals_a = np.array([np.nanmean(per_subj_a[s][k]) for s in common_subjects], dtype=np.float64)
        vals_b = np.array([np.nanmean(per_subj_b[s][k]) for s in common_subjects], dtype=np.float64)
        valid = np.isfinite(vals_a) & np.isfinite(vals_b)
        vals_a = vals_a[valid]
        vals_b = vals_b[valid]
        if vals_a.size < 2:
            out["tests"][k] = {"error": "not enough valid paired subject values"}
            continue

        delta = vals_b - vals_a
        ci_lo, ci_hi = _bootstrap_ci_mean(delta, n_boot=a.n_boot, seed=a.seed + 17 * (j + 1))
        p_val = _permutation_signflip_p(delta, n_perm=a.n_perm, seed=a.seed + 101 * (j + 1))
        # Per-model CIs as well as the paired-delta CI, so Table 1 cells can
        # carry their own uncertainty and not just the between-model contrast.
        a_lo, a_hi = _bootstrap_ci_mean(vals_a, n_boot=a.n_boot, seed=a.seed + 31 * (j + 1))
        b_lo, b_hi = _bootstrap_ci_mean(vals_b, n_boot=a.n_boot, seed=a.seed + 37 * (j + 1))
        eff_dir = "positive favors model_b" if k in higher_better else "negative favors model_b"
        out["tests"][k] = {
            "mean_model_a": float(np.mean(vals_a)),
            "mean_model_b": float(np.mean(vals_b)),
            "bootstrap_95ci_model_a": [a_lo, a_hi],
            "bootstrap_95ci_model_b": [b_lo, b_hi],
            "mean_delta_b_minus_a": float(np.mean(delta)),
            "bootstrap_95ci_delta": [ci_lo, ci_hi],
            "p_value_signflip_two_sided": float(p_val),
            "n_subjects_valid": int(vals_a.size),
            "interpretation_direction": eff_dir,
        }
        per_subject_dump[f"{a.model_a_name}__{k}"] = vals_a
        per_subject_dump[f"{a.model_b_name}__{k}"] = vals_b
        per_subject_dump[f"subjects__{k}"] = np.asarray(
            [s for s, ok in zip(common_subjects, valid.tolist()) if ok], dtype=object
        )

    os.makedirs(os.path.dirname(os.path.abspath(a.output_json)), exist_ok=True)
    with open(a.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {a.output_json}")

    npz_path = os.path.splitext(os.path.abspath(a.output_json))[0] + "_per_subject.npz"
    np.savez(npz_path, **per_subject_dump)
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()

