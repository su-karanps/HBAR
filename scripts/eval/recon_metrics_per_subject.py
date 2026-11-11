#!/usr/bin/env python3
"""Subject-level reconstruction metrics for many tokenizers in a single data pass.

The paired significance script re-reads the whole split once per comparison, so
covering N models costs N*(N-1) passes over high-dimensional fMRI windows and
data loading dominates. Here every model is resident on the GPU at once and each
batch is scored by all of them, which turns the same coverage into one pass.

Windows are also subsampled per subject: the statistics resample subjects, so a
subject mean over a few dozen windows is already stable well before the full
~74 windows per subject are consumed.

Output is an npz keyed exactly like the dumps from paired_tokenizer_significance,
so scripts/eval/summarize_recon_table.py renders CIs and every pairwise
contrast from it without change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

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

from scripts.eval.paired_tokenizer_significance import (  # noqa: E402
    _batch_corr,
    _batch_mse,
    _load_checkpoint,
    _pool_chunk_series_to_parcel,
    _pool_parcel_to_network,
    _shared_1d,
)

METRICS = ["voxel_mse", "voxel_corr", "parcel_mse", "parcel_corr", "network_mse", "network_corr"]


def _pool_chunks_to_parcel_exact(
    chunk_bntk: torch.Tensor, chunk_indices_nk: torch.Tensor, chunk_parcels_n: torch.Tensor, n_parcels: int
) -> torch.Tensor:
    """Voxel-count-weighted parcel mean of a chunk reconstruction [B,N0,T,K] -> [B,T,P].

    Padded K-slots (chunk_indices == -1) are excluded, and each parcel divides by its
    number of valid voxels, which is how the dataset's `parcel_ts` targets are built.
    The older pooling (mean over all K slots, then an unweighted mean over chunks)
    scores a perfect reconstruction at parcel corr ~0.93-0.97.
    """
    valid = (chunk_indices_nk >= 0).to(chunk_bntk.dtype)                      # [N0,K]
    chunk_sum = (chunk_bntk * valid[None, :, None, :]).sum(dim=-1)            # [B,N0,T]
    num = _pool_chunk_series_to_parcel(chunk_sum, chunk_parcels_n, n_parcels)  # per-parcel sum / n_chunks
    ids = chunk_parcels_n.to(device=chunk_bntk.device, dtype=torch.long)
    n_chunks = torch.bincount(ids, minlength=n_parcels).to(chunk_bntk.dtype).clamp_min(1.0)
    n_vox = torch.bincount(ids, weights=valid.sum(dim=-1), minlength=n_parcels).to(chunk_bntk.dtype).clamp_min(1.0)
    return num * (n_chunks / n_vox).view(1, 1, -1)

# A 32-TR window at TR=0.72s resolves 0.043 Hz steps up to a 0.694 Hz Nyquist,
# so the classic slow-5 band (0.01-0.027 Hz) is below what a single window can
# express and is deliberately absent. "dc" is the within-window mean offset.
BANDS_HZ: Dict[str, tuple] = {
    "dc": (0.0, 0.02),
    "low": (0.02, 0.10),
    "mid": (0.10, 0.20),
    "resp": (0.20, 0.35),
    "high": (0.35, 0.70),
}


def band_residual_stats(
    pred: torch.Tensor, true: torch.Tensor, tr: float, prefix: str
) -> Dict[str, np.ndarray]:
    """Per-window residual power by frequency band, for [B, T, X] tensors.

    Reports two views per band: `loss` is residual power over ground-truth power
    within the band, i.e. the fraction of that band's signal the tokenizer fails
    to reproduce; `share` is the band's fraction of total residual power, i.e.
    where the error concentrates.
    """
    resid = (pred - true).float()
    sig = true.float()
    n_t = resid.shape[1]
    R = torch.fft.rfft(resid, dim=1)
    S = torch.fft.rfft(sig, dim=1)
    r_pow = R.real.pow(2) + R.imag.pow(2)
    s_pow = S.real.pow(2) + S.imag.pow(2)
    freqs = torch.fft.rfftfreq(n_t, d=tr).to(resid.device)

    total_r = r_pow.sum(dim=(1, 2)).clamp_min(1e-12)
    total_s = s_pow.sum(dim=(1, 2)).clamp_min(1e-12)
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS_HZ.items():
        m = (freqs >= lo) & (freqs < hi)
        if not bool(m.any()):
            continue
        rp = r_pow[:, m, :].sum(dim=(1, 2))
        sp = s_pow[:, m, :].sum(dim=(1, 2))
        out[f"{prefix}_loss_{name}"] = (rp / sp.clamp_min(1e-12)).detach().cpu().numpy().astype(np.float64)
        out[f"{prefix}_share_{name}"] = (rp / total_r).detach().cpu().numpy().astype(np.float64)
        # How much ground-truth power the band carries at all: `loss` is a ratio
        # against this, so it inflates without bound in near-empty bands.
        out[f"{prefix}_sigshare_{name}"] = (sp / total_s).detach().cpu().numpy().astype(np.float64)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME:CONFIG:CKPT",
        help="Repeatable. Colon-separated name, model config path, checkpoint path.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--windows_per_subject",
        type=int,
        default=16,
        help="Cap windows contributing to each subject mean (0 = use all).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds (HCP-YA rfMRI = 0.72).")
    p.add_argument("--no_bands", action="store_true", help="Skip the frequency-band residual decomposition.")
    p.add_argument("--output_npz", type=str, required=True)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument(
        "--autocast_models",
        type=str,
        default="",
        help="Comma list of model names whose forward runs under bf16 autocast (the training/export scope).",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)

    specs = []
    for entry in a.model:
        parts = entry.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--model must be NAME:CONFIG:CKPT, got: {entry}")
        specs.append(tuple(parts))
    autocast_models = {n for n in a.autocast_models.split(",") if n}
    unknown = autocast_models - {s[0] for s in specs}
    if unknown:
        raise SystemExit(f"--autocast_models names not in --model: {sorted(unknown)}")
    if autocast_models:
        print(f"[autocast] bf16 forward for: {sorted(autocast_models)}", flush=True)

    dcfg = _load_config(a.data_config)
    ds = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=dcfg["data"]["index_path"],
        layout_dir=dcfg["data"]["layout_dir"],
        window_size=int(dcfg["data"].get("window_size", 32)),
        stride=int(dcfg["data"].get("stride", 16)),
        cache_items=int(dcfg["data"].get("cache_items", 8)),
        random_sampling=False,
        max_windows_per_subject=dcfg["data"].get("max_windows_per_subject"),
    )

    with open(dcfg["data"]["index_path"], "r", encoding="utf-8") as f:
        records = json.load(f)
    subj_by_record = [infer_subject_id_from_record(r) for r in records]

    if a.split == "all":
        window_ids = list(range(len(ds)))
        split_desc = "all subjects (INCLUDES TRAINING DATA)"
    else:
        split = load_split(resolve_split_path(dcfg))
        window_ids = window_indices_for_split(ds, split, a.split).tolist()
        split_desc = f"{a.split}: {len(split[a.split])} subjects"

    rng = np.random.default_rng(a.seed)
    rng.shuffle(window_ids)
    if a.windows_per_subject > 0:
        by_subject: Dict[str, List[int]] = defaultdict(list)
        for wid in window_ids:
            sid = str(subj_by_record[int(ds.windows[wid][0])])
            if len(by_subject[sid]) < a.windows_per_subject:
                by_subject[sid].append(wid)
        window_ids = [w for wl in by_subject.values() for w in wl]
        rng.shuffle(window_ids)
    print(f"[split] {split_desc} / {len(window_ids)} windows over {len(specs)} models", flush=True)

    eval_set = Subset(ds, window_ids)
    loader = DataLoader(
        eval_set, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True
    )
    sample = next(iter(loader))
    dev = torch.device(a.device)

    # Oracle: the true chunks must pool exactly onto the stored parcel/network targets.
    p_or = _pool_chunks_to_parcel_exact(
        sample["bottom_tokens"].float(), _shared_1d(sample["chunk_indices"]),
        _shared_1d(sample["token_parcels"]), n_parcels=sample["middle_tokens"].shape[-1],
    )
    n_or = _pool_parcel_to_network(
        sample["middle_tokens"].float(), _shared_1d(sample["parcel_to_network"]),
        n_networks=sample["top_tokens"].shape[-1],
    )
    # Checked numerically, not by correlation, so a scale error cannot pass. Targets are stored
    # in fp16 (|x| <~ 5, half-ulp ~2e-3), which sets the max|err| tolerance.
    for lvl, pred, true in (("parcel", p_or, sample["middle_tokens"].float()), ("network", n_or, sample["top_tokens"].float())):
        e = pred - true
        mse, mx = float(e.pow(2).mean()), float(e.abs().max())
        scale = float((pred * true).sum() / (pred * pred).sum())
        print(f"[oracle] {lvl}: MSE {mse:.2e}  max|err| {mx:.2e}  LS scale {scale:.6f}", flush=True)
        assert mse < 1e-7 and mx < 4e-3 and abs(scale - 1.0) < 1e-4, f"{lvl} pooling does not reproduce the dataset targets"
    # Parcels with no voxels in the layout (14 in DiFuMo-1024, all in network 17 'No network found')
    # have an all-zero target; the pooled prediction is exactly 0 for every model, so they add zero
    # error to MSE and identical constants to the correlation for every model.
    ci0, cp0 = _shared_1d(sample["chunk_indices"]), _shared_1d(sample["token_parcels"])
    n_vox = torch.bincount(cp0, weights=(ci0 >= 0).sum(dim=-1).float(), minlength=sample["middle_tokens"].shape[-1])
    empty = n_vox == 0
    assert float(sample["middle_tokens"][..., empty].abs().max() if empty.any() else 0.0) == 0.0, "empty parcel with nonzero target"
    print(f"[oracle] {int(empty.sum())} parcels have no voxels (target and pooled value both 0)", flush=True)

    models: Dict[str, torch.nn.Module] = {}
    for name, cfg_path, ckpt in specs:
        m = _build_model(_load_config(cfg_path), sample).to(dev).eval()
        _load_checkpoint(m, ckpt)
        models[name] = m
        print(f"[model] {name} <- {ckpt}", flush=True)

    # model -> metric -> subject -> list of per-window values
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {n: defaultdict(lambda: defaultdict(list)) for n in models}
    # Flat per-window values, so within-subject spread can be examined and not
    # just the subject means the bootstrap consumes.
    window_rows: Dict[str, Dict[str, List[float]]] = {n: defaultdict(list) for n in models}
    window_subjects: List[str] = []

    t0 = time.time()
    w0 = 0
    for bi, batch in enumerate(loader):
        bsz = int(batch["voxel_window"].shape[0])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev, non_blocking=True)

        sids = [str(subj_by_record[int(ds.windows[window_ids[w0 + i]][0])]) for i in range(bsz)]

        for name, model in models.items():
            with torch.no_grad():
                if name in autocast_models:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(batch)
                else:
                    out = model(batch)
            if "parcel_ts_recon" in out:
                bm: Dict[str, np.ndarray] = {k: np.full((bsz,), np.nan, dtype=np.float64) for k in METRICS}
                p_true = batch["middle_tokens"].float()
                bm["parcel_mse"] = _batch_mse(out["parcel_ts_recon"], p_true)
                bm["parcel_corr"] = _batch_corr(out["parcel_ts_recon"], p_true)
            else:
                v_true = batch["voxel_window"].float()
                v_pred = out["voxel_recon"].float()
                chunk_parcels = _shared_1d(batch["token_parcels"])
                p_pred = _pool_chunks_to_parcel_exact(
                    out["chunk_recon"].float(), _shared_1d(batch["chunk_indices"]), chunk_parcels,
                    n_parcels=batch["middle_tokens"].shape[-1],
                )
                p_true = batch["middle_tokens"].float()
                n_pred = _pool_parcel_to_network(
                    p_pred, _shared_1d(batch["parcel_to_network"]), n_networks=batch["top_tokens"].shape[-1]
                )
                n_true = batch["top_tokens"].float()
                bm = {
                    "voxel_mse": _batch_mse(v_pred, v_true),
                    "voxel_corr": _batch_corr(v_pred, v_true),
                    "parcel_mse": _batch_mse(p_pred, p_true),
                    "parcel_corr": _batch_corr(p_pred, p_true),
                    "network_mse": _batch_mse(n_pred, n_true),
                    "network_corr": _batch_corr(n_pred, n_true),
                }
                if not a.no_bands:
                    bm.update(band_residual_stats(v_pred, v_true, a.tr, "voxel"))
                    bm.update(band_residual_stats(p_pred, p_true, a.tr, "parcel"))
            for i, sid in enumerate(sids):
                for k, v in bm.items():
                    acc[name][k][sid].append(float(v[i]))
                    window_rows[name][k].append(float(v[i]))

            # Activations for voxel-resolution windows dominate memory, and all
            # models share one GPU, so release each model's before the next runs.
            out = None
            del out
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        window_subjects.extend(sids)
        w0 += bsz
        if bi % 20 == 0:
            el = time.time() - t0
            rate = w0 / max(el, 1e-6)
            eta = (len(window_ids) - w0) / max(rate, 1e-6)
            print(
                f"[{w0}/{len(window_ids)}] {rate:.1f} win/s  elapsed={el/60:.1f}m  eta={eta/60:.1f}m",
                flush=True,
            )

    payload: Dict[str, Any] = {}
    summary: Dict[str, Any] = {
        "split": split_desc,
        "windows": len(window_ids),
        "tr_seconds": a.tr,
        "bands_hz": {k: list(v) for k, v in BANDS_HZ.items()},
        "models": {},
    }
    all_keys = sorted({k for n in models for k in acc[n].keys()})
    for k in all_keys:
        present = [n for n in models if acc[n].get(k)]
        if not present:
            continue
        shared = sorted(set.intersection(*[set(acc[n][k].keys()) for n in present]))
        payload[f"subjects__{k}"] = np.asarray(shared, dtype=object)
        for name in present:
            vals = np.asarray([np.nanmean(acc[name][k][s]) for s in shared], dtype=np.float64)
            stds = np.asarray([np.nanstd(acc[name][k][s]) for s in shared], dtype=np.float64)
            payload[f"{name}__{k}"] = vals
            payload[f"{name}__{k}__within_subject_std"] = stds
            finite = vals[np.isfinite(vals)]
            summary["models"].setdefault(name, {})[k] = (
                float(np.mean(finite)) if finite.size else None
            )

    payload["window_subjects"] = np.asarray(window_subjects, dtype=object)
    for name in models:
        for k, vals in window_rows[name].items():
            payload[f"window__{name}__{k}"] = np.asarray(vals, dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(a.output_npz)), exist_ok=True)
    np.savez(a.output_npz, **payload)
    print(f"Saved: {a.output_npz}")
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {a.output_json}")
    print(f"Total: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
