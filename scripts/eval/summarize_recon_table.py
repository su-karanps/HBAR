#!/usr/bin/env python3
"""Render Table 1 (reconstruction) with per-model bootstrap CIs and paired tests.

Consumes the `*_per_subject.npz` dumps written by
scripts/eval/paired_tokenizer_significance.py, which hold subject-level means
of each reconstruction metric. Because those are per-subject, both the per-model
CIs and every pairwise contrast can be recomputed here without touching a GPU.

Subjects are the resampling unit throughout, so the intervals reflect
between-subject variability rather than between-window variability.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations
from typing import Any, Dict, List

import numpy as np

HIGHER_BETTER = {"voxel_corr", "parcel_corr", "network_corr"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        nargs="+",
        required=True,
        help="Paths or globs for *_per_subject.npz files.",
    )
    p.add_argument("--metrics", nargs="+", default=["voxel_corr", "parcel_corr", "network_corr"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairwise", action="store_true", help="Also print all pairwise tests.")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def collect(paths: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """model -> metric -> {subject_id: value}."""
    store: Dict[str, Dict[str, Dict[str, float]]] = {}
    files: List[str] = []
    for pat in paths:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        files.extend(hits)
    if not files:
        raise SystemExit("No npz files matched.")
    for path in files:
        if not os.path.exists(path):
            print(f"[warn] missing, skipping: {path}")
            continue
        z = np.load(path, allow_pickle=True)
        subj_keys = [k for k in z.files if k.startswith("subjects__")]
        for sk in subj_keys:
            metric = sk[len("subjects__"):]
            subjects = [str(s) for s in z[sk]]
            for key in z.files:
                if key.startswith("subjects__") or not key.endswith(f"__{metric}"):
                    continue
                model = key[: -(len(metric) + 2)]
                vals = np.asarray(z[key], dtype=np.float64)
                if vals.shape[0] != len(subjects):
                    continue
                store.setdefault(model, {}).setdefault(metric, {})
                for s, v in zip(subjects, vals.tolist()):
                    store[model][metric][s] = v
        print(f"[read] {path}")
    return store


def boot_ci(x: np.ndarray, n_boot: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = np.nanmean(x[idx], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def signflip_p(delta: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(delta)))
    signs = rng.choice((-1.0, 1.0), size=(n_perm, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def main() -> None:
    a = parse_args()
    store = collect(a.npz)
    models = sorted(store.keys())
    result: Dict[str, Any] = {"per_model": {}, "pairwise": {}}

    for mi, metric in enumerate(a.metrics):
        rows = [m for m in models if metric in store.get(m, {})]
        if not rows:
            continue
        print(f"\n### {metric}   (higher is better)" if metric in HIGHER_BETTER else f"\n### {metric}   (lower is better)")
        width = max(len(m) for m in rows) + 2
        print("model".ljust(width) + "mean [95% CI]".ljust(34) + "n_subj")
        print("-" * (width + 42))
        for m in rows:
            vals = np.asarray(list(store[m][metric].values()), dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            lo, hi = boot_ci(vals, a.n_boot, a.seed + 13 * (mi + 1))
            mean = float(np.mean(vals))
            print(m.ljust(width) + f"{mean:.4f} [{lo:.4f}, {hi:.4f}]".ljust(34) + str(vals.size))
            result["per_model"].setdefault(metric, {})[m] = {
                "mean": mean,
                "ci95": [lo, hi],
                "n_subjects": int(vals.size),
            }

        if not a.pairwise:
            continue
        print()
        for m_a, m_b in combinations(rows, 2):
            shared = sorted(set(store[m_a][metric]).intersection(store[m_b][metric]))
            if len(shared) < 2:
                continue
            va = np.asarray([store[m_a][metric][s] for s in shared], dtype=np.float64)
            vb = np.asarray([store[m_b][metric][s] for s in shared], dtype=np.float64)
            ok = np.isfinite(va) & np.isfinite(vb)
            delta = vb[ok] - va[ok]
            lo, hi = boot_ci(delta, a.n_boot, a.seed + 7 * (mi + 1))
            p = signflip_p(delta, a.n_perm, a.seed + 101 * (mi + 1))
            star = "*" if p < 0.05 else " "
            print(
                f"  {m_b} - {m_a}: {np.mean(delta):+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}]  p={p:.4f}{star}  (n={int(ok.sum())})"
            )
            result["pairwise"].setdefault(metric, {})[f"{m_b}_minus_{m_a}"] = {
                "mean_delta": float(np.mean(delta)),
                "ci95": [lo, hi],
                "p_signflip": p,
                "n_subjects": int(ok.sum()),
            }

    if a.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.output_json)), exist_ok=True)
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {a.output_json}")


if __name__ == "__main__":
    main()
