#!/usr/bin/env python3
"""
Paired significance test between two downstream probes, computed from the
per-subject prediction dumps written by linear_probe_hcp.py.

Because both probes are evaluated on the same held-out subjects, the comparison
is paired. Which test applies to which metric:

  classification
    accuracy (thresholded predictions) -- exact McNemar on discordant subjects,
        plus a subject bootstrap CI on the accuracy difference;
    AUROC, macro-F1 -- subject bootstrap CI on the difference plus a paired
        model-swap permutation p-value. McNemar does NOT test AUROC: it only
        sees hard labels, so it cannot say anything about a ranking metric.
  regression
    MAE (and squared error) -- sign-flip permutation on the per-subject paired
        error difference, plus a subject bootstrap CI;
    Pearson r, Spearman rho, R^2 -- subject bootstrap CI plus the same paired
        model-swap permutation (these are not per-subject quantities, so a
        per-subject sign flip is not defined for them).

Permutation p-values use 10k Monte-Carlo draws with the add-one correction, so
the finest resolution is 1/(10000+1) ~ 1e-4 and values are reported as
"p = 9.999e-05" rather than "p < 0.0001".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from downstream.probe_eval import classification_metrics, regression_metrics


def _task_kind(path: str, task: str) -> str:
    z = np.load(path, allow_pickle=True)
    if f"{task}__y_prob" in z:
        return "classification"
    if f"{task}__y_pred" in z:
        return "regression"
    raise SystemExit(f"{path} has no predictions for task {task!r}; re-run the probe.")


def _load(path: str, task: str, kind: str = "classification") -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    fields = (
        ("y_true", "y_pred", "y_prob", "classes", "subject_ids")
        if kind == "classification"
        else ("y_true", "y_pred", "train_mean", "subject_ids")
    )
    keys = [f"{task}__{k}" for k in fields]
    missing = [k for k in keys if k not in z]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; re-run the probe with --predictions_npz.")
    return {k.split("__", 1)[1]: z[k] for k in keys}


def _align(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Index arrays putting both probes on the same subject order."""
    sa = [str(s) for s in a["subject_ids"].tolist()]
    sb = [str(s) for s in b["subject_ids"].tolist()]
    common = sorted(set(sa) & set(sb))
    if not common:
        raise SystemExit("No overlapping subjects between the two prediction files.")
    ia = {s: i for i, s in enumerate(sa)}
    ib = {s: i for i, s in enumerate(sb)}
    return (
        np.array([ia[s] for s in common], dtype=np.int64),
        np.array([ib[s] for s in common], dtype=np.int64),
    )


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar test on the subjects where the two models disagree."""
    b_only = int(np.sum(~correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    n_disc = a_only + b_only
    p = 1.0 if n_disc == 0 else float(binomtest(b_only, n_disc, 0.5).pvalue)
    return {
        "n_b_correct_a_wrong": b_only,
        "n_a_correct_b_wrong": a_only,
        "n_discordant": n_disc,
        "p_value_mcnemar_exact": p,
    }


def bootstrap_paired_delta(
    y: np.ndarray,
    pa: Dict[str, np.ndarray],
    pb: Dict[str, np.ndarray],
    metric: str,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    """Percentile CI and two-sided p-value for metric(B) - metric(A) over subjects."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        ma = classification_metrics(y[idx], pa["y_pred"][idx], pa["y_prob"][idx], pa["classes"])
        mb = classification_metrics(y[idx], pb["y_pred"][idx], pb["y_prob"][idx], pb["classes"])
        if ma[metric] is None or mb[metric] is None:
            continue
        deltas.append(float(mb[metric] - ma[metric]))
    if len(deltas) < 100:
        return {"delta": None, "ci95": None, "p_value_bootstrap_two_sided": None}
    arr = np.asarray(deltas)
    obs_a = classification_metrics(y, pa["y_pred"], pa["y_prob"], pa["classes"])[metric]
    obs_b = classification_metrics(y, pb["y_pred"], pb["y_prob"], pb["classes"])[metric]
    # Two-sided p: fraction of the bootstrap distribution on the far side of zero.
    frac = float(np.mean(arr <= 0.0)) if np.mean(arr) > 0 else float(np.mean(arr >= 0.0))
    return {
        "metric_a": None if obs_a is None else float(obs_a),
        "metric_b": None if obs_b is None else float(obs_b),
        "delta_b_minus_a": None if (obs_a is None or obs_b is None) else float(obs_b - obs_a),
        "delta_bootstrap_mean": float(np.mean(arr)),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_value_bootstrap_two_sided": min(1.0, 2.0 * frac),
        "n_boot_effective": int(len(arr)),
    }


def swap_permutation(
    y: np.ndarray,
    pa: Dict[str, np.ndarray],
    pb: Dict[str, np.ndarray],
    metric_fn,
    n_perm: int,
    seed: int,
) -> Dict[str, object]:
    """Paired model-swap permutation: flip A/B per subject, recompute the delta.

    The sign-flip analogue for a metric that is not a per-subject quantity
    (AUROC, macro-F1, r, R^2). Two-sided, add-one smoothed.
    """
    rng = np.random.default_rng(seed)
    obs_a, obs_b = metric_fn(pa), metric_fn(pb)
    if obs_a is None or obs_b is None:
        return {"p_value_permutation_two_sided": None, "n_perm": 0}
    obs = float(obs_b - obs_a)
    n = len(y)
    n_ge = 0
    keys = [k for k in pa if k != "classes"]
    for _ in range(n_perm):
        flip = rng.random(n) < 0.5
        qa = {k: np.where(flip.reshape((-1,) + (1,) * (pa[k].ndim - 1)), pb[k], pa[k]) for k in keys}
        qb = {k: np.where(flip.reshape((-1,) + (1,) * (pa[k].ndim - 1)), pa[k], pb[k]) for k in keys}
        if "classes" in pa:
            qa["classes"], qb["classes"] = pa["classes"], pb["classes"]
        ma, mb = metric_fn(qa), metric_fn(qb)
        if ma is None or mb is None:
            continue
        if abs(float(mb - ma)) >= abs(obs) - 1e-12:
            n_ge += 1
    return {
        "observed_delta": obs,
        "p_value_permutation_two_sided": float((n_ge + 1) / (n_perm + 1)),
        "n_perm": int(n_perm),
    }


def signflip_paired(d: np.ndarray, n_perm: int, seed: int) -> Dict[str, object]:
    """Sign-flip permutation on a per-subject paired difference (B - A)."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    n = len(d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d.reshape(1, -1)).mean(axis=1)
    n_ge = int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
    return {
        "mean_delta_b_minus_a": obs,
        "p_value_signflip_two_sided": float((n_ge + 1) / (n_perm + 1)),
        "n_perm": int(n_perm),
    }


def bootstrap_mean_delta(d: np.ndarray, n_boot: int, seed: int) -> List[float]:
    """Percentile CI for the mean of a per-subject paired difference."""
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = d[idx].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


REG_CORR_METRICS = ("pearson_r", "spearman_rho", "r2")


def regression_delta_tests(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    train_mean: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    """Bootstrap CI + model-swap permutation for r / rho / R^2, in one pass each."""
    n = len(y)
    obs_a = regression_metrics(y, pred_a, train_mean)
    obs_b = regression_metrics(y, pred_b, train_mean)

    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {m: [] for m in REG_CORR_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma = regression_metrics(y[idx], pred_a[idx], train_mean)
        mb = regression_metrics(y[idx], pred_b[idx], train_mean)
        for m in REG_CORR_METRICS:
            if ma[m] is not None and mb[m] is not None:
                draws[m].append(float(mb[m] - ma[m]))

    rng = np.random.default_rng(seed)
    n_ge: Dict[str, int] = {m: 0 for m in REG_CORR_METRICS}
    for _ in range(n_perm):
        flip = rng.random(n) < 0.5
        qa = np.where(flip, pred_b, pred_a)
        qb = np.where(flip, pred_a, pred_b)
        ma = regression_metrics(y, qa, train_mean)
        mb = regression_metrics(y, qb, train_mean)
        for m in REG_CORR_METRICS:
            if ma[m] is None or mb[m] is None or obs_a[m] is None or obs_b[m] is None:
                continue
            if abs(float(mb[m] - ma[m])) >= abs(float(obs_b[m] - obs_a[m])) - 1e-12:
                n_ge[m] += 1

    out: Dict[str, Dict[str, object]] = {}
    for m in REG_CORR_METRICS:
        arr = np.asarray(draws[m])
        ok = obs_a[m] is not None and obs_b[m] is not None
        out[m] = {
            "metric_a": None if obs_a[m] is None else float(obs_a[m]),
            "metric_b": None if obs_b[m] is None else float(obs_b[m]),
            "delta_b_minus_a": float(obs_b[m] - obs_a[m]) if ok else None,
            "ci95": (
                [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
                if len(arr) >= 100
                else None
            ),
            "p_value_permutation_two_sided": float((n_ge[m] + 1) / (n_perm + 1)) if ok else None,
            "n_boot_effective": int(len(arr)),
            "n_perm": int(n_perm),
            "test": "subject bootstrap CI + paired model-swap permutation",
        }
    return out


def compare_regression(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    ia: np.ndarray,
    ib: np.ndarray,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Dict[str, object]:
    y = np.asarray(a["y_true"], dtype=np.float64)[ia]
    if not np.allclose(y, np.asarray(b["y_true"], dtype=np.float64)[ib]):
        raise SystemExit("Regression targets differ after subject alignment.")
    pa = np.asarray(a["y_pred"], dtype=np.float64)[ia]
    pb = np.asarray(b["y_pred"], dtype=np.float64)[ib]
    tm = float(np.asarray(a["train_mean"]).reshape(-1)[0])
    res: Dict[str, object] = {"n_subjects": int(len(y)), "train_mean": tm}
    # Per-subject error differences: sign-flip is well defined here.
    for name, err in (("mae", np.abs), ("mse", lambda e: e ** 2)):
        d = err(pb - y) - err(pa - y)
        res[name] = {
            "a": float(np.mean(err(pa - y))),
            "b": float(np.mean(err(pb - y))),
            "ci95_mean_delta": bootstrap_mean_delta(d, n_boot, seed),
            **signflip_paired(d, n_perm, seed),
            "test": "sign-flip permutation on per-subject paired error difference",
        }
    res.update(regression_delta_tests(y, pa, pb, tm, n_boot, n_perm, seed))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions_a", type=str, required=True)
    p.add_argument("--predictions_b", type=str, required=True)
    p.add_argument("--name_a", type=str, default="model_a")
    p.add_argument("--name_b", type=str, default="model_b")
    p.add_argument("--tasks", type=str, nargs="+", default=["sex", "age", "flanker", "neuroticism"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", type=str, default="")
    args = p.parse_args()

    out: Dict[str, object] = {
        "model_a": {"name": args.name_a, "predictions": args.predictions_a},
        "model_b": {"name": args.name_b, "predictions": args.predictions_b},
        "tasks": {},
    }
    for task in args.tasks:
        kind = _task_kind(args.predictions_a, task)
        a = _load(args.predictions_a, task, kind)
        b = _load(args.predictions_b, task, kind)
        ia, ib = _align(a, b)
        if kind == "regression":
            res = compare_regression(a, b, ia, ib, args.n_boot, args.n_perm, args.seed)
            out["tasks"][task] = res
            print(f"[{task}] {json.dumps(res, indent=2)}")
            continue
        y = a["y_true"][ia]
        if not np.array_equal(y, b["y_true"][ib]):
            raise SystemExit(f"Label mismatch for task {task!r} after subject alignment.")
        pa = {"y_pred": a["y_pred"][ia], "y_prob": a["y_prob"][ia], "classes": a["classes"]}
        pb = {"y_pred": b["y_pred"][ib], "y_prob": b["y_prob"][ib], "classes": b["classes"]}

        corr_a = (pa["y_pred"] == y).astype(np.float64)
        corr_b = (pb["y_pred"] == y).astype(np.float64)
        res: Dict[str, object] = {
            "n_subjects": int(len(y)),
            "accuracy": {
                **mcnemar(corr_a > 0, corr_b > 0),
                "acc_a": float(np.mean(corr_a)),
                "acc_b": float(np.mean(corr_b)),
                "delta_b_minus_a": float(np.mean(corr_b - corr_a)),
                "ci95_mean_delta": bootstrap_mean_delta(corr_b - corr_a, args.n_boot, args.seed),
                "test": "exact McNemar on thresholded predictions (+ subject bootstrap CI)",
            },
        }
        for metric in ("auroc", "f1_macro"):
            res[metric] = bootstrap_paired_delta(y, pa, pb, metric, args.n_boot, args.seed)
            res[metric].update(
                swap_permutation(
                    y,
                    pa,
                    pb,
                    lambda d, _m=metric: classification_metrics(y, d["y_pred"], d["y_prob"], d["classes"])[_m],
                    args.n_perm,
                    args.seed,
                )
            )
            res[metric]["test"] = (
                "subject bootstrap CI + paired model-swap permutation (McNemar does not test this metric)"
            )
        out["tasks"][task] = res
        print(f"[{task}] {json.dumps(res, indent=2)}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
