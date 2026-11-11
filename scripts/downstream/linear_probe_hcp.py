#!/usr/bin/env python3
"""
Linear / ridge / logistic probe on *subject-level* mean pooled features vs HCP
age (regression) and sex (binary classification).  Expects
`pooled_*.npy` + `rec_idx_*.npy` + `meta_*.json` (from tokenization or baselines).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from downstream.hcp_metadata import _norm_key, load_hcp_csv, load_subject_columns
from downstream.probe_eval import (
    ALPHA_GRID,
    C_GRID,
    evaluate_regression_task,
    evaluate_task,
    save_predictions,
)


def _aggregate(
    rec_idx: np.ndarray, X: np.ndarray, subj_list: List[str]
) -> Tuple[List[str], np.ndarray]:
    """Map record index -> subject id string, mean over windows."""
    bucket: Dict[int, List[np.ndarray]] = defaultdict(list)
    for r, x in zip(rec_idx.tolist(), X):
        bucket[int(r)].append(x)
    sids: List[str] = []
    rows: List[np.ndarray] = []
    for r in sorted(bucket):
        sids.append(subj_list[int(r)] if int(r) < len(subj_list) else f"r{r}")
        rows.append(np.stack(bucket[int(r)]).mean(axis=0).astype(np.float32))
    return sids, np.stack(rows, axis=0) if rows else (sids, np.zeros((0, 0), np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pooled_npy",
        type=str,
        required=True,
        help="Per-window float features (N_win, D), e.g. .../pooled_m0vqvae.npy",
    )
    p.add_argument("--rec_idx_npy", type=str, default=None, help="Defaults to sibling rec_idx_*.npy if omitted.")
    p.add_argument("--meta_json", type=str, default=None, help="Defaults to meta_*.json next to pooled if possible.")
    p.add_argument(
        "--hcp_metadata_csv",
        type=str,
        default=os.path.join(REPO, "data", "hcp_metadata.csv"),
    )
    p.add_argument("--id_column", type=str, default="subject_id")
    p.add_argument("--age_column", type=str, default="age")
    p.add_argument("--sex_column", type=str, default="sex")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--subject_split_json",
        type=str,
        default="",
        help=(
            "Path to a JSON split file. Canonical splits have train/val/test keys "
            "(see scripts/make_subject_split.py); legacy files with only a \"val\" key "
            "are still accepted. If the path does not exist, a random split is "
            "generated and saved there so subsequent models reuse the same subjects."
        ),
    )
    p.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which held-out split of a canonical split file to report on.",
    )
    p.add_argument("--n_boot", type=int, default=10000, help="Bootstrap resamples for CIs (0 to disable).")
    p.add_argument(
        "--probe_c",
        type=str,
        default="selected",
        choices=["selected", "fixed"],
        help=(
            "'selected' (protocol v2): logistic C chosen by inner stratified CV on the training "
            "subjects only, from a fixed grid shared by every representation. 'fixed' reproduces "
            "a fixed C=1 probe. Either way the C=1 result is recorded under "
            "task['fixed_C_reference']."
        ),
    )
    p.add_argument(
        "--regression_tasks",
        type=str,
        nargs="*",
        default=["flanker", "neuroticism"],
        help=(
            "Prespecified continuous endpoints, reported regardless of outcome "
            "(flanker = primary cognitive, neuroticism = secondary behavioural). "
            "Pass with no values to disable."
        ),
    )
    p.add_argument("--inner_cv_folds", type=int, default=5, help="Folds of the inner selection CV (train only).")
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--cv_repeats", type=int, default=5, help="Repeated stratified K-fold repeats (0 to disable).")
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Path to write metrics (optional).",
    )
    p.add_argument(
        "--predictions_npz",
        type=str,
        default="",
        help="Path to write per-subject y_true/y_pred/y_prob (optional but recommended).",
    )
    args = p.parse_args()

    Xw = np.load(args.pooled_npy)
    base, tag = os.path.split(args.pooled_npy)
    if args.rec_idx_npy is None:
        m = [n for n in os.listdir(base) if n.startswith("rec_idx_") and n.endswith(".npy")]
        if len(m) == 1:
            args.rec_idx_npy = os.path.join(base, m[0])
        else:
            args.rec_idx_npy = args.pooled_npy.replace("pooled_", "rec_idx_")
    ridx = np.load(args.rec_idx_npy)
    assert ridx.shape[0] == Xw.shape[0]
    mpath = args.meta_json
    if not mpath:
        cand = [n for n in os.listdir(base) if n.startswith("meta_") and n.endswith(".json")]
        mpath = os.path.join(base, cand[0]) if len(cand) == 1 else None
    if not mpath or not os.path.isfile(mpath):
        raise SystemExit("Need --meta_json with subject_id_per_record, or a unique meta_*.json next to pooled file.")
    with open(mpath, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)
    subj = meta.get("subject_id_per_record") or []
    if not subj and "index_path" in meta:
        with open(str(meta["index_path"]), "r", encoding="utf-8") as f:
            idx = json.load(f)
        from downstream.hcp_metadata import infer_subject_id_from_record
        subj = [infer_subject_id_from_record(r) for r in idx]

    sids, X = _aggregate(ridx, Xw, subj)
    pheno = load_hcp_csv(
        args.hcp_metadata_csv,
        id_column=args.id_column,
        age_column=args.age_column,
        sex_column=args.sex_column,
    )
    m = { _norm_key(p.subject_id): (p.age, p.sex) for p in pheno }
    cont = (
        load_subject_columns(args.hcp_metadata_csv, list(args.regression_tasks), id_column=args.id_column)
        if args.regression_tasks
        else {}
    )
    y_age, y_sex, ok = [], [], []
    y_cont: Dict[str, List[float]] = {t: [] for t in args.regression_tasks}
    for i, s in enumerate(sids):
        k = _norm_key(s)
        if k not in m:
            for alt in (s, s.lstrip("0"), "HCP" + s):
                if _norm_key(alt) in m:
                    k = _norm_key(alt)
                    break
        if k not in m:
            continue
        ag, se = m[k]
        y_age.append(ag)
        y_sex.append(int(se))
        for t in args.regression_tasks:
            y_cont[t].append(float(cont.get(k, {}).get(t, float("nan"))))
        ok.append(i)
    if not ok:
        raise SystemExit("No label overlap between pooled feature subjects and HCP table.")
    Xn = X[np.array(ok, dtype=np.int32)]
    y_age = np.array([y_age[i] for i in range(len(y_age))], np.float32)
    y_sex = np.array(y_sex, np.int64)
    y_reg = {t: np.array(v, np.float64) for t, v in y_cont.items()}

    n = Xn.shape[0]
    norm_sids = [_norm_key(s) for s in [sids[i] for i in ok]]
    if args.subject_split_json and os.path.isfile(args.subject_split_json):
        with open(args.subject_split_json) as f:
            split_data = json.load(f)
        if not isinstance(split_data, dict):
            raise SystemExit(f"Malformed split file: {args.subject_split_json}")
        if "train" in split_data and "test" in split_data:
            # Canonical tokenizer split: probe trains on the same subjects the
            # tokenizer trained on and reports on subjects it never saw.
            train_set = {_norm_key(str(s)) for s in split_data["train"]}
            eval_set = {_norm_key(str(s)) for s in split_data[args.eval_split]}
            split_kind = f"canonical/{args.eval_split}"
        else:
            train_set = None
            eval_set = {_norm_key(str(s)) for s in split_data.get("val", [])}
            split_kind = "legacy/val"
        if train_set is not None:
            tr = [i for i, s in enumerate(norm_sids) if s in train_set]
        else:
            tr = [i for i, s in enumerate(norm_sids) if s not in eval_set]
        va = [i for i, s in enumerate(norm_sids) if s in eval_set]
        if not va:
            raise SystemExit(
                f"subject_split_json {args.eval_split!r} set had no overlap with the pooled features."
            )
        print(f"[split] {split_kind}: train={len(tr)} eval={len(va)} subjects", flush=True)
    else:
        split_kind = "random"
        rng = np.random.default_rng(args.seed)
        n = Xn.shape[0]
        perm = rng.permutation(n)
        n_v = max(1, int(round(n * args.val_fraction)))
        val_set = set(norm_sids[i] for i in perm[:n_v].tolist())
        tr = [i for i in range(n) if norm_sids[i] not in val_set]
        va = [i for i in range(n) if norm_sids[i] in val_set]
        if args.subject_split_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.subject_split_json)), exist_ok=True)
            with open(args.subject_split_json, "w") as f:
                json.dump({"val": sorted(val_set), "seed": args.seed, "val_fraction": args.val_fraction}, f, indent=2)
            print(f"[split] saved {len(val_set)} val subjects → {args.subject_split_json}", flush=True)
    # HCP-YA age is released as 3 bucket labels (23, 28, 33), so age is a
    # 3-way classification task rather than regression.
    def _age_to_cls(a: np.ndarray) -> np.ndarray:
        buckets = np.sort(np.unique(a[a > 0]))
        lut: Dict[float, int] = {float(b): i for i, b in enumerate(buckets)}
        return np.array([lut.get(float(v), -1) if v > 0 else -1 for v in a], dtype=np.int64)

    y_by_task = {"sex": y_sex, "age": _age_to_cls(y_age)}

    res: Dict[str, Any] = {
        "split_kind": split_kind,
        "eval_split": args.eval_split,
        "n_subjects": int(n),
        "n_with_labels": len(ok),
        "feature_dim": int(Xn.shape[1]),
        "seed": args.seed,
        "pooled_npy": os.path.abspath(args.pooled_npy),
        "protocol": {
            "classification_probe": "logistic_l2_balanced",
            "classification_C": args.probe_c,
            "C_grid": [float(c) for c in C_GRID],
            "regression_probe": "ridge_l2",
            "alpha_grid": [float(a) for a in ALPHA_GRID],
            "inner_cv_folds": int(args.inner_cv_folds),
            "standardiser": "fit on training subjects only",
            "selection_subjects": "training subjects only (inner CV)",
            "eval_subjects": args.eval_split,
        },
    }
    preds: Dict[str, Dict[str, np.ndarray]] = {}
    pred_sids: Dict[str, List[str]] = {}
    for task, y_all in y_by_task.items():
        tr_m = [i for i in tr if y_all[i] >= 0]
        va_m = [i for i in va if y_all[i] >= 0]
        if len(tr_m) < 2 or len(va_m) < 1 or len(np.unique(y_all[tr_m])) < 2:
            res[task] = None
            continue
        metrics, arrays = evaluate_task(
            Xn[tr_m, :],
            y_all[tr_m],
            Xn[va_m, :],
            y_all[va_m],
            seed=args.seed,
            n_boot=args.n_boot,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            select_C=(args.probe_c == "selected"),
            cv_folds=args.inner_cv_folds,
        )
        res[task] = metrics
        preds[task] = arrays
        pred_sids[task] = [norm_sids[i] for i in va_m]

    for task in args.regression_tasks:
        y_all = y_reg[task]
        tr_m = [i for i in tr if np.isfinite(y_all[i])]
        va_m = [i for i in va if np.isfinite(y_all[i])]
        if len(tr_m) < 10 or len(va_m) < 3:
            res[task] = None
            continue
        metrics, arrays = evaluate_regression_task(
            Xn[tr_m, :],
            y_all[tr_m],
            Xn[va_m, :],
            y_all[va_m],
            seed=args.seed,
            n_boot=args.n_boot,
            cv_folds=args.inner_cv_folds,
        )
        res[task] = metrics
        preds[task] = arrays
        pred_sids[task] = [norm_sids[i] for i in va_m]

    # Flat aliases kept so existing summary/report scripts keep working.
    for task in ("sex", "age"):
        m = res.get(task) or {}
        res[f"{task}_acc"] = m.get("acc")
        res[f"{task}_f1"] = m.get("f1_macro")
        res[f"{task}_auroc"] = m.get("auroc")
        res[f"chance_{task}_acc"] = m.get("chance_acc")

    print(json.dumps(res, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    if args.predictions_npz and preds:
        save_predictions(args.predictions_npz, preds, pred_sids)
        print(f"[predictions] wrote {args.predictions_npz}", flush=True)


if __name__ == "__main__":
    main()
