"""
Shared probe evaluation: point metrics, bootstrap CIs, repeated
cross-validation, and per-subject prediction dumps.

Two task families share one protocol:
  * classification (sex, age) -- balanced logistic regression;
  * regression (flanker, neuroticism) -- ridge.
Both standardise on the training subjects only and pick their regularisation
strength by inner cross-validation on the training subjects only.

Predictions are written to disk so that any additional metric (AUROC, AUPRC,
calibration, paired tests between models) can be recomputed later without
re-running feature extraction.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METRIC_KEYS = ("acc", "f1_macro", "auroc")
REG_METRIC_KEYS = ("pearson_r", "spearman_rho", "r2", "mae")

# Fixed selection grids, identical for every representation and every task.
# Both are wide enough that no representation is pinned to an endpoint: the
# 1024-D atlas baselines select the strong-regularisation end (C ~ 1e-3) and the
# 18-36-D tokenizer features select the top of the ridge grid on the behavioural
# endpoints. A selection pinned to the edge of the grid is not a selection, and
# for the baselines it would also be an unfairly under-regularised probe.
C_GRID: Tuple[float, ...] = tuple(float(c) for c in np.logspace(-5.0, 2.0, 8))
ALPHA_GRID: Tuple[float, ...] = tuple(float(a) for a in np.logspace(-2.0, 6.0, 17))


def fit_logistic(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int,
    max_iter: int = 1000,
    C: float = 1.0,
):
    """Standardise on train, fit balanced logistic regression, return (pred, prob)."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    clf = LogisticRegression(
        C=float(C), max_iter=max_iter, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf.fit(X_tr_s, y_tr)
    prob = clf.predict_proba(X_va_s)
    pred = clf.classes_[np.argmax(prob, axis=1)]
    return pred, prob, clf.classes_


def select_logistic_C(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    c_grid: Sequence[float] = C_GRID,
    n_folds: int = 5,
    max_iter: int = 1000,
) -> Tuple[float, List[float]]:
    """Pick C by inner stratified CV on the *training* subjects only (AUROC)."""
    n_folds = int(min(n_folds, np.min(np.bincount(y_tr.astype(np.int64)))))
    if n_folds < 2:
        return float(c_grid[len(c_grid) // 2]), []
    scoring = "roc_auc" if len(np.unique(y_tr)) == 2 else "roc_auc_ovr"
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: List[float] = []
    for c in c_grid:
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=float(c),
                        max_iter=max_iter,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
        s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
        scores.append(float(np.mean(s)))
    return float(c_grid[int(np.argmax(scores))]), scores


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: Sequence[int]
) -> Dict[str, Optional[float]]:
    """Accuracy, macro F1, and AUROC (binary or one-vs-rest macro)."""
    out: Dict[str, Optional[float]] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    present = np.unique(y_true)
    if len(present) < 2:
        out["auroc"] = None
        return out
    try:
        if len(classes) == 2:
            pos_col = int(np.where(np.asarray(classes) == classes[1])[0][0])
            out["auroc"] = float(roc_auc_score(y_true, y_prob[:, pos_col]))
        else:
            out["auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(classes))
            )
    except ValueError:
        out["auroc"] = None
    return out


def bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: Sequence[int],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Optional[List[float]]]:
    """Percentile bootstrap CIs over held-out subjects (resamples subjects, refits nothing)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = classification_metrics(y_true[idx], y_pred[idx], y_prob[idx], classes)
        for k in METRIC_KEYS:
            if m[k] is not None:
                draws[k].append(float(m[k]))
    out: Dict[str, Optional[List[float]]] = {}
    for k in METRIC_KEYS:
        if len(draws[k]) < 100:
            out[k] = None
        else:
            arr = np.asarray(draws[k])
            out[k] = [float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0))]
    return out


def chance_references(y_tr: np.ndarray, y_va: np.ndarray) -> Dict[str, Optional[float]]:
    """Metric-specific chance levels: the train-majority predictor, and AUROC 0.5.

    Macro-F1 is *not* calibrated by majority-class accuracy, so the majority
    predictor's own macro-F1 is reported alongside it.
    """
    majority = int(np.bincount(y_tr.astype(np.int64)).argmax())
    const = np.full_like(y_va, majority)
    return {
        "chance_acc": float(np.mean(y_va == majority)),
        "chance_f1_macro": float(f1_score(y_va, const, average="macro", zero_division=0)),
        "chance_auroc": 0.5,
        "majority_class_train": majority,
    }


def repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    C: float = 1.0,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Repeated stratified K-fold over subjects; reports mean/std of each metric.

    This refits *only the probe*; the tokenizer that produced ``X`` is frozen and
    was trained on the canonical training subjects, so folds drawn from those
    subjects are not an inductive end-to-end CV of the tokenizer.
    """
    if len(np.unique(y)) < 2 or np.min(np.bincount(y.astype(np.int64))) < n_splits:
        return {k: None for k in METRIC_KEYS}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    per_fold: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        pred, prob, classes = fit_logistic(X[tr], y[tr], X[te], seed=seed + fold, C=C)
        m = classification_metrics(y[te], pred, prob, classes)
        for k in METRIC_KEYS:
            if m[k] is not None:
                per_fold[k].append(float(m[k]))
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for k in METRIC_KEYS:
        if not per_fold[k]:
            out[k] = None
        else:
            arr = np.asarray(per_fold[k])
            out[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n_folds": int(len(arr)),
            }
    return out


def evaluate_task(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    n_boot: int,
    cv_splits: int,
    cv_repeats: int,
    select_C: bool = True,
    c_grid: Sequence[float] = C_GRID,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Point metrics on the held-out split, bootstrap CIs, repeated-CV, and chance levels.

    With ``select_C`` the regularisation strength comes from an inner stratified
    CV on the training subjects only; the fixed-C=1 result is still recorded
    under ``fixed_C_reference`` so the protocol change stays auditable.
    """
    c_used, c_scores = 1.0, []
    if select_C:
        c_used, c_scores = select_logistic_C(X_tr, y_tr, seed=seed, c_grid=c_grid, n_folds=cv_folds)
    pred, prob, classes = fit_logistic(X_tr, y_tr, X_va, seed=seed, C=c_used)
    res: Dict[str, Any] = dict(classification_metrics(y_va, pred, prob, classes))
    res["ci95"] = bootstrap_cis(y_va, pred, prob, classes, n_boot=n_boot, seed=seed)
    res.update(chance_references(y_tr, y_va))
    res["probe"] = "logistic_l2_balanced"
    res["C"] = float(c_used)
    res["C_selected_by_inner_cv"] = bool(select_C)
    res["C_grid"] = [float(c) for c in c_grid]
    res["C_at_grid_edge"] = bool(select_C and c_used in (min(c_grid), max(c_grid)))
    res["C_inner_cv_scores"] = c_scores
    res["auroc_averaging"] = "binary" if len(classes) == 2 else "ovr_macro"
    res["n_train"] = int(len(y_tr))
    res["n_eval"] = int(len(y_va))
    res["class_counts_train"] = {int(c): int(np.sum(y_tr == c)) for c in np.unique(y_tr)}
    res["class_counts_eval"] = {int(c): int(np.sum(y_va == c)) for c in np.unique(y_va)}
    if select_C:
        p0, q0, k0 = fit_logistic(X_tr, y_tr, X_va, seed=seed, C=1.0)
        res["fixed_C_reference"] = {"C": 1.0, **classification_metrics(y_va, p0, q0, k0)}
    if cv_repeats > 0:
        res["cv"] = repeated_cv(
            np.concatenate([X_tr, X_va], axis=0),
            np.concatenate([y_tr, y_va], axis=0),
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            seed=seed,
            C=c_used,
        )
        res["cv_note"] = (
            "repeated stratified K-fold refits the PROBE ONLY on train+eval subjects pooled; "
            "the tokenizer is frozen and saw the training subjects, so this is not inductive "
            "end-to-end CV"
        )
    return res, {"y_true": y_va, "y_pred": pred, "y_prob": prob, "classes": np.asarray(classes)}


# --------------------------------------------------------------------------- #
# Regression endpoints (prespecified: flanker primary, neuroticism secondary)
# --------------------------------------------------------------------------- #


def fit_ridge(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, alpha: float) -> np.ndarray:
    """Standardise on train, fit ridge, return held-out predictions."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    reg = Ridge(alpha=float(alpha))
    reg.fit(X_tr_s, y_tr)
    return reg.predict(X_va_s).astype(np.float64)


def select_ridge_alpha(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    n_folds: int = 5,
) -> Tuple[float, List[float]]:
    """Pick alpha by inner K-fold CV on the *training* subjects only (neg MSE)."""
    cv = KFold(n_splits=int(min(n_folds, len(y_tr))), shuffle=True, random_state=seed)
    scores: List[float] = []
    for a in alpha_grid:
        pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=float(a)))])
        s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_squared_error")
        scores.append(float(np.mean(s)))
    return float(alpha_grid[int(np.argmax(scores))]), scores


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, train_mean: float
) -> Dict[str, Optional[float]]:
    """Pearson r, Spearman rho, R^2 against the train-mean predictor, and MAE.

    R^2 = 1 - SS_res / SS_tot with SS_tot measured against ``train_mean`` (the
    chance predictor fit on training subjects), not the held-out mean, so it is
    a genuine out-of-sample skill score.
    """
    from scipy.stats import pearsonr, spearmanr

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(train_mean)) ** 2))
    out: Dict[str, Optional[float]] = {
        "r2": (None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
    # ptp, not std: a constant predictor has std ~1e-14 from cancellation.
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        out["pearson_r"] = None
        out["spearman_rho"] = None
        return out
    out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    out["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
    return out


def regression_bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mean: float,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Optional[List[float]]]:
    """Percentile bootstrap CIs over held-out subjects (resamples subjects, refits nothing)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: Dict[str, List[float]] = {k: [] for k in REG_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(y_true[idx], y_pred[idx], train_mean)
        for k in REG_METRIC_KEYS:
            if m[k] is not None:
                draws[k].append(float(m[k]))
    out: Dict[str, Optional[List[float]]] = {}
    for k in REG_METRIC_KEYS:
        if len(draws[k]) < 100:
            out[k] = None
        else:
            arr = np.asarray(draws[k])
            out[k] = [float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0))]
    return out


def evaluate_regression_task(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    n_boot: int,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Ridge probe on held-out subjects, with CIs and the train-mean chance reference."""
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64)
    alpha, alpha_scores = select_ridge_alpha(X_tr, y_tr, seed=seed, alpha_grid=alpha_grid, n_folds=cv_folds)
    pred = fit_ridge(X_tr, y_tr, X_va, alpha)
    train_mean = float(np.mean(y_tr))
    res: Dict[str, Any] = dict(regression_metrics(y_va, pred, train_mean))
    res["ci95"] = regression_bootstrap_cis(y_va, pred, train_mean, n_boot=n_boot, seed=seed)
    const = np.full_like(y_va, train_mean)
    res["chance"] = {
        "predictor": "train_mean",
        "train_mean": train_mean,
        **{k: v for k, v in regression_metrics(y_va, const, train_mean).items()},
    }
    res["probe"] = "ridge_l2"
    res["alpha"] = float(alpha)
    res["alpha_selected_by_inner_cv"] = True
    res["alpha_grid"] = [float(a) for a in alpha_grid]
    res["alpha_at_grid_edge"] = bool(alpha in (min(alpha_grid), max(alpha_grid)))
    res["alpha_inner_cv_neg_mse"] = alpha_scores
    res["n_train"] = int(len(y_tr))
    res["n_eval"] = int(len(y_va))
    res["target_stats_train"] = {"mean": train_mean, "sd": float(np.std(y_tr, ddof=1))}
    res["target_stats_eval"] = {"mean": float(np.mean(y_va)), "sd": float(np.std(y_va, ddof=1))}
    # train_mean travels with the predictions so paired R^2 tests need no refit.
    return res, {"y_true": y_va, "y_pred": pred, "train_mean": np.asarray(train_mean)}


def save_predictions(path: str, tasks: Dict[str, Dict[str, np.ndarray]], subject_ids: Dict[str, List[str]]) -> None:
    """Write one npz holding per-subject truth/prediction/probability for every task."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    for task, arrays in tasks.items():
        for key, arr in arrays.items():
            payload[f"{task}__{key}"] = arr
        payload[f"{task}__subject_ids"] = np.asarray(subject_ids[task], dtype=object)
    np.savez(path, **payload)
