"""
Canonical subject-level train/val/test splits.

Every consumer (tokenizer training, reconstruction eval, paired significance
tests, downstream probes) resolves its data through this module so that a
subject never appears in more than one split, and so that reported numbers can
be attributed to a named split.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


def norm_subject_id(s: str) -> str:
    """Normalise a subject ID so 'sub-113619', '113619' and '0113619' all match."""
    k = "".join(c for c in str(s).strip().lower() if c not in " _-")
    if k.startswith("sub") and len(k) > 3 and k[3:].isdigit():
        k = k[3:]
    return k.lstrip("0") or "0"


def read_index_subject_ids(index_path: str) -> List[str]:
    with open(index_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    ids: List[str] = []
    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject")
        if sid is None:
            path = str(rec.get("voxel_path") or rec.get("path") or "")
            sid = os.path.basename(path).replace(".npy", "").split("_")[0]
        ids.append(str(sid))
    return ids


def make_subject_split(
    subject_ids: Sequence[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, object]:
    """Partition unique subjects into disjoint train/val/test sets."""
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0.0:
        raise ValueError(f"train_frac + val_frac must be < 1 (got test_frac={test_frac})")

    uniq = sorted({norm_subject_id(s) for s in subject_ids})
    n = len(uniq)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    idx = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    return {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "n_subjects": n,
        **{name: sorted(uniq[i] for i in ids.tolist()) for name, ids in idx.items()},
    }


def save_split(split: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    missing = [k for k in SPLIT_NAMES if k not in split]
    if missing:
        raise ValueError(f"Split file {path} is missing keys: {missing}")
    sets = {k: {norm_subject_id(s) for s in split[k]} for k in SPLIT_NAMES}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"Split file {path} has {len(overlap)} subjects in both {a} and {b}")
    return split


def subject_set(split: Dict[str, object], name: str) -> set:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split name {name!r}; expected one of {SPLIT_NAMES}")
    return {norm_subject_id(s) for s in split[name]}


def record_indices_for_split(
    records: Sequence[Dict[str, object]], split: Dict[str, object], name: str
) -> List[int]:
    wanted = subject_set(split, name)
    out: List[int] = []
    for i, rec in enumerate(records):
        sid = rec.get("subject_id") or rec.get("subject")
        if sid is None:
            path = str(rec.get("voxel_path") or rec.get("path") or "")
            sid = os.path.basename(path).replace(".npy", "").split("_")[0]
        if norm_subject_id(str(sid)) in wanted:
            out.append(i)
    return out


def window_indices_for_split(dataset, split: Dict[str, object], name: str) -> np.ndarray:
    """Indices into `dataset` whose windows come from subjects in the named split."""
    keep = set(record_indices_for_split(dataset.records, split, name))
    idx = [i for i, (rec_idx, _) in enumerate(dataset.windows) if rec_idx in keep]
    if not idx:
        raise ValueError(f"Split {name!r} matched no windows in the dataset index.")
    return np.asarray(idx, dtype=np.int64)


def resolve_split_path(data_cfg: Dict[str, object]) -> str:
    """Read the split file path out of a data config, erroring loudly if absent."""
    path = data_cfg.get("data", {}).get("subject_split_json")
    if not path:
        raise ValueError(
            "Data config has no data.subject_split_json. Generate one with "
            "scripts/make_subject_split.py and point the config at it."
        )
    if not os.path.isfile(str(path)):
        raise FileNotFoundError(f"subject_split_json does not exist: {path}")
    return str(path)


def describe(split: Dict[str, object], dataset=None) -> str:
    parts: List[str] = []
    for name in SPLIT_NAMES:
        n_subj = len(split[name])
        if dataset is not None:
            n_win = len(window_indices_for_split(dataset, split, name))
            parts.append(f"{name}={n_subj} subj/{n_win} win")
        else:
            parts.append(f"{name}={n_subj} subj")
    return "  ".join(parts)
