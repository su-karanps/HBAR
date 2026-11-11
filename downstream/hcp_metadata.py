"""
Load HCP (or similar) subject-level CSV for age/sex and match to index records.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SubjectPheno:
    subject_id: str
    age: float
    sex: int  # 0/1 (binary) or -1 for unknown


def _to_float(x: str) -> float:
    try:
        return float(x.strip())
    except Exception:
        return float("nan")


def _norm_key(s: str) -> str:
    """Normalise a subject ID for fuzzy matching.

    Strips separators, the BIDS 'sub' prefix, and leading zeros so that
    'sub-113619', '113619', 'Sub_113619', and '0113619' all map to '113619'.
    """
    k = "".join(c for c in str(s).strip().lower() if c not in " _-")
    # Strip BIDS-style 'sub' prefix (e.g. 'sub113619' → '113619')
    if k.startswith("sub") and len(k) > 3 and k[3:].isdigit():
        k = k[3:]
    return k.lstrip("0") or "0"


def load_hcp_csv(
    path: str,
    id_column: str = "subject",
    age_column: str = "age",
    sex_column: str = "sex",
) -> List[SubjectPheno]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows: List[SubjectPheno] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"Empty or missing header in {path}")
        low = {n.lower().strip(): n for n in r.fieldnames}
        ikey = low.get(id_column.lower()) or list(r.fieldnames)[0]
        akey = low.get(age_column.lower()) or "age"
        skey = low.get(sex_column.lower()) or "sex"
        for row in r:
            sid = (row.get(ikey) or "").strip()
            if not sid:
                continue
            age = _to_float((row.get(akey) or "nan").strip())
            sex_raw = (row.get(skey) or "").strip()
            if sex_raw.upper() in {"M", "1", "MALE"}:
                sex = 1
            elif sex_raw.upper() in {"F", "0", "FEMALE", "2"}:
                sex = 0
            else:
                try:
                    v = int(float(sex_raw))
                    sex = 1 if v == 1 else 0
                except Exception:
                    sex = -1
            rows.append(SubjectPheno(subject_id=sid, age=age, sex=sex))
    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return rows


def load_subject_columns(
    path: str,
    columns: List[str],
    id_column: str = "subject_id",
    missing_sentinel: float = -1.0,
) -> Dict[str, Dict[str, float]]:
    """Map normalised subject_id -> {column: float}, with ``missing_sentinel`` -> NaN.

    Used for the continuous behavioural endpoints (flanker, neuroticism), which
    the HCP table stores as plain numeric columns with -1 marking "not available".
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"Empty or missing header in {path}")
        low = {n.lower().strip(): n for n in r.fieldnames}
        ikey = low.get(id_column.lower()) or list(r.fieldnames)[0]
        missing_cols = [c for c in columns if c.lower() not in low]
        if missing_cols:
            raise ValueError(f"{path} has no column(s) {missing_cols}; has {r.fieldnames}")
        for row in r:
            sid = (row.get(ikey) or "").strip()
            if not sid:
                continue
            vals: Dict[str, float] = {}
            for c in columns:
                v = _to_float(row.get(low[c.lower()]) or "nan")
                vals[c] = float("nan") if v == missing_sentinel else v
            out[_norm_key(sid)] = vals
    return out


def pheno_to_dict(rows: List[SubjectPheno]) -> Dict[str, Tuple[float, int]]:
    """Map normalized subject_id -> (age, sex) — last wins on duplicates."""
    m: Dict[str, Tuple[float, int]] = {}
    for s in rows:
        m[_norm_key(s.subject_id)] = (s.age, s.sex)
    return m


def infer_subject_id_from_record(rec: Dict[str, Any]) -> str:
    for k in (
        "subject",
        "subject_id",
        "HCP",
        "hcp",
        "eid",
        "eID",
        "id",
        "participant_id",
        "Participant",
    ):
        v = rec.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    p = str(rec.get("voxel_path") or rec.get("path") or "")
    base = os.path.basename(p)
    return base.replace(".npy", "").split("_")[0] or base
