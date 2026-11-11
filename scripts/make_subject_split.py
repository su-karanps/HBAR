#!/usr/bin/env python3
"""Generate a canonical subject-level train/val/test split from a dataset index."""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.fmri_multiscale.splits import (
    describe,
    make_subject_split,
    read_index_subject_ids,
    save_split,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_path", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output_json) and not args.overwrite:
        raise SystemExit(f"{args.output_json} exists; pass --overwrite to replace it.")

    subject_ids = read_index_subject_ids(args.index_path)
    split = make_subject_split(subject_ids, args.train_frac, args.val_frac, args.seed)
    save_split(split, args.output_json)
    print(f"[split] {describe(split)}  seed={args.seed}")
    print(f"[split] wrote {args.output_json}")


if __name__ == "__main__":
    main()
