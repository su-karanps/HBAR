import argparse
import json
import os
from glob import glob
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-shard index.json files into one training manifest.")
    p.add_argument("--shards_root", type=str, required=True, help="Root directory containing shard_*/index.json")
    p.add_argument("--output_index", type=str, required=True, help="Path to merged index JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard_index_files = sorted(glob(os.path.join(args.shards_root, "shard_*", "index.json")))
    if not shard_index_files:
        raise FileNotFoundError(f"No shard index files found under {args.shards_root}")

    merged: List[Dict[str, object]] = []
    for idx_file in shard_index_files:
        with open(idx_file, "r", encoding="utf-8") as f:
            recs = json.load(f)
        shard_dir = os.path.dirname(idx_file)
        for r in recs:
            # Ensure sample path is absolute for robust multi-node reads.
            rp = str(r["path"])
            if not os.path.isabs(rp):
                r["path"] = os.path.abspath(os.path.join(shard_dir, rp))
            merged.append(r)

    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    with open(args.output_index, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(shard_index_files)} shards -> {len(merged)} records")
    print(f"Wrote merged index: {args.output_index}")


if __name__ == "__main__":
    main()

