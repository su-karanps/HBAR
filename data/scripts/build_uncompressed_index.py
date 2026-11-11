"""Convert the compressed per-subject caches into memory-mappable .npy arrays.

build_multiscale_hcpya.py writes one compressed `<subject>.npz` per scan (voxel_ts, parcel_ts,
network_ts). Training reads random windows from many subjects, so the loader expects one
uncompressed float16 .npy per scale and subject, listed in `merged_index_uncompressed.json`.

    python3 data/scripts/build_uncompressed_index.py \
        --index_path /path/to/build/merged_index.json --output_dir /path/to/build
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np


def _convert(task):
    i, rec, out_dir, subjects_per_dir = task
    sid = str(rec["subject_id"])
    sub_dir = os.path.join(out_dir, f"shard_{i // subjects_per_dir:04d}")
    os.makedirs(sub_dir, exist_ok=True)
    out = {"subject_id": sid, "timepoints": int(rec["timepoints"]), "source_path": str(rec["path"])}
    with np.load(rec["path"]) as z:
        for scale in ("voxel", "parcel", "network"):
            path = os.path.join(sub_dir, f"s{i:06d}_{sid}_{scale}.npy")
            if not os.path.exists(path):
                np.save(path + ".tmp.npy", np.asarray(z[f"{scale}_ts"], dtype=np.float16))
                os.replace(path + ".tmp.npy", path)
            out[f"{scale}_path"] = path
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index_path", required=True, help="merged_index.json from merge_multiscale_shards.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subjects_per_dir", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    with open(args.index_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    out_dir = os.path.abspath(args.output_dir)
    tasks = [(i, r, out_dir, args.subjects_per_dir) for i, r in enumerate(records)]
    with Pool(args.num_workers) as pool:
        out = pool.map(_convert, tasks, chunksize=1)

    index_path = os.path.join(out_dir, "merged_index_uncompressed.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{len(out)} subjects -> {index_path}")


if __name__ == "__main__":
    main()
