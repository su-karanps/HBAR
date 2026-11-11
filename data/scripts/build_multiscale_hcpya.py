import argparse
import os
import re
import sys
from glob import glob
from typing import List, Optional, Sequence, Tuple

import nibabel as nib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.fmri_multiscale.preprocess import MultiScalePreprocessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-scale fMRI cache from HCP-YA derivatives.")
    p.add_argument(
        "--mode",
        type=str,
        default="process_shard",
        choices=["build_layout", "process_shard"],
        help="build_layout: compute shared layout only; process_shard: process scans using existing layout.",
    )
    p.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="HCP-YA derivatives root. Expected layout: sub-*/func/*_desc-preproc_bold.nii.gz",
    )
    p.add_argument("--gm_mask", type=str, default=None, help="Gray matter mask NIfTI (required for mode=build_layout).")
    p.add_argument("--output_dir", type=str, required=True, help="Output cache directory for this shard.")
    p.add_argument(
        "--layout_dir",
        type=str,
        default=None,
        help="Path to precomputed shared layout directory containing layout.npz and layout_meta.json.",
    )
    p.add_argument("--difumo_maps", type=str, default=None, help="Optional DiFuMo maps NIfTI.")
    p.add_argument("--difumo_labels_csv", type=str, default=None, help="DiFuMo labels CSV with a Yeo_networks17 column (required for mode=build_layout).")
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--n_parcels", type=int, default=1024)
    p.add_argument("--voxel_storage_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--no_zscore", action="store_true")
    p.add_argument(
        "--run_selection",
        type=str,
        default="LR_largest",
        choices=["LR_largest", "RL_largest", "largest", "LR", "RL"],
        help="Run selection policy for each subject.",
    )
    p.add_argument("--num_shards", type=int, default=1, help="Total number of array shards.")
    p.add_argument("--shard_index", type=int, default=0, help="0-based shard index.")
    return p.parse_args()


def _extract_subject_id(path: str) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+)", path)
    if m:
        return m.group(1)
    return os.path.basename(path)


def _timepoints(path: str) -> int:
    img = nib.load(path)
    shape = img.shape
    if len(shape) != 4:
        return -1
    return int(shape[3])


def _pick_largest(files: Sequence[str]) -> Optional[str]:
    if not files:
        return None
    best = None
    best_t = -1
    for fp in files:
        t = _timepoints(fp)
        if t > best_t:
            best_t = t
            best = fp
    return best


def _select_run(files: Sequence[str], policy: str) -> Optional[str]:
    if not files:
        return None
    lr_files = [f for f in files if "_dir-LR_" in os.path.basename(f)]
    rl_files = [f for f in files if "_dir-RL_" in os.path.basename(f)]

    if policy == "largest":
        return _pick_largest(files)
    if policy == "LR":
        return _pick_largest(lr_files) or _pick_largest(rl_files) or _pick_largest(files)
    if policy == "RL":
        return _pick_largest(rl_files) or _pick_largest(lr_files) or _pick_largest(files)
    if policy == "LR_largest":
        return _pick_largest(lr_files) or _pick_largest(rl_files) or _pick_largest(files)
    if policy == "RL_largest":
        return _pick_largest(rl_files) or _pick_largest(lr_files) or _pick_largest(files)
    raise ValueError(f"Unsupported run selection policy: {policy}")


def _discover_subject_dirs(root_dir: str) -> List[str]:
    return sorted(glob(os.path.join(root_dir, "sub-*")))


def _discover_selected_fmri_paths(root_dir: str, policy: str) -> Tuple[List[str], List[str]]:
    selected: List[str] = []
    missing: List[str] = []
    for subj_dir in _discover_subject_dirs(root_dir):
        sid = os.path.basename(subj_dir)
        func_dir = os.path.join(subj_dir, "func")
        files = sorted(glob(os.path.join(func_dir, "*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")))
        if not files:
            files = sorted(glob(os.path.join(func_dir, "*_desc-preproc_bold.nii.gz")))
        picked = _select_run(files, policy=policy)
        if picked is None:
            missing.append(sid)
        else:
            selected.append(picked)
    return selected, missing


def _shard(items: Sequence[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_index]


def main() -> None:
    args = parse_args()
    if args.mode == "build_layout":
        if args.gm_mask is None or args.difumo_labels_csv is None:
            raise ValueError("--gm_mask and --difumo_labels_csv are required in mode=build_layout")
        os.makedirs(args.output_dir, exist_ok=True)
        builder = MultiScalePreprocessor(
            output_dir=args.output_dir,
            gm_mask_path=args.gm_mask,
            difumo_maps_path=args.difumo_maps,
            difumo_labels_csv_path=args.difumo_labels_csv,
            n_parcels=args.n_parcels,
            chunk_size=args.chunk_size,
            voxel_storage_dtype=args.voxel_storage_dtype,
        )
        print(f"Built shared layout at {args.output_dir}")
        print(
            f"Layout meta: voxels={builder.layout['meta'].n_voxels}, "
            f"bottom_tokens={builder.layout['meta'].n_bottom_tokens}, "
            f"chunk_size={builder.layout['meta'].chunk_size}, "
            f"dtype={builder.layout['meta'].voxel_storage_dtype}"
        )
        return

    if args.layout_dir is None:
        raise ValueError("--layout_dir is required in mode=process_shard")

    all_paths, missing_subjects = _discover_selected_fmri_paths(args.root_dir, policy=args.run_selection)
    shard_paths = _shard(all_paths, args.num_shards, args.shard_index)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Discovered subjects with selected runs: {len(all_paths)}")
    print(f"Subjects with no usable runs: {len(missing_subjects)}")
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard_paths)} scans")
    if shard_paths:
        print(f"First shard example: {_extract_subject_id(shard_paths[0])} -> {os.path.basename(shard_paths[0])}")
    else:
        print("No scans assigned to this shard; exiting.")
        return

    builder = MultiScalePreprocessor(
        output_dir=args.output_dir,
        layout_dir=args.layout_dir,
    )
    records = builder.process_many(fmri_paths=shard_paths, zscore_voxels=not args.no_zscore)
    print(f"Built cache shard at {args.output_dir}")
    print(f"Volumes processed in shard: {len(records)}")


if __name__ == "__main__":
    main()

