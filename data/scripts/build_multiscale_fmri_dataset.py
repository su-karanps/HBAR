import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.fmri_multiscale.preprocess import MultiScalePreprocessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-scale fMRI cache for tokenizer training.")
    p.add_argument("--fmri_glob", type=str, required=True, help="Glob to 4D fMRI NIfTI files.")
    p.add_argument("--gm_mask", type=str, required=True, help="Gray matter mask NIfTI.")
    p.add_argument("--output_dir", type=str, required=True, help="Output cache directory.")
    p.add_argument("--difumo_maps", type=str, default=None, help="Optional DiFuMo-1024 maps NIfTI.")
    p.add_argument(
        "--difumo_labels_csv",
        type=str,
        required=True,
        help="DiFuMo labels CSV with a Yeo_networks17 column (parcel->network).",
    )
    p.add_argument("--chunk_size", type=int, default=16, help="Bottom token chunk size K.")
    p.add_argument("--n_parcels", type=int, default=1024)
    p.add_argument(
        "--voxel_storage_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Storage dtype for voxel_ts cache. float16 cuts storage roughly in half.",
    )
    p.add_argument("--no_zscore", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fmri_paths = sorted(glob.glob(args.fmri_glob))
    if not fmri_paths:
        raise FileNotFoundError(f"No files matched --fmri_glob={args.fmri_glob}")

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
    records = builder.process_many(fmri_paths=fmri_paths, zscore_voxels=not args.no_zscore)
    print(f"Built cache at {args.output_dir}")
    print(f"Volumes processed: {len(records)}")
    print(f"Example sample: {records[0]['path']}")


if __name__ == "__main__":
    main()
