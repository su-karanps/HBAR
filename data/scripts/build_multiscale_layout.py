import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.fmri_multiscale.preprocess import MultiScalePreprocessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build shared multi-scale layout once (no subject preprocessing).")
    p.add_argument("--gm_mask", type=str, required=True, help="Gray matter mask NIfTI.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write layout.npz and layout_meta.json.")
    p.add_argument("--difumo_maps", type=str, default=None, help="Optional DiFuMo maps NIfTI.")
    p.add_argument("--difumo_labels_csv", type=str, required=True, help="DiFuMo labels CSV with a Yeo_networks17 column.")
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--n_parcels", type=int, default=1024)
    p.add_argument("--voxel_storage_dtype", type=str, default="float16", choices=["float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

