import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.fmri_multiscale.dataset import make_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify multi-scale fMRI dataloader and save QC plots.")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--num_batches", type=int, default=2)
    return p.parse_args()


def _save_plot(out_path: str, voxel_w: np.ndarray, parcel_w: np.ndarray, network_w: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    t_show = min(150, voxel_w.shape[0])
    axes[0].plot(voxel_w[:t_show, :5], linewidth=1.0)
    axes[0].set_title("Voxel window")
    axes[1].plot(parcel_w[:t_show, :5], linewidth=1.0)
    axes[1].set_title("Parcel window")
    axes[2].plot(network_w[:t_show, :5], linewidth=1.0)
    axes[2].set_title("Network window")
    for ax in axes:
        ax.set_xlabel("TR")
        ax.set_ylabel("Signal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = os.path.join(args.cache_dir, "verify_outputs")
    os.makedirs(out_dir, exist_ok=True)

    loader = make_dataloader(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        shuffle=False,
        num_workers=0,
    )
    print(f"Total windows: {len(loader.dataset)}")

    for bi, batch in enumerate(loader):
        voxel = batch["voxel_window"]  # [B,T,V]
        bottom = batch["bottom_tokens"]  # [B,N0,T,K]
        middle = batch["middle_tokens"]  # [B,T,P]
        top = batch["top_tokens"]  # [B,T,N]
        print(
            f"Batch {bi}: voxel={tuple(voxel.shape)} bottom={tuple(bottom.shape)} "
            f"middle={tuple(middle.shape)} top={tuple(top.shape)}"
        )
        print(
            f"  stats voxel(mean,std)=({voxel.mean().item():.4f},{voxel.std().item():.4f}) "
            f"bottom(mean,std)=({bottom.mean().item():.4f},{bottom.std().item():.4f})"
        )

        _save_plot(
            out_path=os.path.join(out_dir, f"batch_{bi}_sample0_multiscale.png"),
            voxel_w=voxel[0].numpy(),
            parcel_w=middle[0].numpy(),
            network_w=top[0].numpy(),
        )
        if bi + 1 >= args.num_batches:
            break

    print(f"Verification outputs saved under {out_dir}")


if __name__ == "__main__":
    main()
