"""Code-shape and bit-count check for an npq_hier_residual config.

Builds the model from each YAML (random init; no checkpoint, no training), runs one
val window through it, and counts stored bits from the *actual* quantized tensors
q0/q1/q2 ([B, N, Tq, D] -> N * Tq * D sign bits per window). The ratio is taken
against the fp16 masked input of that window (unique valid voxels x T x 16).

    python3 scripts/eval/check_code_shape_bits.py \
        --data_config configs/fmri/data.yaml \
        --config configs/fmri/hbar52_fixed.yaml \
        --config configs/fmri/hbar415_fixed.yaml \
        --out_json results/code_shape_bits.json
"""
import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset  # noqa: E402
from data.fmri_multiscale.splits import load_split, window_indices_for_split  # noqa: E402
from train_fmri import _build_model, _load_config  # noqa: E402
from trainer_fmri import _STATIC_BATCH_KEYS  # noqa: E402

INPUT_BITS = 16


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_config", required=True)
    p.add_argument("--config", action="append", required=True, help="model YAML; repeatable")
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    dcfg = _load_config(args.data_config)["data"]
    ds = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=dcfg["index_path"],
        layout_dir=dcfg["layout_dir"],
        window_size=int(dcfg.get("window_size", 32)),
        stride=int(dcfg.get("stride", 16)),
    )
    va_idx = window_indices_for_split(ds, load_split(dcfg["subject_split_json"]), "val")
    batch = next(iter(DataLoader(Subset(ds, [int(va_idx[0])]), batch_size=1)))

    t_window = int(batch["voxel_window"].shape[1])
    ci = batch["chunk_indices"][0]
    n_voxels = int(torch.unique(ci[ci >= 0]).numel())
    input_bits = n_voxels * t_window * INPUT_BITS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fwd = {k: (v[0] if k in _STATIC_BATCH_KEYS and v.ndim >= 2 else v).to(device)
           for k, v in batch.items() if torch.is_tensor(v)}

    results = []
    for cfg_path in args.config:
        mcfg = _load_config(cfg_path)
        torch.manual_seed(0)
        model = _build_model(mcfg, batch).to(device).eval()
        with torch.no_grad():
            out = model(fwd)
        recon_shape = tuple(out["voxel_recon"].shape)
        assert recon_shape == tuple(batch["voxel_window"].shape), (
            f"voxel_recon {recon_shape} != voxel_window {tuple(batch['voxel_window'].shape)}"
        )
        levels = {}
        total = 0
        for name, key in (("bottom", "q0"), ("middle", "q1"), ("top", "q2")):
            if key not in out:
                continue
            q = out[key]
            assert q.ndim == 4 and q.shape[0] == 1, f"{key} shape {tuple(q.shape)} is not [1, N, Tq, D]"
            _, n, tq, d = (int(s) for s in q.shape)
            quant_dim = int(mcfg["model"][f"{name}_quant_dim"])
            assert d == quant_dim, f"{key} code dim {d} != {name}_quant_dim {quant_dim}"
            level = getattr(model, f"level_{name}")
            if getattr(level, "quantizer", "bsq") == "rvq":
                # Residual VQ: one integer id per stage per (spatial, time-slot) code, so the
                # stored bits are N * Tq * sum_i log2(K_i); the D real-valued code entries are
                # not sign bits (reported as `elements` for reference).
                sizes = [int(k) for k in level.rvq.codebook_sizes]
                bits_per_code = sum(math.log2(k) for k in sizes)
                bits = int(round(n * tq * bits_per_code))
                levels[name] = {
                    "shape": [n, tq, d], "bits": bits, "quantizer": "rvq",
                    "codebook_sizes": sizes, "bits_per_code": bits_per_code, "elements": n * tq * d,
                }
            else:
                # BSQ codes are +-c for a single scale c: exactly one bit per element.
                mags = torch.unique(q.abs().float().round(decimals=5))
                assert mags.numel() == 1 and mags.item() > 0, f"{key} is not a sign code (|q| values {mags[:5].tolist()})"
                bits = n * tq * d
                levels[name] = {"shape": [n, tq, d], "bits": bits}
            total += bits
        ratio = input_bits / total
        row = {
            "config": cfg_path,
            "levels": levels,
            "bits_per_window": total,
            "input_bits_per_window": input_bits,
            "compression_per_window": ratio,
            "bits_per_input_sample": total / (n_voxels * t_window),
        }
        results.append(row)
        print(f"{cfg_path}")
        for name, lv in levels.items():
            if lv.get("quantizer") == "rvq":
                print(f"  {name:6s} [N, Tq, D] = {lv['shape']}  rvq K={lv['codebook_sizes']} "
                      f"({lv['bits_per_code']:g} bits/code)  bits = {lv['bits']:,}  "
                      f"(code elements N*Tq*D = {lv['elements']:,})")
            else:
                print(f"  {name:6s} [N, Tq, D] = {lv['shape']}  bits = {lv['bits']:,}")
        print(f"  total {total:,} bits/window  ->  {ratio:.2f}x  ({row['bits_per_input_sample']:.4f} bits/sample)")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "data_config": args.data_config,
            "val_window_index": int(va_idx[0]),
            "n_unique_voxels": n_voxels,
            "window_tr": t_window,
            "input_bits": INPUT_BITS,
            "models": results,
        }, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
