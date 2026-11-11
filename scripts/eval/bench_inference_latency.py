#!/usr/bin/env python3
"""Inference latency of a trained tokenizer on real held-out windows.

Times, with CUDA synchronisation, (a) the full forward pass (encode + BSQ +
residual decode) and (b) `encode_only=True` (codes only), at several batch
sizes and precisions. Data loading and preprocessing are excluded: the batch
is moved to the device before the timer starts. Reports median and IQR of
per-batch wall time over `--n_iters` batches after `--warmup` untimed ones.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset  # noqa: E402
from data.fmri_multiscale.splits import load_split, resolve_split_path, window_indices_for_split  # noqa: E402
from scripts.eval.paired_tokenizer_significance import _load_checkpoint  # noqa: E402
from train_fmri import _build_model, _load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="NAME:CONFIG:CKPT (repeatable)")
    p.add_argument("--data_config", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_sizes", default="1,4,8")
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--n_iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", required=True)
    a = p.parse_args()

    dcfg = _load_config(a.data_config)["data"]
    ds = FMRIMultiScaleWindowDataset(
        cache_dir=None, index_path=dcfg["index_path"], layout_dir=dcfg["layout_dir"],
        window_size=dcfg.get("window_size", 32), stride=dcfg.get("stride", 16),
        cache_items=dcfg.get("cache_items", 8),
    )
    split = load_split(resolve_split_path({"data": dcfg}))
    window_ids = window_indices_for_split(ds, split, a.split).tolist()
    rng = np.random.default_rng(a.seed)
    rng.shuffle(window_ids)
    max_bs = max(int(b) for b in a.batch_sizes.split(","))
    window_ids = window_ids[: max_bs * (a.n_iters + a.warmup)]
    dev = torch.device("cuda")
    results = {"device": torch.cuda.get_device_name(0), "split": a.split, "n_iters": a.n_iters,
               "warmup": a.warmup, "timer": "torch.cuda.synchronize + perf_counter around model(batch) only; "
               "data loading, host->device copy and preprocessing excluded", "models": {}}
    for spec in a.model:
        name, cfg_path, ckpt = spec.split(":", 2)
        loader0 = DataLoader(Subset(ds, window_ids), batch_size=max_bs, shuffle=False, num_workers=2)
        sample = next(iter(loader0))
        model = _build_model(_load_config(cfg_path), sample).to(dev).eval()
        _load_checkpoint(model, ckpt)
        n_params = sum(t.numel() for t in model.parameters())
        results["models"][name] = {"config": cfg_path, "ckpt": ckpt, "n_params": n_params, "runs": {}}
        for bs in (int(b) for b in a.batch_sizes.split(",")):
            loader = DataLoader(Subset(ds, window_ids), batch_size=bs, shuffle=False, num_workers=2)
            for dt in a.dtypes.split(","):
                for mode in ("full", "encode_only"):
                    times = []
                    torch.cuda.reset_peak_memory_stats(dev)
                    for bi, batch in enumerate(loader):
                        if bi >= a.warmup + a.n_iters:
                            break
                        batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                        torch.cuda.synchronize(dev)
                        t0 = time.perf_counter()
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dt == "bf16")):
                            model(batch, encode_only=(mode == "encode_only"))
                        torch.cuda.synchronize(dev)
                        if bi >= a.warmup:
                            times.append(time.perf_counter() - t0)
                    q = statistics.quantiles(times, n=4) if len(times) >= 4 else [min(times), statistics.median(times), max(times)]
                    r = {"batch_size": bs, "dtype": dt, "mode": mode,
                         "ms_per_batch_median": 1000 * statistics.median(times),
                         "ms_per_batch_iqr": [1000 * q[0], 1000 * q[2]],
                         "ms_per_window_median": 1000 * statistics.median(times) / bs,
                         "windows_per_s": bs / statistics.median(times),
                         "peak_alloc_gb": torch.cuda.max_memory_allocated(dev) / 1024**3}
                    results["models"][name]["runs"][f"bs{bs}_{dt}_{mode}"] = r
                    print(f"[{name}] bs={bs} {dt} {mode}: {r['ms_per_window_median']:.1f} ms/window "
                          f"({r['ms_per_batch_median']:.1f} ms/batch, peak {r['peak_alloc_gb']:.1f} GB)", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(a.output_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[saved] {a.output_json}")


if __name__ == "__main__":
    main()
