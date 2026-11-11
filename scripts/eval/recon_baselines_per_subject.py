"""Model-free reconstruction baselines (mean, random) on exactly the windows recon_metrics_per_subject.py scores.

Window selection is copied verbatim from recon_metrics_per_subject.py (same split, seed shuffle, per-subject
cap), so the rows are comparable with Table 1. Metrics use the same _batch_mse / _batch_corr (whole-window
flattened Pearson). Baselines are applied per scope to that scope's target:
  mean   — each unit's within-window temporal mean, constant over the 32 TRs
  random — Gaussian noise with the window's global mean/std at that scope (seeded)
Per-window values -> subject means -> mean over subjects; 95% percentile bootstrap over subjects (10k).

    python3 scripts/eval/recon_baselines_per_subject.py --data_config ... --split test \
        --windows_per_subject 16 --seed 0 --output_json results/recon_baselines_test.json
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset  # noqa: E402
from data.fmri_multiscale.splits import load_split, resolve_split_path, window_indices_for_split  # noqa: E402
from downstream.hcp_metadata import infer_subject_id_from_record  # noqa: E402
from train_fmri import _load_config  # noqa: E402
from scripts.eval.paired_tokenizer_significance import _batch_corr, _batch_mse  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data_config", required=True)
p.add_argument("--split", default="test", choices=["train", "val", "test"])
p.add_argument("--windows_per_subject", type=int, default=16)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--num_workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output_json", required=True)
a = p.parse_args()
torch.manual_seed(a.seed)

dcfg = _load_config(a.data_config)
ds = FMRIMultiScaleWindowDataset(
    cache_dir=None, index_path=dcfg["data"]["index_path"], layout_dir=dcfg["data"]["layout_dir"],
    window_size=int(dcfg["data"].get("window_size", 32)), stride=int(dcfg["data"].get("stride", 16)),
    cache_items=int(dcfg["data"].get("cache_items", 8)), random_sampling=False,
    max_windows_per_subject=dcfg["data"].get("max_windows_per_subject"))
records = json.load(open(dcfg["data"]["index_path"], "r", encoding="utf-8"))
subj_by_record = [infer_subject_id_from_record(r) for r in records]
split = load_split(resolve_split_path(dcfg))
window_ids = window_indices_for_split(ds, split, a.split).tolist()
rng = np.random.default_rng(a.seed)
rng.shuffle(window_ids)
if a.windows_per_subject > 0:
    by_subject = defaultdict(list)
    for wid in window_ids:
        sid = str(subj_by_record[int(ds.windows[wid][0])])
        if len(by_subject[sid]) < a.windows_per_subject:
            by_subject[sid].append(wid)
    window_ids = [w for wl in by_subject.values() for w in wl]
    rng.shuffle(window_ids)
print(f"[split] {a.split}: {len(split[a.split])} subjects / {len(window_ids)} windows", flush=True)

loader = DataLoader(Subset(ds, window_ids), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
gen = torch.Generator().manual_seed(a.seed)
SCOPES = {"voxel": "voxel_window", "parcel": "middle_tokens", "network": "top_tokens"}
acc = {b: defaultdict(lambda: defaultdict(list)) for b in ("mean", "random")}
w0 = 0
for bi, batch in enumerate(loader):
    bsz = int(batch["voxel_window"].shape[0])
    sids = [str(subj_by_record[int(ds.windows[window_ids[w0 + i]][0])]) for i in range(bsz)]
    for scope, key in SCOPES.items():
        x = batch[key].float()                       # [B, T, U] (time on dim 1, as in _baseline_preds)
        mean_pred = x.mean(dim=1, keepdim=True).expand_as(x)
        flat = x.reshape(bsz, -1)
        mu = flat.mean(dim=1).view(bsz, *([1] * (x.dim() - 1)))
        sd = flat.std(dim=1, unbiased=False).clamp_min(1e-6).view(bsz, *([1] * (x.dim() - 1)))
        rnd = torch.randn(x.shape, generator=gen) * sd + mu
        for bname, pred in (("mean", mean_pred), ("random", rnd)):
            mse, cor = _batch_mse(pred, x), _batch_corr(pred, x)
            for i, sid in enumerate(sids):
                acc[bname][f"{scope}_mse"][sid].append(float(mse[i]))
                acc[bname][f"{scope}_corr"][sid].append(float(cor[i]))
    w0 += bsz
    if bi % 50 == 0:
        print(f"[{w0}/{len(window_ids)}]", flush=True)

out = {"split": f"{a.split}: {len(split[a.split])} subjects", "windows": len(window_ids), "time_axis_shape_check": {k: list(batch[v].shape) for k, v in SCOPES.items()}, "baselines": {}}
brng = np.random.default_rng(0)
for bname, d in acc.items():
    for k, per in d.items():
        subj = sorted(per); vals = np.array([np.mean(per[s]) for s in subj])
        boots = [vals[brng.integers(0, len(vals), len(vals))].mean() for _ in range(10000)]
        out["baselines"].setdefault(bname, {})[k] = {"mean": float(vals.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_subjects": len(subj)}
json.dump(out, open(a.output_json, "w"), indent=1)
print(json.dumps(out, indent=1))
