import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.fmri_multiscale.dataset import FMRIMultiScaleWindowDataset
from data.fmri_multiscale.splits import describe, load_split, window_indices_for_split
from models.fmri import (
    BrainStructuredTokenizerModel,
    FixedPriorTokenizerModel,
    LearnedHierarchyTokenizerModel,
    NPQChunkTokenizerModel,
    NPQParcelTokenizerModel,
    NPQHierResidualTokenizerModel,
    SingleLevelVQVAEModel,
)
from trainer_fmri import FMRITokenizerTrainer, TrainState, append_jsonl


def _maybe_plot_metrics(metrics_path: str, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    keys = sorted(set(rows[0]["train"].keys()) | set(rows[0]["val"].keys()))
    fig, axes = plt.subplots(1, min(3, len(keys)), figsize=(15, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    plot_keys = [k for k in ["total", "recon", "vq"] if k in keys][: len(axes)]
    for ax, k in zip(axes, plot_keys):
        tr = [r["train"].get(k, np.nan) for r in rows]
        va = [r["val"].get(k, np.nan) for r in rows]
        ax.plot(epochs, tr, label=f"train_{k}")
        ax.plot(epochs, va, label=f"val_{k}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(k)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(txt)
    except Exception:
        # allow json fallback when pyyaml is unavailable
        return json.loads(txt)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(n: int, val_fraction: float, seed: int, split_shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(n)
    if split_shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(all_idx)
    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n_val, n - 1))
    return all_idx[n_val:], all_idx[:n_val]


def _build_model(cfg: Dict[str, Any], sample_batch: Dict[str, torch.Tensor]) -> torch.nn.Module:
    model_type = cfg["model"]["type"]
    t_window = int(sample_batch["bottom_tokens"].shape[2])
    k_voxels = int(sample_batch["bottom_tokens"].shape[3])
    n_parcels = int(sample_batch["middle_tokens"].shape[2])
    n_networks = int(sample_batch["top_tokens"].shape[2])
    common = dict(
        t_window=t_window,
        k_voxels=k_voxels,
        token_dim=int(cfg["model"].get("token_dim", 256)),
        bottom_depth=int(cfg["model"].get("bottom_depth", 3)),
        middle_depth=int(cfg["model"].get("middle_depth", 2)),
        top_depth=int(cfg["model"].get("top_depth", 1)),
        num_heads=int(cfg["model"].get("num_heads", 8)),
        middle_codebook_size=int(cfg["model"].get("middle_codebook_size", 1024)),
        top_codebook_size=int(cfg["model"].get("top_codebook_size", 256)),
        commitment_weight=float(cfg["model"].get("commitment_weight", 0.25)),
    )
    if model_type == "model1":
        return FixedPriorTokenizerModel(
            n_parcels=n_parcels,
            n_networks=n_networks,
            **common,
        )
    if model_type == "model2":
        return LearnedHierarchyTokenizerModel(
            n_middle=int(cfg["model"].get("n_middle", n_parcels)),
            n_top=int(cfg["model"].get("n_top", n_networks)),
            decoder_use_bottom_skip=bool(cfg["model"].get("decoder_use_bottom_skip", True)),
            quantizer=str(cfg["model"].get("quantizer", "vq")),
            quant_dim=int(cfg["model"].get("quant_dim", 32)),
            middle_rvq_sizes=cfg["model"].get("middle_rvq_sizes"),
            top_rvq_sizes=cfg["model"].get("top_rvq_sizes"),
            n_chunks=int(sample_batch["bottom_tokens"].shape[1]) if cfg["model"].get("chunk_pos_embed", False) else 0,
            middle_identity=bool(cfg["model"].get("middle_identity", False)),
            **common,
        )
    if model_type == "model3":
        return BrainStructuredTokenizerModel(
            n_middle=int(cfg["model"].get("n_middle", n_parcels)),
            n_top=int(cfg["model"].get("n_top", n_networks)),
            n_parcels_target=n_parcels,
            n_networks_target=n_networks,
            decoder_use_bottom_skip=bool(cfg["model"].get("decoder_use_bottom_skip", True)),
            **common,
        )
    if model_type == "vqvae_single":
        return SingleLevelVQVAEModel(
            t_window=t_window,
            k_voxels=k_voxels,
            token_dim=int(cfg["model"].get("token_dim", 192)),
            depth=int(cfg["model"].get("depth", 3)),
            num_heads=int(cfg["model"].get("num_heads", 8)),
            codebook_size=int(cfg["model"].get("codebook_size", 1024)),
            commitment_weight=float(cfg["model"].get("commitment_weight", 0.25)),
        )
    if model_type == "npq_parcel":
        mcfg = cfg["model"]
        t_win = int(sample_batch["middle_tokens"].shape[1])
        n_par = int(sample_batch["middle_tokens"].shape[2])
        return NPQParcelTokenizerModel(
            t_window=t_win,
            n_parcels=n_par,
            token_dim=int(mcfg.get("token_dim", 192)),
            quant_dim=int(mcfg.get("quant_dim", 18)),
            encoder_depth=int(mcfg.get("encoder_depth", 3)),
            decoder_depth=int(mcfg.get("decoder_depth", 0)),
            num_heads=int(mcfg.get("num_heads", 8)),
            bsq_beta=float(mcfg.get("bsq_beta", 0.1)),
            bsq_gamma0=float(mcfg.get("bsq_gamma0", 1.0)),
            bsq_gamma=float(mcfg.get("bsq_gamma", 1.0)),
            bsq_zeta=float(mcfg.get("bsq_zeta", 1.0)),
            bsq_group_size=int(mcfg.get("bsq_group_size", 9)),
            bsq_l2_norm=bool(mcfg.get("bsq_l2_norm", False)),
            bsq_inv_temperature=float(mcfg.get("bsq_inv_temperature", 1.0)),
            bsq_persample_entropy_compute=str(mcfg.get("bsq_persample_entropy_compute", "analytical")),
        )
    if model_type == "npq_chunk":
        mcfg = cfg["model"]
        return NPQChunkTokenizerModel(
            t_window=t_window,
            k_voxels=k_voxels,
            token_dim=int(mcfg.get("token_dim", 192)),
            quant_dim=int(mcfg.get("quant_dim", 18)),
            encoder_depth=int(mcfg.get("encoder_depth", 3)),
            decoder_depth=int(mcfg.get("decoder_depth", 0)),
            num_heads=int(mcfg.get("num_heads", 8)),
            bsq_beta=float(mcfg.get("bsq_beta", 0.1)),
            bsq_gamma0=float(mcfg.get("bsq_gamma0", 1.0)),
            bsq_gamma=float(mcfg.get("bsq_gamma", 1.0)),
            bsq_zeta=float(mcfg.get("bsq_zeta", 1.0)),
            bsq_group_size=int(mcfg.get("bsq_group_size", 9)),
            bsq_l2_norm=bool(mcfg.get("bsq_l2_norm", False)),
            bsq_inv_temperature=float(mcfg.get("bsq_inv_temperature", 1.0)),
            bsq_persample_entropy_compute=str(mcfg.get("bsq_persample_entropy_compute", "analytical")),
        )
    if model_type == "npq_hier_residual":
        mcfg = cfg["model"]
        return NPQHierResidualTokenizerModel(
            t_window=t_window,
            k_voxels=k_voxels,
            n_parcels=int(mcfg.get("n_parcels", n_parcels)),
            n_networks=int(mcfg.get("n_networks", n_networks)),
            token_dim=int(mcfg.get("token_dim", 192)),
            bottom_depth=int(mcfg.get("bottom_depth", 3)),
            middle_depth=int(mcfg.get("middle_depth", 2)),
            top_depth=int(mcfg.get("top_depth", 1)),
            num_heads=int(mcfg.get("num_heads", 8)),
            bottom_quant_dim=int(mcfg.get("bottom_quant_dim", 16)),
            middle_quant_dim=int(mcfg.get("middle_quant_dim", 10)),
            top_quant_dim=int(mcfg.get("top_quant_dim", 6)),
            bottom_group_size=int(mcfg.get("bottom_group_size", 8)),
            middle_group_size=int(mcfg.get("middle_group_size", 5)),
            top_group_size=int(mcfg.get("top_group_size", 3)),
            bsq_beta=float(mcfg.get("bsq_beta", 0.1)),
            bsq_gamma0=float(mcfg.get("bsq_gamma0", 1.0)),
            bsq_gamma=float(mcfg.get("bsq_gamma", 1.0)),
            bsq_zeta=float(mcfg.get("bsq_zeta", 1.0)),
            bsq_l2_norm=bool(mcfg.get("bsq_l2_norm", False)),
            bsq_inv_temperature=float(mcfg.get("bsq_inv_temperature", 1.0)),
            bsq_persample_entropy_compute=str(mcfg.get("bsq_persample_entropy_compute", "analytical")),
            bsq_cb_entropy_scope=str(mcfg.get("bsq_cb_entropy_scope", "batch")),
            n_time_tokens_bottom=int(mcfg.get("n_time_tokens_bottom", 1)),
            n_time_tokens_middle=int(mcfg.get("n_time_tokens_middle", 1)),
            n_time_tokens_top=int(mcfg.get("n_time_tokens_top", 1)),
            bottom_encoder_hidden=int(mcfg.get("bottom_encoder_hidden", 64)),
            bottom_temporal_summary=str(mcfg.get("bottom_temporal_summary", "attn_pool")),
            bottom_temporal_pool_slots=(
                int(mcfg["bottom_temporal_pool_slots"])
                if mcfg.get("bottom_temporal_pool_slots") is not None
                else None
            ),
            bottom_recon_head_layers=int(mcfg.get("bottom_recon_head_layers", 1)),
            bottom_use_voxel_slots=bool(mcfg.get("bottom_use_voxel_slots", False)),
            bottom_use_parcel_cond=bool(mcfg.get("bottom_use_parcel_cond", False)),
            middle_recon_head_layers=int(mcfg.get("middle_recon_head_layers", 1)),
            middle_use_parcel_slots=bool(mcfg.get("middle_use_parcel_slots", False)),
            middle_resid_loss=bool(mcfg.get("middle_resid_loss", False)),
            middle_detach_from_bottom=bool(mcfg.get("middle_detach_from_bottom", False)),
            middle_bsq_inv_temperature=(
                float(mcfg["middle_bsq_inv_temperature"])
                if mcfg.get("middle_bsq_inv_temperature") is not None
                else None
            ),
            cross_scale_mode=str(mcfg.get("cross_scale_mode", "fixed")),
            cross_scale_pool_num_heads=(
                int(mcfg["cross_scale_pool_num_heads"])
                if mcfg.get("cross_scale_pool_num_heads") is not None
                else None
            ),
            learned_atlas_aux=bool(mcfg.get("learned_atlas_aux", False)),
            input_mode=str(mcfg.get("input_mode", "voxel")),
            n_levels=int(mcfg.get("n_levels", 3)),
            shuffle_parcel_to_network_seed=(
                int(mcfg["shuffle_parcel_to_network_seed"])
                if mcfg.get("shuffle_parcel_to_network_seed") is not None
                else None
            ),
            shuffle_chunk_to_parcel_seed=(
                int(mcfg["shuffle_chunk_to_parcel_seed"])
                if mcfg.get("shuffle_chunk_to_parcel_seed") is not None
                else None
            ),
            quantizer=str(mcfg.get("quantizer", "bsq")),
            rvq_codebook_sizes=(
                [int(k) for k in mcfg["rvq_codebook_sizes"]]
                if mcfg.get("rvq_codebook_sizes") is not None
                else None
            ),
            rvq_commitment=float(mcfg.get("rvq_commitment", 0.25)),
            rvq_ema_decay=float(mcfg.get("rvq_ema_decay", 0.99)),
            rvq_dead_code_steps=int(mcfg.get("rvq_dead_code_steps", 200)),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _unwrap_raw_model(model: torch.nn.Module) -> torch.nn.Module:
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _set_backbone_requires_grad(model: torch.nn.Module, freeze_backbone: bool) -> None:
    """When freeze_backbone=True, only CrossAttentionPool parameters receive gradients."""
    raw = _unwrap_raw_model(model)
    pool_prefixes = ("chunk_to_parcel_pool.", "parcel_to_network_pool.")
    for name, p in raw.named_parameters():
        if any(name.startswith(px) for px in pool_prefixes):
            p.requires_grad = True
        else:
            p.requires_grad = not bool(freeze_backbone)


def _parse_train_stages(train_cfg: Dict[str, Any]) -> Optional[list]:
    raw = train_cfg.get("train_stages")
    if not raw:
        return None
    out = []
    for i, block in enumerate(raw):
        if "epochs" not in block:
            raise ValueError(f"train_stages[{i}] missing required key 'epochs'")
        out.append(dict(block))
    return out


def _stage_index_merged(train_cfg: Dict[str, Any], stages: list, epoch: int) -> Tuple[int, Dict[str, Any], str]:
    acc = 0
    for idx, st in enumerate(stages):
        e = int(st["epochs"])
        if epoch < acc + e:
            merged = dict(train_cfg)
            merged["lr"] = float(st.get("lr", train_cfg.get("lr", 2e-4)))
            if "pool_lr" in st:
                merged["pool_lr"] = float(st["pool_lr"])
            elif train_cfg.get("pool_lr") is not None:
                merged["pool_lr"] = float(train_cfg["pool_lr"])
            else:
                merged.pop("pool_lr", None)
            merged["freeze_backbone"] = bool(st.get("freeze_backbone", False))
            name = str(st.get("name", f"stage_{idx}"))
            return idx, merged, name
        acc += e
    st = stages[-1]
    merged = dict(train_cfg)
    merged["lr"] = float(st.get("lr", train_cfg.get("lr", 2e-4)))
    if "pool_lr" in st:
        merged["pool_lr"] = float(st["pool_lr"])
    elif train_cfg.get("pool_lr") is not None:
        merged["pool_lr"] = float(train_cfg["pool_lr"])
    else:
        merged.pop("pool_lr", None)
    merged["freeze_backbone"] = bool(st.get("freeze_backbone", False))
    return len(stages) - 1, merged, str(st.get("name", "last"))


def _make_adamw_optimizer(model: torch.nn.Module, merged_train: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(merged_train.get("lr", 2e-4))
    wd = float(merged_train.get("weight_decay", 0.01))
    fused_opt = bool(merged_train.get("fused_adamw", True)) and torch.cuda.is_available()
    pool_lr = merged_train.get("pool_lr", None)

    if pool_lr is not None:
        pool_lr = float(pool_lr)
        pool_prefixes = ("chunk_to_parcel_pool.", "parcel_to_network_pool.")
        raw = _unwrap_raw_model(model)
        pool_params, backbone_params = [], []
        for name, p in raw.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(px) for px in pool_prefixes):
                pool_params.append(p)
            else:
                backbone_params.append(p)
        if not pool_params:
            groups = [{"params": backbone_params, "lr": lr}]
        elif not backbone_params:
            groups = [{"params": pool_params, "lr": pool_lr}]
        else:
            groups = [
                {"params": backbone_params, "lr": lr},
                {"params": pool_params, "lr": pool_lr},
            ]
        if fused_opt:
            try:
                return torch.optim.AdamW(groups, weight_decay=wd, fused=True)
            except TypeError:
                return torch.optim.AdamW(groups, weight_decay=wd)
        return torch.optim.AdamW(groups, weight_decay=wd)
    params = [p for p in model.parameters() if p.requires_grad]
    if fused_opt:
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
        except TypeError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def _lambdas_for_epoch(model_cfg: Dict[str, Any], epoch: int, num_epochs: int) -> Dict[str, float]:
    lam = dict(model_cfg["loss"]["lambdas"])
    sch = model_cfg.get("loss", {}).get("align_scale_schedule")
    if not sch or num_epochs <= 0:
        return lam
    kind = str(sch.get("kind", "linear")).lower()
    start = float(sch.get("start", lam.get("align_scale", 0.0)))
    end = float(sch.get("end", 0.0))
    if kind == "constant":
        lam["align_scale"] = float(sch.get("value", start))
        return lam
    if num_epochs <= 1:
        lam["align_scale"] = start
        return lam
    t = epoch / float(num_epochs - 1)
    lam["align_scale"] = start + t * (end - start)
    return lam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-scale fMRI tokenizer models.")
    p.add_argument("--config", type=str, required=True, help="Path to model YAML config.")
    p.add_argument("--data_config", type=str, required=True, help="Path to data YAML config.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=None,
        help=(
            "Override train.grad_accum_steps from the config. Use this to hold the "
            "effective batch size fixed when changing the number of GPUs."
        ),
    )
    p.add_argument(
        "--init_from",
        type=str,
        default=None,
        help=(
            "Path to a .pth checkpoint to warm-start weights from (strict=False). "
            "Keys present in the checkpoint but not in the model (or vice-versa) are "
            "silently skipped — useful for loading a fixed-atlas baseline into a "
            "learned/soft model where the CrossAttentionPool keys are new."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the newest ckpt_ep*.pth in --output_dir (weights, optimizer, "
            "epoch, step). No-op if none exists, so it is safe to pass on every launch. "
            "When a checkpoint is found, --init_from is ignored."
        ),
    )
    return p.parse_args()


def _latest_checkpoint(out_dir: str) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    ckpts = [f for f in os.listdir(out_dir) if f.startswith("ckpt_ep") and f.endswith(".pth")]
    if not ckpts:
        return None
    # Names are ckpt_ep{epoch:04d}_it{step:06d}.pth, so lexical order is epoch order.
    return os.path.join(out_dir, max(ckpts))


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed() -> Tuple[bool, int, int]:
    if not _is_distributed():
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, world_size


def _cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _reduce_scalar(value: float, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(dist.get_world_size())
    return float(t.item())


def _init_wandb(
    *,
    is_main: bool,
    args: argparse.Namespace,
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
) -> Optional[Any]:
    if not is_main:
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not installed; continuing without wandb logging.")
        return None

    run_name = os.environ.get("WANDB_RUN_NAME", os.path.basename(args.output_dir.rstrip("/")))
    project = os.environ.get("WANDB_PROJECT", "hbar-fmri")
    entity = os.environ.get("WANDB_ENTITY")
    tags = [str(model_cfg.get("model", {}).get("type", "unknown-model"))]
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        config={"model": model_cfg, "data": data_cfg, "seed": args.seed, "output_dir": args.output_dir},
    )
    return run


def main() -> None:
    args = parse_args()
    dist_enabled, local_rank, world_size = _setup_distributed()
    is_main = (not dist_enabled) or (dist.get_rank() == 0)
    model_cfg = _load_config(args.config)
    data_cfg = _load_config(args.data_config)
    _set_seed(args.seed + local_rank)
    train_cfg = model_cfg.get("train", {})

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("enable_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("enable_tf32", True))
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(train_cfg.get("enable_flash_sdp", True)))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(bool(train_cfg.get("enable_mem_efficient_sdp", True)))
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(bool(train_cfg.get("enable_math_sdp", False)))
    torch.set_float32_matmul_precision(str(train_cfg.get("matmul_precision", "high")))

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump({"model": model_cfg, "data": data_cfg, "seed": args.seed}, f, indent=2)

    ds = FMRIMultiScaleWindowDataset(
        cache_dir=None,
        index_path=data_cfg["data"]["index_path"],
        layout_dir=data_cfg["data"]["layout_dir"],
        window_size=int(data_cfg["data"].get("window_size", 32)),
        stride=int(data_cfg["data"].get("stride", 16)),
        cache_items=int(data_cfg["data"].get("cache_items", 8)),
        max_windows_per_subject=data_cfg["data"].get("max_windows_per_subject"),
    )
    split_json = data_cfg["data"].get("subject_split_json")
    split_shuffle = bool(data_cfg["data"].get("split_shuffle", True))
    if split_json:
        split = load_split(str(split_json))
        tr_idx = window_indices_for_split(ds, split, "train")
        va_idx = window_indices_for_split(ds, split, "val")
        split_desc = f"subject-level ({split_json}): {describe(split, ds)}"
    else:
        tr_idx, va_idx = _split_indices(
            len(ds),
            val_fraction=float(data_cfg["data"].get("val_fraction", 0.1)),
            seed=args.seed,
            split_shuffle=split_shuffle,
        )
        split_desc = (
            f"LEGACY window-level (val_fraction="
            f"{float(data_cfg['data'].get('val_fraction', 0.1))}, split_shuffle={split_shuffle}); "
            "subjects may span train and val -- set data.subject_split_json for a subject-disjoint split"
        )
    tr_set = Subset(ds, tr_idx.tolist())
    va_set = Subset(ds, va_idx.tolist())

    batch_size = int(model_cfg["train"].get("batch_size", 2))
    num_workers = int(model_cfg["train"].get("num_workers", 0))
    pin_memory = bool(model_cfg["train"].get("pin_memory", True))
    persistent_workers = bool(model_cfg["train"].get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(model_cfg["train"].get("prefetch_factor", 2)) if num_workers > 0 else None
    shuffle_train = bool(data_cfg["data"].get("shuffle_train", True))
    tr_sampler = DistributedSampler(tr_set, shuffle=shuffle_train) if dist_enabled else None
    va_sampler = DistributedSampler(va_set, shuffle=False) if dist_enabled else None
    tr_loader = DataLoader(
        tr_set,
        batch_size=batch_size,
        shuffle=(tr_sampler is None and shuffle_train),
        sampler=tr_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    va_loader = DataLoader(
        va_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=va_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    if is_main:
        print(
            f"[data] config={args.data_config} "
            f"index_path={data_cfg['data']['index_path']} "
            f"layout_dir={data_cfg['data']['layout_dir']}"
        )
        print(f"[split] {split_desc}")
        print(f"Dataset windows: train={len(tr_set)}, val={len(va_set)}, world_size={world_size}")
        if split_json:
            with open(os.path.join(args.output_dir, "split_used.json"), "w", encoding="utf-8") as f:
                json.dump(split, f, indent=2)

    sample_batch = next(iter(tr_loader))
    model = _build_model(model_cfg, sample_batch)
    if dist_enabled:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    model = model.to(device)
    resume_ckpt = None
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = _latest_checkpoint(args.output_dir)
        if resume_path is None:
            if is_main:
                print(f"[resume] no checkpoint in {args.output_dir}; starting from scratch")
        else:
            resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(resume_ckpt["model"])
            start_epoch = int(resume_ckpt["state"]["epoch"]) + 1
            global_step = int(resume_ckpt["state"]["step"])
            if is_main:
                print(f"[resume] loaded {resume_path}: start_epoch={start_epoch} global_step={global_step}")
                if args.init_from:
                    print("[resume] ignoring --init_from (weights come from the resume checkpoint)")
    if args.init_from and resume_ckpt is None:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        cleaned: Dict[str, Any] = {}
        for k, v in state.items():
            name = k
            if name.startswith("module."):
                name = name[len("module."):]
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
            cleaned[name] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if is_main:
            print(f"[init_from] loaded weights from {args.init_from}")
            print(f"[init_from] missing keys (new in model): {len(missing)}  unexpected keys (not in model): {len(unexpected)}")
            if missing:
                print(f"[init_from]   missing (first 10): {missing[:10]}")
            if unexpected:
                print(f"[init_from]   unexpected (first 10): {unexpected[:10]}")
    if is_main:
        total_params, trainable_params = _count_parameters(model)
        print(
            f"[model] type={model_cfg['model']['type']} "
            f"total_params={total_params:,} trainable_params={trainable_params:,}"
        )
    compile_enabled = bool(train_cfg.get("compile", False))
    if compile_enabled and hasattr(torch, "compile"):
        compile_mode = str(train_cfg.get("compile_mode", "default"))
        compile_fullgraph = bool(train_cfg.get("compile_fullgraph", False))
        if is_main:
            print(f"[compile] enabling torch.compile(mode={compile_mode}, fullgraph={compile_fullgraph})")
        model = torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph)
    stages = _parse_train_stages(train_cfg)
    total_epochs = sum(int(s["epochs"]) for s in stages) if stages else int(train_cfg.get("epochs", 10))
    if stages and is_main:
        print(f"[train_stages] using {len(stages)} stages, total_epochs={total_epochs} (ignores train.epochs)")
    if dist_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(stages),
        )

    if stages:
        _, merged0, s0name = _stage_index_merged(train_cfg, stages, start_epoch)
        _set_backbone_requires_grad(model, merged0.get("freeze_backbone", False))
        optimizer = _make_adamw_optimizer(model, merged0)
        if is_main:
            print(
                f"[train_stages] initial stage={s0name!r} freeze_backbone={merged0.get('freeze_backbone')} "
                f"lr={merged0.get('lr')} pool_lr={merged0.get('pool_lr', 'n/a')}"
            )
    else:
        _set_backbone_requires_grad(model, False)
        optimizer = _make_adamw_optimizer(model, dict(train_cfg))

    # Keep wandb init after dataset/model setup so startup is not blocked on network.
    wandb_run = _init_wandb(is_main=is_main, args=args, model_cfg=model_cfg, data_cfg=data_cfg)

    trainer = FMRITokenizerTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_type=model_cfg["model"]["type"],
        lambdas=_lambdas_for_epoch(model_cfg, 0, total_epochs),
        dep_mask_ratio=float(model_cfg["loss"].get("dep_mask_ratio", 0.15)),
        use_amp=bool(model_cfg["train"].get("use_amp", True)),
        amp_dtype=str(model_cfg["train"].get("amp_dtype", "bf16")),
        is_main_process=is_main,
        distributed=dist_enabled,
        wandb_run=wandb_run,
        wandb_log_every=int(os.environ.get("WANDB_LOG_EVERY", "10")),
        vq_commitment_warmup_steps=int(train_cfg.get("vq_commitment_warmup_steps", 0)),
        vq_commitment_start=float(train_cfg.get("vq_commitment_start", 0.0)),
        vq_commitment_end=float(train_cfg.get("vq_commitment_end", 0.0)),
        bsq_entropy_warmup_steps=int(train_cfg.get("bsq_entropy_warmup_steps", 0)),
        bsq_zeta_start=float(train_cfg.get("bsq_zeta_start", 0.0)),
        bsq_zeta_end=float(train_cfg.get("bsq_zeta_end", 1.0)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
        skip_nonfinite_loss=bool(train_cfg.get("skip_nonfinite_loss", True)),
        grad_accum_steps=int(
            args.grad_accum_steps
            if args.grad_accum_steps is not None
            else train_cfg.get("grad_accum_steps", 1)
        ),
        chunk_loss_fn=str(train_cfg.get("chunk_loss_fn", "mse")),
    )

    if resume_ckpt is not None:
        trainer.optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scaler" in resume_ckpt:
            trainer.scaler.load_state_dict(resume_ckpt["scaler"])
        # Reseed so the resumed epoch does not replay epoch 0's batch order.
        _set_seed(args.seed + local_rank + 1000 * start_epoch)
        del resume_ckpt

    ckpt_every = int(train_cfg.get("ckpt_every", 1))
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    prev_stage_idx: Optional[int] = None
    for ep in range(start_epoch, total_epochs):
        stage_name = "train"
        if stages:
            stage_idx, merged_ep, stage_name = _stage_index_merged(train_cfg, stages, ep)
            if prev_stage_idx is None:
                prev_stage_idx = stage_idx
            elif stage_idx != prev_stage_idx:
                _set_backbone_requires_grad(model, merged_ep.get("freeze_backbone", False))
                trainer.optimizer = _make_adamw_optimizer(model, merged_ep)
                prev_stage_idx = stage_idx
                if is_main:
                    print(
                        f"[train_stages] -> stage={stage_name!r} ep={ep} "
                        f"freeze_backbone={merged_ep.get('freeze_backbone')} "
                        f"lr={merged_ep.get('lr')} pool_lr={merged_ep.get('pool_lr', 'n/a')}"
                    )
        trainer.lambdas = _lambdas_for_epoch(model_cfg, ep, total_epochs)
        align_eff = float(trainer.lambdas.get("align_scale", 0.0))
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        if tr_sampler is not None and shuffle_train:
            tr_sampler.set_epoch(ep)
        train_start = time.perf_counter()
        tr_stats = trainer.train_one_epoch(
            tr_loader,
            epoch=ep,
            log_interval=int(model_cfg["train"].get("log_interval", 20)),
            global_step_start=global_step,
        )
        train_time = time.perf_counter() - train_start
        global_step += len(tr_loader)
        val_start = time.perf_counter()
        va_stats = trainer.validate(va_loader, epoch=ep)
        val_time = time.perf_counter() - val_start
        epoch_time = time.perf_counter() - epoch_start
        tr_stats = {k: _reduce_scalar(v, device=device, enabled=dist_enabled) for k, v in tr_stats.items()}
        va_stats = {k: _reduce_scalar(v, device=device, enabled=dist_enabled) for k, v in va_stats.items()}
        row = {
            "time": datetime.utcnow().isoformat(),
            "epoch": ep,
            "step": global_step,
            "train": tr_stats,
            "val": va_stats,
            "timing": {"epoch_time_s": epoch_time, "train_time_s": train_time, "val_time_s": val_time},
            "train_stage": stage_name,
            "align_scale_lambda_effective": align_eff,
        }
        if torch.cuda.is_available():
            row["gpu_mem"] = {
                "alloc_gb": float(torch.cuda.memory_allocated(device) / (1024**3)),
                "reserved_gb": float(torch.cuda.memory_reserved(device) / (1024**3)),
                "max_alloc_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
                "max_reserved_gb": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
            }
        if is_main:
            train_time_str = str(timedelta(seconds=int(round(train_time))))
            print(f"[train][ep={ep}] Total time: {train_time_str} ({train_time / max(len(tr_loader), 1):.4f} s / it)")
            append_jsonl(metrics_path, row)
            print(f"[epoch][ep={ep}] epoch_time={epoch_time:.2f}s train_time={train_time:.2f}s val_time={val_time:.2f}s")
            if "gpu_mem" in row:
                gm = row["gpu_mem"]
                print(
                    "[gpu][ep={}] alloc={:.2f}G reserved={:.2f}G peak_alloc={:.2f}G peak_reserved={:.2f}G".format(
                        ep, gm["alloc_gb"], gm["reserved_gb"], gm["max_alloc_gb"], gm["max_reserved_gb"]
                    )
                )
            if wandb_run is not None:
                wb_metrics = {
                    "epoch": ep,
                    "step": global_step,
                    "train_stage": stage_name,
                    "align_scale_lambda_effective": align_eff,
                    **{f"train/{k}": v for k, v in tr_stats.items()},
                    **{f"val/{k}": v for k, v in va_stats.items()},
                }
                if "gpu_mem" in row:
                    wb_metrics.update({f"gpu/{k}": v for k, v in row["gpu_mem"].items()})
                wandb_run.log(wb_metrics, step=global_step)
            if (ep + 1) % ckpt_every == 0 or (ep + 1) == total_epochs:
                ckpt = trainer.save_checkpoint(args.output_dir, TrainState(epoch=ep, step=global_step), extra={"model_type": model_cfg["model"]["type"]})
                print(f"[checkpoint] {ckpt}")

    if is_main:
        _maybe_plot_metrics(metrics_path=metrics_path, out_dir=args.output_dir)
        print(f"Training complete. Outputs: {args.output_dir}")
        if wandb_run is not None:
            wandb_run.finish()
    _cleanup_distributed(dist_enabled)


if __name__ == "__main__":
    main()

