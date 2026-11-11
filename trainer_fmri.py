import contextlib
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fmri.losses import (
    alignment_losses,
    dependency_loss_masked_parcel,
    recon_delta_mse,
    recon_mse,
    weighted_total_loss,
)

_STATIC_BATCH_KEYS = {"chunk_indices", "token_parcels", "token_networks", "parcel_to_network"}


@dataclass
class TrainState:
    epoch: int
    step: int


class FMRITokenizerTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_type: str,
        lambdas: Dict[str, float],
        dep_mask_ratio: float = 0.15,
        use_amp: bool = True,
        is_main_process: bool = True,
        distributed: bool = False,
        wandb_run=None,
        wandb_log_every: int = 10,
        amp_dtype: str = "fp16",
        vq_commitment_warmup_steps: int = 0,
        vq_commitment_start: float = 0.0,
        vq_commitment_end: float = 0.0,
        bsq_entropy_warmup_steps: int = 0,
        bsq_zeta_start: float = 0.0,
        bsq_zeta_end: float = 1.0,
        grad_clip_norm: float = 0.0,
        skip_nonfinite_loss: bool = True,
        grad_accum_steps: int = 1,
        chunk_loss_fn: str = "mse",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.lambdas = lambdas
        self.dep_mask_ratio = dep_mask_ratio
        self.is_main_process = is_main_process
        self.distributed = distributed
        self.wandb_run = wandb_run
        self.wandb_log_every = max(1, int(wandb_log_every))
        self.vq_commitment_warmup_steps = max(0, int(vq_commitment_warmup_steps))
        self.vq_commitment_start = float(vq_commitment_start)
        self.vq_commitment_end = float(vq_commitment_end)
        self.bsq_entropy_warmup_steps = max(0, int(bsq_entropy_warmup_steps))
        self.bsq_zeta_start = float(bsq_zeta_start)
        self.bsq_zeta_end = float(bsq_zeta_end)
        self.grad_clip_norm = float(grad_clip_norm)
        self.skip_nonfinite_loss = bool(skip_nonfinite_loss)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        _valid_chunk_loss_fns = {"mse", "l1", "huber"}
        if chunk_loss_fn not in _valid_chunk_loss_fns:
            raise ValueError(f"chunk_loss_fn must be one of {_valid_chunk_loss_fns}, got {chunk_loss_fn!r}")
        self.chunk_loss_fn = chunk_loss_fn
        self._nonfinite_skips = 0
        self.use_amp = use_amp and torch.cuda.is_available()
        self._amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        amp_dtype = str(amp_dtype).lower()
        self._autocast_dtype = torch.float16
        if amp_dtype in {"bf16", "bfloat16"}:
            self._autocast_dtype = torch.bfloat16
        self._use_grad_scaler = self.use_amp and self._autocast_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler(self._amp_device_type, enabled=self._use_grad_scaler)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_grad_scaler)

    def _apply_vq_commitment_warmup(self, global_step: int):
        if self.vq_commitment_warmup_steps <= 0:
            return None
        frac = min(max(float(global_step), 0.0), float(self.vq_commitment_warmup_steps)) / float(
            self.vq_commitment_warmup_steps
        )
        value = self.vq_commitment_start + frac * (self.vq_commitment_end - self.vq_commitment_start)

        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        for mod in model_obj.modules():
            # VectorQuantizerEMA commitment.
            if hasattr(mod, "commitment_weight"):
                mod.commitment_weight = float(value)
            # BSQ commitment.
            if hasattr(mod, "beta"):
                mod.beta = float(value)
        return value

    def _apply_bsq_entropy_warmup(self, global_step: int):
        """Ramp BSQ `zeta` (entropy penalty weight) from start -> end over warmup steps.

        Stacked with the existing commit warmup: commit grows gradually so the
        encoder has room to learn, and the entropy (codebook diversity) pressure
        is held low early and ramped up once recon starts landing.  Without this
        ramp the entropy term dominates the vq total from step 0 and starves the
        reconstruction gradient, which is what the early v1 NPQ runs hit."""
        if self.bsq_entropy_warmup_steps <= 0:
            return None
        frac = min(max(float(global_step), 0.0), float(self.bsq_entropy_warmup_steps)) / float(
            self.bsq_entropy_warmup_steps
        )
        value = self.bsq_zeta_start + frac * (self.bsq_zeta_end - self.bsq_zeta_start)
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        for mod in model_obj.modules():
            # BSQ entropy weight.
            if hasattr(mod, "zeta") and hasattr(mod, "gamma") and hasattr(mod, "gamma0"):
                mod.zeta = float(value)
        return value

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if k in _STATIC_BATCH_KEYS and v.ndim >= 2:
                    # Layout tensors are identical across samples; keep one copy.
                    v = v[0]
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _losses_for_batch(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if self.model_type == "npq_parcel":
            p_true = batch["middle_tokens"].float()
            p_hat = out["parcel_ts_recon"]
            losses["recon"] = recon_mse(p_true, p_hat)
            if float(self.lambdas.get("delta", 0.0)) > 0.0:
                losses["delta"] = recon_delta_mse(p_true, p_hat)
        else:
            x_true = batch["voxel_window"]  # [B,T,V]
            x_hat = out["voxel_recon"]
            losses["recon"] = recon_mse(x_true, x_hat)
            if float(self.lambdas.get("delta", 0.0)) > 0.0:
                losses["delta"] = recon_delta_mse(x_true, x_hat)
        # Shallow hierarchies (n_levels < 3) have no middle and/or top level, so
        # sum whichever BSQ levels the model actually produced.  Order is kept
        # (middle, top, bottom) so the three-level value is unchanged.
        vq_total = None
        for _vq_key in ("vq_middle", "vq_top", "vq_bottom"):
            _vq = out.get(_vq_key)
            if torch.is_tensor(_vq):
                vq_total = _vq if vq_total is None else vq_total + _vq
        if vq_total is None:
            vq_total = torch.zeros((), device=self.device)
        losses["vq"] = vq_total

        chunk_losses_apply = self.model_type == "npq_hier_residual" or (
            # Flat BSQ chunk baseline: opt-in chunk-level supervision so it can be
            # given the same reconstruction/temporal objective as the hierarchy.
            # Existing npq_chunk configs have no `recon_chunk` λ and are unaffected.
            self.model_type == "npq_chunk" and float(self.lambdas.get("recon_chunk", 0.0)) > 0.0
        )
        if chunk_losses_apply:
            if "chunk_recon" in out:
                if self.chunk_loss_fn == "l1":
                    losses["recon_chunk"] = torch.nn.functional.l1_loss(out["chunk_recon"], batch["bottom_tokens"])
                elif self.chunk_loss_fn == "huber":
                    losses["recon_chunk"] = torch.nn.functional.huber_loss(out["chunk_recon"], batch["bottom_tokens"])
                else:
                    losses["recon_chunk"] = torch.nn.functional.mse_loss(out["chunk_recon"], batch["bottom_tokens"])
                if float(self.lambdas.get("delta_chunk", 0.0)) > 0.0:
                    # Penalise temporal gradient mismatches within each chunk window.
                    # chunk_recon / bottom_tokens are [B, N, T, K] or [B, T_window, K].
                    # Diff along the time dimension (dim=-2) to get T-1 differences.
                    cr = out["chunk_recon"]
                    bt = batch["bottom_tokens"]
                    losses["delta_chunk"] = torch.nn.functional.l1_loss(
                        cr.diff(dim=-2), bt.diff(dim=-2)
                    )
            # `parcel_target` / `network_target` are only produced by the shuffled-
            # grouping control, where the dataset's atlas summaries are no longer the
            # aggregate of the model's groups.  Otherwise the dataset targets are used.
            parcel_target = out.get("parcel_target", batch.get("middle_tokens"))
            if "parcel_pred" in out and parcel_target is not None:
                losses["recon_parcel"] = torch.nn.functional.mse_loss(out["parcel_pred"], parcel_target)
            network_target = out.get("network_target", batch.get("top_tokens"))
            if "network_pred" in out and network_target is not None:
                losses["recon_network"] = torch.nn.functional.mse_loss(out["network_pred"], network_target)
            if "align_scale" in out and torch.is_tensor(out["align_scale"]):
                losses["align_scale"] = out["align_scale"]
            if "parcel_resid_pred" in out:
                # Only emitted when the model has middle_resid_loss: true.
                losses["recon_parcel_resid"] = torch.nn.functional.mse_loss(
                    out["parcel_resid_pred"].float(), out["parcel_resid_target"]
                )

        if self.model_type == "model3":
            true_parcel = batch["middle_tokens"]  # [B,T,P]
            true_network = batch["top_tokens"]  # [B,T,N]
            l_align_p, l_align_n = alignment_losses(
                pred_parcel_btp=out["pred_parcel_summary"],
                pred_network_btn=out["pred_network_summary"],
                true_parcel_btp=true_parcel,
                true_network_btn=true_network,
            )
            losses["align"] = l_align_p + l_align_n

            l_cons_p = torch.nn.functional.mse_loss(out["pred_parcel_summary"], out["recon_parcel_summary"])
            l_cons_n = torch.nn.functional.mse_loss(out["pred_network_summary"], out["recon_network_summary"])
            losses["cons"] = l_cons_p + l_cons_n

            l_dep, dep_stats = dependency_loss_masked_parcel(
                pred_parcel_btp=out["dep_parcel_pred"],
                true_parcel_btp=true_parcel,
                mask_ratio=self.dep_mask_ratio,
            )
            losses["dep"] = l_dep
            losses["dep_mask_ratio_realized"] = dep_stats["dep_mask_ratio_realized"]

        losses["total"] = weighted_total_loss(
            losses={
                k: v
                for k, v in losses.items()
                if k in {
                    "recon",
                    "delta",
                    "vq",
                    "align",
                    "cons",
                    "dep",
                    "recon_chunk",
                    "delta_chunk",
                    "recon_parcel",
                    "recon_network",
                    "align_scale",
                    "recon_parcel_resid",
                }
            },
            lambdas=self.lambdas,
        )
        for mk in [
            "codebook_usage_middle",
            "codebook_usage_top",
            "codebook_usage_bottom",
            "codebook_perplexity_middle",
            "codebook_perplexity_top",
            "codebook_perplexity_bottom",
            "bsq_cb_entropy",
            "bsq_persample_entropy",
            "bsq_commit_loss",
            "bsq_cb_entropy_bottom",
            "bsq_cb_entropy_middle",
            "bsq_cb_entropy_top",
            "bsq_persample_entropy_bottom",
            "bsq_persample_entropy_middle",
            "bsq_persample_entropy_top",
            "bsq_commit_loss_bottom",
            "bsq_commit_loss_middle",
            "bsq_commit_loss_top",
            "codebook_usage_bottom_stage1",
            "codebook_perplexity_bottom_stage1",
            "atlas_latent_mse",
        ]:
            if mk in out and torch.is_tensor(out[mk]):
                losses[mk] = out[mk].detach()
        return losses

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds_int = max(0, int(round(seconds)))
        return str(timedelta(seconds=seconds_int))

    def train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        log_interval: int = 20,
        global_step_start: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        n_steps = 0
        total_steps = len(loader)
        loader_iter = iter(loader)
        global_iter_time = 0.0
        global_data_time = 0.0
        global_step_time = 0.0
        global_windows = 0.0
        interval_iter_time = 0.0
        interval_data_time = 0.0
        interval_step_time = 0.0
        interval_windows = 0.0
        interval_steps = 0
        ga = self.grad_accum_steps
        # Micro-step counter within the current accumulation window.  Reset to 0
        # on every optimizer step.
        accum_idx = 0
        # Track whether the pending accumulation window has been corrupted by a
        # non-finite micro-batch; if so, we zero_grad and skip the optimizer step
        # at the boundary instead of applying partial gradients.
        window_corrupt = False
        while True:
            data_t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            data_time = time.perf_counter() - data_t0
            n_steps += 1
            global_step = int(global_step_start + n_steps)
            vq_commitment_weight = self._apply_vq_commitment_warmup(global_step)
            bsq_zeta_weight = self._apply_bsq_entropy_warmup(global_step)
            is_boundary = (accum_idx + 1) == ga
            compute_t0 = time.perf_counter()
            batch = self._prepare_batch(batch)
            if accum_idx == 0:
                self.optimizer.zero_grad(set_to_none=True)
                window_corrupt = False
            try:
                autocast_ctx = torch.amp.autocast(self._amp_device_type, enabled=self.use_amp, dtype=self._autocast_dtype)
            except AttributeError:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
            # Under DDP, skip the all-reduce on non-boundary micro-steps so that
            # gradient-accumulation doesn't pay for a full gradient sync every
            # micro-batch.  The `no_sync` context is only valid when using DDP.
            if ga > 1 and self.distributed and hasattr(self.model, "no_sync") and not is_boundary:
                sync_ctx = self.model.no_sync()
            else:
                sync_ctx = contextlib.nullcontext()
            with sync_ctx:
                with autocast_ctx:
                    out = self.model(batch)
                    losses = self._losses_for_batch(batch, out)
                total_loss = losses["total"]
                loss_finite = bool(torch.isfinite(total_loss).item())
                if self.skip_nonfinite_loss and not loss_finite:
                    self._nonfinite_skips += 1
                    if self.is_main_process and (self._nonfinite_skips <= 5 or self._nonfinite_skips % 50 == 0):
                        print(
                            f"[warn] non-finite loss at global_step={global_step} "
                            f"(skip #{self._nonfinite_skips}); skipping optimizer step"
                        )
                    window_corrupt = True
                else:
                    # Scale loss so accumulated grads match a single big-batch step.
                    scaled_loss = total_loss / float(ga)
                    if self._use_grad_scaler:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

            # Only touch the optimizer at accumulation boundaries.
            if is_boundary:
                if window_corrupt:
                    # Drop the partially-accumulated window rather than applying
                    # gradients mixed with the non-finite micro-batch.
                    self.optimizer.zero_grad(set_to_none=True)
                elif self._use_grad_scaler:
                    if self.grad_clip_norm > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                accum_idx = 0
            else:
                accum_idx += 1
            iter_time = time.perf_counter() - compute_t0
            step_time = data_time + iter_time
            local_windows = float(batch["voxel_window"].shape[0]) if "voxel_window" in batch else 0.0
            global_iter_time += iter_time
            global_data_time += data_time
            global_step_time += step_time
            global_windows += local_windows
            interval_iter_time += iter_time
            interval_data_time += data_time
            interval_step_time += step_time
            interval_windows += local_windows
            interval_steps += 1

            for k, v in losses.items():
                if torch.is_tensor(v):
                    running[k] = running.get(k, 0.0) + float(v.detach().item())
                else:
                    running[k] = running.get(k, 0.0) + float(v)
            if vq_commitment_weight is not None:
                running["vq_commitment_weight"] = running.get("vq_commitment_weight", 0.0) + float(vq_commitment_weight)
            if bsq_zeta_weight is not None:
                running["bsq_zeta"] = running.get("bsq_zeta", 0.0) + float(bsq_zeta_weight)

            if n_steps % log_interval == 0:
                msg = ", ".join(f"{k}={running[k]/n_steps:.4f}" for k in sorted(running.keys()) if k != "dep_mask_ratio_realized")
                if self.is_main_process:
                    int_iter_avg = interval_iter_time / max(interval_steps, 1)
                    int_data_avg = interval_data_time / max(interval_steps, 1)
                    int_step_avg = interval_step_time / max(interval_steps, 1)
                    glob_iter_avg = global_iter_time / max(n_steps, 1)
                    glob_data_avg = global_data_time / max(n_steps, 1)
                    glob_step_avg = global_step_time / max(n_steps, 1)
                    data_pct = 100.0 * interval_data_time / max(interval_step_time, 1e-8)
                    windows_per_sec = interval_windows / max(interval_step_time, 1e-8)
                    eta = self._format_eta((total_steps - n_steps) * int_step_avg)
                    print(
                        f"[train][ep={epoch}][it={n_steps}/{total_steps}] "
                        f"eta={eta} "
                        f"time={iter_time:.4f} (int={int_iter_avg:.4f}, avg={glob_iter_avg:.4f}) "
                        f"data={data_time:.4f} (int={int_data_avg:.4f}, avg={glob_data_avg:.4f}) "
                        f"step_int={int_step_avg:.4f} step_avg={glob_step_avg:.4f} "
                        f"data_pct={data_pct:.1f}% windows/s={windows_per_sec:.2f} "
                        f"{msg}"
                    )
                interval_iter_time = 0.0
                interval_data_time = 0.0
                interval_step_time = 0.0
                interval_windows = 0.0
                interval_steps = 0

            if self.is_main_process and self.wandb_run is not None and (global_step % self.wandb_log_every == 0):
                log_data = {}
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        log_data[f"train/{k}"] = float(v.detach().item())
                    else:
                        log_data[f"train/{k}"] = float(v)
                log_data["train/epoch"] = int(epoch)
                log_data["train/iter_time"] = float(iter_time)
                log_data["train/data_time"] = float(data_time)
                log_data["train/step_time"] = float(step_time)
                log_data["train/data_pct"] = float(data_time / max(step_time, 1e-8))
                log_data["train/windows_per_sec"] = float(local_windows / max(step_time, 1e-8))
                if vq_commitment_weight is not None:
                    log_data["train/vq_commitment_weight"] = float(vq_commitment_weight)
                if bsq_zeta_weight is not None:
                    log_data["train/bsq_zeta"] = float(bsq_zeta_weight)
                self.wandb_run.log(log_data, step=global_step)

        stats = {k: v / max(n_steps, 1) for k, v in running.items()}
        if n_steps > 0:
            stats["iter_time"] = global_iter_time / n_steps
            stats["data_time"] = global_data_time / n_steps
            stats["step_time"] = global_step_time / n_steps
            stats["data_pct"] = global_data_time / max(global_step_time, 1e-8)
            stats["windows_per_sec"] = global_windows / max(global_step_time, 1e-8)
        return stats

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running: Dict[str, float] = {}
        n_steps = 0
        for batch in loader:
            n_steps += 1
            batch = self._prepare_batch(batch)
            try:
                autocast_ctx = torch.amp.autocast(self._amp_device_type, enabled=self.use_amp, dtype=self._autocast_dtype)
            except AttributeError:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
            with autocast_ctx:
                out = self.model(batch)
                losses = self._losses_for_batch(batch, out)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())
        stats = {k: v / max(n_steps, 1) for k, v in running.items()}
        if self.is_main_process:
            msg = ", ".join(f"{k}={stats[k]:.4f}" for k in sorted(stats.keys()) if k != "dep_mask_ratio_realized")
            print(f"[val][ep={epoch}] {msg}")
        return stats

    def save_checkpoint(self, out_dir: str, state: TrainState, extra: Dict[str, Any]) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"ckpt_ep{state.epoch:04d}_it{state.step:06d}.pth")
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        raw_sd = model_obj.state_dict()
        clean_sd = {}
        for k, v in raw_sd.items():
            name = k
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
            clean_sd[name] = v
        torch.save(
            {
                "model": clean_sd,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "state": {"epoch": state.epoch, "step": state.step},
                "extra": extra,
            },
            path,
        )
        return path


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

