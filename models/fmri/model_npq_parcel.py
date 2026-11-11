"""
Single-level BSQ tokenizer on DiFuMo parcel time series only (no voxel encoder).

Input: `middle_tokens` [B, T, P] — parcel BOLD within each window.
Each parcel's T-length trace is embedded to `token_dim`; a Transformer mixes across
parcels; BSQ produces discrete codes per parcel. Decoder reconstructs parcel BOLD.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .blocks import TransformerEncoder
from .bsq import BinarySphericalQuantizer


class NPQParcelTokenizerModel(nn.Module):
    """
    Parcel-only NPQ (BSQ): treats P parcels as sequence positions, T as per-parcel features.
    """

    def __init__(
        self,
        t_window: int,
        n_parcels: int,
        token_dim: int = 192,
        quant_dim: int = 18,
        encoder_depth: int = 3,
        decoder_depth: int = 0,
        num_heads: int = 8,
        bsq_beta: float = 0.1,
        bsq_gamma0: float = 1.0,
        bsq_gamma: float = 1.0,
        bsq_zeta: float = 1.0,
        bsq_group_size: int = 9,
        bsq_l2_norm: bool = False,
        bsq_inv_temperature: float = 1.0,
        bsq_persample_entropy_compute: str = "analytical",
    ) -> None:
        super().__init__()
        self.t_window = int(t_window)
        self.n_parcels = int(n_parcels)

        self.parcel_ts_embed = nn.Linear(self.t_window, token_dim)
        self.enc_transformer = TransformerEncoder(token_dim, depth=encoder_depth, num_heads=num_heads)
        self.quant_proj = nn.Linear(token_dim, quant_dim)
        self.bsq = BinarySphericalQuantizer(
            embed_dim=quant_dim,
            beta=bsq_beta,
            gamma0=bsq_gamma0,
            gamma=bsq_gamma,
            zeta=bsq_zeta,
            group_size=bsq_group_size,
            l2_norm=bsq_l2_norm,
            inv_temperature=bsq_inv_temperature,
            persample_entropy_compute=bsq_persample_entropy_compute,
        )
        self.post_quant_proj = nn.Linear(quant_dim, token_dim)
        self.dec_transformer = (
            TransformerEncoder(token_dim, depth=decoder_depth, num_heads=num_heads) if decoder_depth > 0 else None
        )
        self.recon_head = nn.Linear(token_dim, self.t_window)

    def forward(self, batch: Dict[str, torch.Tensor], encode_only: bool = False) -> Dict[str, torch.Tensor]:
        # [B, T, P] — time-major parcel BOLD
        parcel_btp = batch["middle_tokens"].float()
        b, t_win, n_p = parcel_btp.shape
        if t_win != self.t_window or n_p != self.n_parcels:
            raise ValueError(
                f"middle_tokens shape {(b, t_win, n_p)} does not match model "
                f"(t_window={self.t_window}, n_parcels={self.n_parcels})"
            )
        # [B, P, T] — one token per parcel, features = full timecourse in window
        x = parcel_btp.permute(0, 2, 1).contiguous()
        z = self.parcel_ts_embed(x)
        z = self.enc_transformer(z)
        h = self.quant_proj(z)
        q, bsq_loss, bsq_stats = self.bsq(h)
        base = {
            "z0": z,
            "q0": q,
            "vq_middle": bsq_loss,
            "vq_top": torch.zeros((), device=q.device, dtype=q.dtype),
            "codebook_usage_middle": bsq_stats["codebook_usage"],
            "codebook_usage_top": torch.zeros((), device=q.device, dtype=q.dtype),
            "codebook_perplexity_middle": bsq_stats["codebook_perplexity"],
            "codebook_perplexity_top": torch.ones((), device=q.device, dtype=q.dtype),
            "entropy_loss_middle": bsq_stats.get("entropy_penalty", torch.zeros((), device=q.device)),
            "entropy_loss_top": torch.zeros((), device=q.device, dtype=q.dtype),
            "bsq_commit_loss": bsq_stats["commit_loss"],
            "bsq_cb_entropy": bsq_stats["cb_entropy"],
            "bsq_persample_entropy": bsq_stats["persample_entropy"],
        }
        if encode_only:
            return base
        dec = self.post_quant_proj(q)
        if self.dec_transformer is not None:
            dec = self.dec_transformer(dec)
        # [B, P, T] -> [B, T, P]
        parcel_ts_recon = self.recon_head(dec).permute(0, 2, 1).contiguous()
        return {**base, "parcel_ts_recon": parcel_ts_recon}
