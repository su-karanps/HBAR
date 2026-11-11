from typing import Dict

import torch
import torch.nn as nn

from .blocks import BottomChunkEncoder, TransformerEncoder
from .bsq import BinarySphericalQuantizer
from .hierarchy import masked_chunk_reconstruction_to_voxel


def _shared_chunk_indices(chunk_indices: torch.Tensor) -> torch.Tensor:
    return chunk_indices if chunk_indices.ndim == 2 else chunk_indices[0]


class NPQChunkTokenizerModel(nn.Module):
    """
    Single-level chunk tokenizer with BSQ (NPQ).
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
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
        self.k_voxels = int(k_voxels)

        self.encoder = BottomChunkEncoder(t_window=t_window, k_voxels=k_voxels, dim=token_dim)
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
        self.recon_head = nn.Linear(token_dim, t_window * k_voxels)

    def forward(self, batch: Dict[str, torch.Tensor], encode_only: bool = False) -> Dict[str, torch.Tensor]:
        bottom = batch["bottom_tokens"]  # [B, N0, T, K]
        chunk_indices = _shared_chunk_indices(batch["chunk_indices"])
        n_voxels = int(batch["voxel_window"].shape[-1])

        z = self.encoder(bottom)
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
        chunk_recon = self.recon_head(dec).view(dec.shape[0], dec.shape[1], self.t_window, self.k_voxels)
        voxel_recon = masked_chunk_reconstruction_to_voxel(chunk_recon, chunk_indices, n_voxels=n_voxels)
        return {**base, "chunk_recon": chunk_recon, "voxel_recon": voxel_recon}
