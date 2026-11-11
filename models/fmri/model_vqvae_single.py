from typing import Dict

import torch
import torch.nn as nn

from .blocks import BottomChunkEncoder, TransformerEncoder
from .hierarchy import masked_chunk_reconstruction_to_voxel
from .quant import VectorQuantizerEMA


def _shared_chunk_indices(chunk_indices: torch.Tensor) -> torch.Tensor:
    return chunk_indices if chunk_indices.ndim == 2 else chunk_indices[0]


class SingleLevelVQVAEModel(nn.Module):
    """
    Baseline: single-level VQ-VAE over bottom chunk tokens only.
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        token_dim: int = 192,
        depth: int = 3,
        num_heads: int = 8,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.t_window = int(t_window)
        self.k_voxels = int(k_voxels)
        self.encoder = BottomChunkEncoder(t_window=t_window, k_voxels=k_voxels, dim=token_dim)
        self.transformer = TransformerEncoder(token_dim, depth=depth, num_heads=num_heads)
        self.quant = VectorQuantizerEMA(codebook_size=codebook_size, dim=token_dim, commitment_weight=commitment_weight)
        self.recon_head = nn.Linear(token_dim, t_window * k_voxels)

    def forward(self, batch: Dict[str, torch.Tensor], encode_only: bool = False) -> Dict[str, torch.Tensor]:
        bottom = batch["bottom_tokens"]  # [B,N0,T,K]
        chunk_indices = _shared_chunk_indices(batch["chunk_indices"])
        n_voxels = int(batch["voxel_window"].shape[-1])
        z = self.encoder(bottom)
        z = self.transformer(z)
        q, qs = self.quant(z)
        base = {
            "z0": z,
            "q0": q,
            "vq_middle": qs["vq_loss"],
            "vq_top": torch.zeros((), device=q.device, dtype=q.dtype),
            "codebook_usage_middle": qs["codebook_usage"],
            "codebook_usage_top": torch.zeros((), device=q.device, dtype=q.dtype),
            "codebook_perplexity_middle": qs["codebook_perplexity"],
            "codebook_perplexity_top": torch.ones((), device=q.device, dtype=q.dtype),
        }
        if encode_only:
            return base
        chunk_recon = self.recon_head(q).view(q.shape[0], q.shape[1], self.t_window, self.k_voxels)
        voxel_recon = masked_chunk_reconstruction_to_voxel(chunk_recon, chunk_indices, n_voxels=n_voxels)
        return {**base, "chunk_recon": chunk_recon, "voxel_recon": voxel_recon}

