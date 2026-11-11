from typing import Dict

import torch
import torch.nn as nn

from .hierarchy import HierarchicalTokenizerCore, masked_chunk_reconstruction_to_voxel


def _shared_1d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 1 else x[0]


def _shared_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x[0]


def _pool_by_ids(tokens_bnd: torch.Tensor, ids_n: torch.Tensor, n_ids: int) -> torch.Tensor:
    """
    Mean pool [B,N,D] to [B,n_ids,D] by integer ids_n.
    """
    b, _, d = tokens_bnd.shape
    device = tokens_bnd.device
    ids_n = ids_n.to(device)
    out = torch.zeros((b, n_ids, d), device=device, dtype=tokens_bnd.dtype)
    counts = torch.bincount(ids_n, minlength=n_ids).to(device=device, dtype=tokens_bnd.dtype).clamp_min(1.0)
    for i in range(b):
        out_i = torch.zeros((n_ids, d), device=device, dtype=tokens_bnd.dtype)
        out_i.index_add_(0, ids_n, tokens_bnd[i])
        out[i] = out_i / counts.unsqueeze(-1)
    return out


class FixedPriorTokenizerModel(nn.Module):
    """
    Model 1:
      - Bottom tokens: sub-parcel voxel chunks
      - Middle tokens: fixed 1024 parcels via pooling
      - Top tokens: fixed 17 networks via pooling
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        n_parcels: int = 1024,
        n_networks: int = 17,
        token_dim: int = 256,
        bottom_depth: int = 3,
        middle_depth: int = 2,
        top_depth: int = 1,
        num_heads: int = 8,
        middle_codebook_size: int = 1024,
        top_codebook_size: int = 256,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_parcels = int(n_parcels)
        self.n_networks = int(n_networks)
        self.core = HierarchicalTokenizerCore(
            t_window=t_window,
            k_voxels=k_voxels,
            token_dim=token_dim,
            bottom_depth=bottom_depth,
            middle_depth=middle_depth,
            top_depth=top_depth,
            num_heads=num_heads,
            middle_codebook_size=middle_codebook_size,
            top_codebook_size=top_codebook_size,
            commitment_weight=commitment_weight,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bottom = batch["bottom_tokens"]  # [B,N0,T,K]
        chunk_parcels = _shared_1d(batch["token_parcels"])  # [N0]
        parcel_to_network_ids = _shared_1d(batch["parcel_to_network"])  # [P]
        chunk_indices = batch["chunk_indices"] if "chunk_indices" in batch else None
        n_voxels = int(batch["voxel_window"].shape[-1])

        z0 = self.core.encode_bottom(bottom)  # [B,N0,D]

        h1 = _pool_by_ids(z0, chunk_parcels, self.n_parcels)  # [B,P,D]
        h1 = self.core.middle_transformer(h1)
        h2 = _pool_by_ids(h1, parcel_to_network_ids, self.n_networks)  # [B,N,D]
        h2 = self.core.top_transformer(h2)

        q1, q2, q_stats = self.core.quantize_middle_top(h1, h2)
        mid_dec = self.core.decode_middle_from_top(q1, q2)
        chunk_recon = self.core.decode_bottom_chunks(z0, mid_dec)  # [B,N0,T,K]

        out = {
            "z0": z0,
            "h1": h1,
            "h2": h2,
            "q1": q1,
            "q2": q2,
            "chunk_recon": chunk_recon,
        }
        out.update(q_stats)
        if chunk_indices is not None:
            out["voxel_recon"] = masked_chunk_reconstruction_to_voxel(chunk_recon, _shared_2d(chunk_indices), n_voxels=n_voxels)
        return out

