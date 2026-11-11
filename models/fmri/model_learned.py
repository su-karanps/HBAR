from typing import Dict

import torch
import torch.nn as nn

from .hierarchy import HierarchicalTokenizerCore, LearnedHierarchyEncoder, masked_chunk_reconstruction_to_voxel


def _shared_chunk_indices(chunk_indices: torch.Tensor) -> torch.Tensor:
    return chunk_indices if chunk_indices.ndim == 2 else chunk_indices[0]


class LearnedHierarchyTokenizerModel(nn.Module):
    """
    Model 2:
      - Bottom tokens: sub-parcel chunks
      - Middle/top tokens: learned query-token hierarchy
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        n_middle: int = 1024,
        n_top: int = 17,
        token_dim: int = 256,
        bottom_depth: int = 3,
        middle_depth: int = 2,
        top_depth: int = 1,
        num_heads: int = 8,
        middle_codebook_size: int = 1024,
        top_codebook_size: int = 256,
        commitment_weight: float = 0.25,
        decoder_use_bottom_skip: bool = True,
        n_chunks: int = 0,
        middle_identity: bool = False,
        **quant_kwargs,
    ) -> None:
        super().__init__()
        self.decoder_use_bottom_skip = bool(decoder_use_bottom_skip)
        # Optional learned per-chunk embeddings (n_chunks > 0). Without them, decoder_use_bottom_skip=False
        # feeds identical (zero) queries to every chunk, so every chunk decodes to the same output.
        # chunk_pos is added to the encoder's chunk tokens (lets the learned queries localize); dec_query
        # replaces the zero decoder query. Neither carries data, so recon still passes only through the codes.
        self.chunk_pos = nn.Parameter(torch.randn(n_chunks, token_dim) * 0.02) if n_chunks > 0 else None
        self.dec_query = nn.Parameter(torch.randn(n_chunks, token_dim) * 0.02) if n_chunks > 0 else None
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
            **quant_kwargs,
        )
        self.learned_encoder = LearnedHierarchyEncoder(token_dim=token_dim, n_middle=n_middle, n_top=n_top, num_heads=num_heads)
        # middle_identity: the middle level is the chunk tokens themselves (one code per chunk), and only
        # the chunk -> top grouping is learned. With n_middle == n_chunks learned queries, attention
        # starts near-uniform and every middle token collapses to the same vector (and the same code).
        self.middle_identity = bool(middle_identity)
        if self.middle_identity:
            if n_chunks <= 0 or int(n_middle) != int(n_chunks):
                raise ValueError("middle_identity requires chunk_pos_embed and n_middle == n_chunks")
            self.learned_encoder.bottom_to_middle = nn.Identity()
            self.core.decode_bottom_from_middle = nn.Identity()
            self.dec_query = None

    def forward(self, batch: Dict[str, torch.Tensor], encode_only: bool = False, return_attn: bool = False) -> Dict[str, torch.Tensor]:
        bottom = batch["bottom_tokens"]  # [B,N0,T,K]
        chunk_indices = _shared_chunk_indices(batch["chunk_indices"])
        n_voxels = int(batch["voxel_window"].shape[-1])

        z0 = self.core.encode_bottom(bottom)
        if self.chunk_pos is not None:
            z0 = z0 + self.chunk_pos.unsqueeze(0).to(z0.dtype)
        if self.middle_identity:
            a01 = None
            h1 = z0
            if return_attn:
                h2, a12 = self.learned_encoder.middle_to_top(h1, return_attn=True)
            else:
                h2 = self.learned_encoder.middle_to_top(h1)
        elif return_attn:
            h1, h2, a01, a12 = self.learned_encoder(z0, return_attn=True)
        else:
            h1, h2 = self.learned_encoder(z0, return_attn=False)
        h1 = self.core.middle_transformer(h1)
        h2 = self.core.top_transformer(h2)

        q1, q2, q_stats = self.core.quantize_middle_top(h1, h2)
        out: Dict = {"z0": z0, "h1": h1, "h2": h2, "q1": q1, "q2": q2}
        out.update(q_stats)
        if encode_only:
            return out
        mid_dec = self.core.decode_middle_from_top(q1, q2)
        if self.middle_identity:
            # mid_dec is per chunk (middle == chunks), already refined by the top codes.
            chunk_recon = self.core.bottom_recon_head(mid_dec).view(
                z0.shape[0], z0.shape[1], self.core.t_window, self.core.k_voxels
            )
        elif self.dec_query is not None and not self.decoder_use_bottom_skip:
            query = self.dec_query.unsqueeze(0).expand(z0.shape[0], -1, -1).to(z0.dtype)
            bottom_dec = self.core.decode_bottom_from_middle(query, mid_dec)
            chunk_recon = self.core.bottom_recon_head(bottom_dec).view(
                z0.shape[0], z0.shape[1], self.core.t_window, self.core.k_voxels
            )
        else:
            chunk_recon = self.core.decode_bottom_chunks(
                z0,
                mid_dec,
                use_bottom_skip=self.decoder_use_bottom_skip,
            )
        out["chunk_recon"] = chunk_recon
        out["voxel_recon"] = masked_chunk_reconstruction_to_voxel(chunk_recon, chunk_indices, n_voxels=n_voxels)
        if return_attn:
            out["attn_bottom_to_middle"] = a01
            out["attn_middle_to_top"] = a12
        return out

