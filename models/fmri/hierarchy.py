from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import BottomChunkEncoder, CrossAttentionPool, CrossAttentionRefine, TransformerEncoder
from .quant import ResidualVQEMA, VectorQuantizerEMA


def masked_chunk_reconstruction_to_voxel(
    chunk_pred_bntk: torch.Tensor,
    chunk_indices_nk: torch.Tensor,
    n_voxels: int,
) -> torch.Tensor:
    """
    Scatter [B, N0, T, K] chunk predictions back to [B, T, V], averaging overlaps.
    """
    b, n0, t, k = chunk_pred_bntk.shape
    device = chunk_pred_bntk.device
    out = torch.zeros((b, t, n_voxels), device=device, dtype=chunk_pred_bntk.dtype)
    cnt = torch.zeros((b, t, n_voxels), device=device, dtype=chunk_pred_bntk.dtype)
    idx = chunk_indices_nk.to(device)  # [N0, K]
    valid = idx >= 0
    for j in range(k):
        jj = idx[:, j]  # [N0]
        m = valid[:, j]
        if not torch.any(m):
            continue
        jv = jj[m]
        vals = chunk_pred_bntk[:, m, :, j].permute(0, 2, 1)  # [B,T,M]
        out.index_add_(2, jv, vals)
        cnt.index_add_(2, jv, torch.ones_like(vals))
    return out / cnt.clamp_min(1.0)


def aggregate_mean_by_ids(ts_btv: torch.Tensor, ids_v: torch.Tensor, n_ids: int) -> torch.Tensor:
    """
    Aggregate [B,T,V] -> [B,T,n_ids] via average by ids.
    """
    b, t, _ = ts_btv.shape
    device = ts_btv.device
    ids_v = ids_v.to(device)
    out = torch.zeros((b, t, n_ids), device=device, dtype=ts_btv.dtype)
    counts = torch.bincount(ids_v, minlength=n_ids).to(device=device, dtype=ts_btv.dtype).clamp_min(1.0)
    for i in range(b):
        for tt in range(t):
            out[i, tt] = torch.bincount(ids_v, weights=ts_btv[i, tt], minlength=n_ids).to(ts_btv.dtype) / counts
    return out


class ProjectedRVQ(nn.Module):
    """ResidualVQEMA on a low-dim projection, with the VectorQuantizerEMA interface (q, stats).

    token_dim -> quant_dim -> RVQ (EMA codebooks, cosine lookup, dead-code restarts) -> token_dim,
    the same quantizer as the flat RVQ baseline. stats["indices"] is one integer per token
    (mixed radix over stages), and codebook_size is prod(K_i), so bits per token = sum log2(K_i).
    """

    def __init__(self, token_dim: int, quant_dim: int, codebook_sizes: List[int], commitment: float = 0.25,
                 ema_decay: float = 0.99, dead_code_steps: int = 200) -> None:
        super().__init__()
        self.codebook_sizes = [int(k) for k in codebook_sizes]
        self.codebook_size = 1
        for k in self.codebook_sizes:
            self.codebook_size *= k
        self.proj_in = nn.Linear(token_dim, quant_dim)
        self.rvq = ResidualVQEMA(dim=quant_dim, codebook_sizes=self.codebook_sizes, commitment=commitment,
                                 ema_decay=ema_decay, dead_code_steps=dead_code_steps)
        self.proj_out = nn.Linear(quant_dim, token_dim)

    def forward(self, z_bnd: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, _ = z_bnd.shape
        q, vq_loss, st = self.rvq(self.proj_in(z_bnd))
        idx = torch.zeros_like(st["indices"][0])
        for k, i in zip(self.codebook_sizes, st["indices"]):
            idx = idx * k + i
        stats = {
            "vq_loss": vq_loss,
            "codebook_usage": st["codebook_usage"],
            "codebook_perplexity": st["codebook_perplexity"],
            "codebook_perplexity_stage1": st["codebook_perplexity_stage1"],
            "indices": idx.reshape(b, n),
            "stage_indices": [i.reshape(b, n) for i in st["indices"]],
        }
        return self.proj_out(q), stats


class HierarchicalTokenizerCore(nn.Module):
    """
    Shared components used by all three models.
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        token_dim: int = 256,
        bottom_depth: int = 3,
        middle_depth: int = 2,
        top_depth: int = 1,
        num_heads: int = 8,
        middle_codebook_size: int = 1024,
        top_codebook_size: int = 256,
        commitment_weight: float = 0.25,
        quantizer: str = "vq",
        quant_dim: int = 32,
        middle_rvq_sizes: Optional[List[int]] = None,
        top_rvq_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.t_window = int(t_window)
        self.k_voxels = int(k_voxels)
        self.token_dim = int(token_dim)

        self.bottom_encoder = BottomChunkEncoder(t_window=t_window, k_voxels=k_voxels, dim=token_dim)
        self.bottom_transformer = TransformerEncoder(token_dim, depth=bottom_depth, num_heads=num_heads)
        self.middle_transformer = TransformerEncoder(token_dim, depth=middle_depth, num_heads=num_heads)
        self.top_transformer = TransformerEncoder(token_dim, depth=top_depth, num_heads=num_heads)

        if quantizer == "rvq":
            # Swaps only the quantizer: EMA residual VQ with dead-code restarts (flat RVQ baseline).
            if not middle_rvq_sizes or not top_rvq_sizes:
                raise ValueError("quantizer='rvq' requires middle_rvq_sizes and top_rvq_sizes")
            self.quant_middle = ProjectedRVQ(token_dim, quant_dim, middle_rvq_sizes, commitment=commitment_weight)
            self.quant_top = ProjectedRVQ(token_dim, quant_dim, top_rvq_sizes, commitment=commitment_weight)
        elif quantizer == "vq":
            self.quant_middle = VectorQuantizerEMA(
                codebook_size=middle_codebook_size,
                dim=token_dim,
                commitment_weight=commitment_weight,
            )
            self.quant_top = VectorQuantizerEMA(
                codebook_size=top_codebook_size,
                dim=token_dim,
                commitment_weight=commitment_weight,
            )
        else:
            raise ValueError(f"unknown quantizer {quantizer!r}")

        self.decode_middle_from_top = CrossAttentionRefine(token_dim, num_heads=num_heads)
        self.decode_bottom_from_middle = CrossAttentionRefine(token_dim, num_heads=num_heads)
        self.bottom_recon_head = nn.Linear(token_dim, t_window * k_voxels)

    def encode_bottom(self, bottom_tokens_bntk: torch.Tensor) -> torch.Tensor:
        z0 = self.bottom_encoder(bottom_tokens_bntk)
        return self.bottom_transformer(z0)

    def quantize_middle_top(self, h1_bnd: torch.Tensor, h2_bnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        q1, s1 = self.quant_middle(h1_bnd)
        q2, s2 = self.quant_top(h2_bnd)
        stats = {
            "vq_middle": s1["vq_loss"],
            "vq_top": s2["vq_loss"],
            "codebook_usage_middle": s1["codebook_usage"],
            "codebook_usage_top": s2["codebook_usage"],
            "codebook_perplexity_middle": s1["codebook_perplexity"],
            "codebook_perplexity_top": s2["codebook_perplexity"],
        }
        return q1, q2, stats

    def decode_bottom_chunks(
        self,
        z0_bnd: torch.Tensor,
        middle_dec_bnd: torch.Tensor,
        use_bottom_skip: bool = True,
    ) -> torch.Tensor:
        # Optional anti-collapse mode: remove direct bottom-token skip pathway so
        # reconstruction must route through quantized middle/top representations.
        query = z0_bnd if use_bottom_skip else torch.zeros_like(z0_bnd)
        bottom_dec = self.decode_bottom_from_middle(query, middle_dec_bnd)
        chunk = self.bottom_recon_head(bottom_dec)
        return chunk.view(chunk.shape[0], chunk.shape[1], self.t_window, self.k_voxels)


class LearnedHierarchyEncoder(nn.Module):
    def __init__(self, token_dim: int, n_middle: int, n_top: int, num_heads: int = 8) -> None:
        super().__init__()
        self.bottom_to_middle = CrossAttentionPool(token_dim, num_queries=n_middle, num_heads=num_heads)
        self.middle_to_top = CrossAttentionPool(token_dim, num_queries=n_top, num_heads=num_heads)

    def forward(self, z0_bnd: torch.Tensor, return_attn: bool = False):
        if return_attn:
            h1, a01 = self.bottom_to_middle(z0_bnd, return_attn=True)
            h2, a12 = self.middle_to_top(h1, return_attn=True)
            return h1, h2, a01, a12
        h1 = self.bottom_to_middle(z0_bnd)
        h2 = self.middle_to_top(h1)
        return h1, h2

