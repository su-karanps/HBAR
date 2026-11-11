from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_expansion: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        mlp_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, num_heads=num_heads, mlp_expansion=mlp_expansion, dropout=dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class CrossAttentionPool(nn.Module):
    """
    Cross-attention pooling from source tokens to target token count via learned queries.
    """

    def __init__(self, dim: int, num_queries: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        source_tokens: torch.Tensor,
        query_override: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        # source_tokens: [B, Ns, D]
        b = source_tokens.shape[0]
        if query_override is not None:
            q = query_override
        else:
            q = self.queries.unsqueeze(0).expand(b, -1, -1)
        h, attn = self.attn(
            self.norm_q(q),
            self.norm_kv(source_tokens),
            self.norm_kv(source_tokens),
            need_weights=return_attn,
            average_attn_weights=False,
        )
        if return_attn:
            # average over heads -> [B, Nt, Ns]
            attn = attn.mean(dim=1)
            return h, attn
        return h


class CrossAttentionRefine(nn.Module):
    """
    Refine query tokens with source context via cross-attention.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, expansion=4.0, dropout=dropout)

    def forward(self, query_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(
            self.norm_q(query_tokens),
            self.norm_kv(source_tokens),
            self.norm_kv(source_tokens),
            need_weights=False,
        )
        x = query_tokens + h
        x = x + self.ffn(self.norm_ffn(x))
        return x


class BottomChunkEncoder(nn.Module):
    """
    Encodes [B, N0, T, K] chunked voxel windows into [B, N0, Tq, D] or [B, N0, D].

    temporal_summary modes
    ----------------------
    "mean" (Tq=1, default):
        Hard-average conv features across all T timesteps → [B, N0, D].
        Backward-compatible with pre-v4 TransformerEncoder models.

    "window_split" (Tq>=1):
        Hard-partition T into Tq non-overlapping windows of width T/Tq, run the
        conv1d encoder within each window independently, then mean-pool within
        the window.  Token i is ONLY allowed to see TRs [i*W, (i+1)*W), so the
        reconstruction gradient for that time-slice flows exclusively to token i.
        This is the v5 design; it fixes the v4 "attn_pool" failure where all Tq
        queries converged to the same global representation because nothing
        structurally tied query i to a particular temporal window.

        Returns 4-D: [B, N0, Tq, D].

    "attn_pool" (Tq>=1):
        Soft attention-pooling with Tq learned query vectors over ALL T timesteps.
        Returns 4-D: [B, N0, Tq, D].
        Kept for reference / ablation.  NOT recommended for new runs because the
        queries have no structural incentive to specialize to distinct windows.
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        dim: int,
        hidden: int = 64,
        n_time_tokens: int = 1,
        temporal_summary: str = "mean",
        temporal_pool_slots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.t_window = int(t_window)
        self.k_voxels = int(k_voxels)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.n_time_tokens = int(n_time_tokens)
        self.temporal_summary = str(temporal_summary)

        if self.temporal_summary == "window_split":
            if self.t_window % self.n_time_tokens != 0:
                raise ValueError(
                    f"t_window={self.t_window} must be divisible by "
                    f"n_time_tokens={self.n_time_tokens} for 'window_split'"
                )
            self.t_per_token = self.t_window // self.n_time_tokens
        elif self.temporal_summary == "mean":
            if self.n_time_tokens != 1:
                raise ValueError("temporal_summary='mean' requires n_time_tokens == 1")
            self.t_per_token = self.t_window
        elif self.temporal_summary == "attn_pool":
            self.t_per_token = self.t_window
            self.pool_queries = nn.Parameter(torch.zeros(self.n_time_tokens, hidden))
            nn.init.trunc_normal_(self.pool_queries, std=0.02)
        else:
            raise ValueError(f"Unknown temporal_summary: {self.temporal_summary!r}")

        # Conv encoder: the kernel always runs along the temporal axis of each window.
        # For 'window_split' it is applied to each sub-window (width = t_per_token)
        # independently; for other modes it sees the full T.
        self.temporal = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Optional position-aware pooling for 'window_split'. The default mean over
        # the W TRs of a token window is shift- and reversal-invariant beyond the conv's 7-TR
        # receptive field, so at Tq 1 (W = 32) the token cannot encode WHEN anything happened.
        # With temporal_pool_slots = Q, each window is pooled by Q learned weight vectors over
        # its W positions, initialised to the means of Q consecutive W/Q-TR segments.
        self.temporal_pool_slots = None if temporal_pool_slots is None else int(temporal_pool_slots)
        n_pool = 1
        if self.temporal_pool_slots is not None:
            if self.temporal_summary != "window_split":
                raise ValueError("temporal_pool_slots requires temporal_summary='window_split'")
            q = self.temporal_pool_slots
            if q < 1 or self.t_per_token % q != 0:
                raise ValueError(f"t_per_token={self.t_per_token} must be divisible by temporal_pool_slots={q}")
            seg = self.t_per_token // q
            init = torch.zeros(q, self.t_per_token)
            for i in range(q):
                init[i, i * seg:(i + 1) * seg] = 1.0 / seg
            self.pool_weights = nn.Parameter(init)  # [Q, W]
            n_pool = q
        # Projection: each time-token sees K voxels × hidden features (× pool slots).
        self.proj = nn.Linear(hidden * k_voxels * n_pool, dim)

    def forward(self, x_bn_tk: torch.Tensor) -> torch.Tensor:
        b, n0, t, k = x_bn_tk.shape

        if self.temporal_summary == "window_split":
            tq = self.n_time_tokens
            w = self.t_per_token  # TRs per token window
            # Split T into Tq windows: [B, N0, T, K] → [B, N0, Tq, W, K]
            x_w = x_bn_tk.reshape(b, n0, tq, w, k)
            # Run conv on each (chunk × voxel × window): fold Tq into the batch.
            # [B, N0, Tq, W, K] → [B*N0*Tq*K, 1, W]
            x_w = x_w.permute(0, 1, 2, 4, 3).contiguous()  # [B, N0, Tq, K, W]
            x_w = x_w.reshape(b * n0 * tq * k, 1, w)
            h = self.temporal(x_w)       # [B*N0*Tq*K, H, W]
            if self.temporal_pool_slots is not None:
                h = torch.einsum("chw,qw->chq", h, self.pool_weights.to(h.dtype))  # [B*N0*Tq*K, H, Q]
                h = h.reshape(b, n0, tq, k * self.hidden * self.temporal_pool_slots)
                return self.proj(h)      # [B, N0, Tq, D]
            h = h.mean(dim=-1)           # [B*N0*Tq*K, H]  -- pool within window
            # Reassemble: [B, N0, Tq, K*H] → proj → [B, N0, Tq, D]
            h = h.reshape(b, n0, tq, k * self.hidden)
            return self.proj(h)          # [B, N0, Tq, D]  -- 4-D

        # --- shared path: run conv over the full T axis ---
        x = x_bn_tk.permute(0, 1, 3, 2).reshape(b * n0 * k, 1, t)
        h = self.temporal(x)  # [B*N0*K, H, T]

        if self.temporal_summary == "attn_pool":
            h_t = h.permute(0, 2, 1)  # [B*N0*K, T, H]
            scale = h_t.shape[-1] ** -0.5
            attn = torch.einsum("qh,bth->bqt", self.pool_queries, h_t) * scale
            attn = attn.softmax(dim=-1)
            pooled = torch.einsum("bqt,bth->bqh", attn, h_t)  # [B*N0*K, Q, H]
            pooled = pooled.reshape(b, n0, k, self.n_time_tokens, self.hidden)
            pooled = pooled.permute(0, 1, 3, 2, 4).contiguous()  # [B, N0, Q, K, H]
            pooled = pooled.reshape(b, n0, self.n_time_tokens, k * self.hidden)
            return self.proj(pooled)  # [B, N0, Q, D]  -- 4-D

        # "mean": backward-compat 3-D output for non-hierarchical models.
        pooled = h.mean(dim=-1)        # [B*N0*K, H]
        pooled = pooled.reshape(b, n0, k * self.hidden)
        return self.proj(pooled)       # [B, N0, D]  -- 3-D


class FactorizedSpaceTimeEncoder(nn.Module):
    """
    Transformer that operates on [B, N, Tq, D] tensors with factorized
    space-then-time self-attention.

    Each block:
        1. spatial self-attention across the N axis, done independently for each
           of the Tq time-slots (Tq is folded into the batch dim);
        2. temporal self-attention across the Tq axis, done independently for
           each of the N spatial tokens (N is folded into the batch dim);
        3. FFN on the time-attention output.

    Spatial attn cost is Tq * O(N^2) per layer (vs. the naive "flatten N*Tq"
    version's O((N*Tq)^2)), which matters at the bottom level where N0 is ~12k.
    Temporal attn cost is N * O(Tq^2) per layer; negligible for Tq <= 8.

    A learned temporal positional embedding is added once at the input so the
    temporal attention has a notion of ordering within a window.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        mlp_expansion: float = 4.0,
        dropout: float = 0.0,
        max_tq: int = 32,
        temporal_batch_chunk: int = 32768,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.max_tq = int(max_tq)
        # SDPA (and most flash-attn kernels) launch with the batch axis as a CUDA
        # grid dimension, which is capped at 65535.  At the bottom level the
        # temporal attention sees B*N0 in the batch slot (e.g. 8 * 12358 ~ 99k),
        # blowing that limit and raising cudaErrorInvalidConfiguration.  We chunk
        # the batch axis here to stay well under the limit.
        self.temporal_batch_chunk = int(temporal_batch_chunk)
        self.temporal_pos = nn.Parameter(torch.zeros(1, 1, self.max_tq, self.dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.spatial_blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, mlp_expansion=mlp_expansion, dropout=dropout)
             for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, mlp_expansion=mlp_expansion, dropout=dropout)
             for _ in range(depth)]
        )

    def _apply_temporal(self, tb: nn.Module, h_bnt_d: torch.Tensor) -> torch.Tensor:
        bn = h_bnt_d.shape[0]
        if bn <= self.temporal_batch_chunk:
            return tb(h_bnt_d)
        pieces = []
        for i in range(0, bn, self.temporal_batch_chunk):
            pieces.append(tb(h_bnt_d[i : i + self.temporal_batch_chunk]))
        return torch.cat(pieces, dim=0)

    def forward(self, x_bntqd: torch.Tensor) -> torch.Tensor:
        b, n, tq, d = x_bntqd.shape
        if tq > self.max_tq:
            raise ValueError(f"FactorizedSpaceTimeEncoder got tq={tq} > max_tq={self.max_tq}")
        x_bntqd = x_bntqd + self.temporal_pos[:, :, :tq, :]
        for sb, tb in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: fold Tq into batch.  [B, N, Tq, D] -> [B*Tq, N, D]
            h = x_bntqd.permute(0, 2, 1, 3).contiguous().reshape(b * tq, n, d)
            h = sb(h)
            x_bntqd = h.reshape(b, tq, n, d).permute(0, 2, 1, 3).contiguous()
            # Temporal: fold N into batch.  [B, N, Tq, D] -> [B*N, Tq, D]
            h = x_bntqd.reshape(b * n, tq, d)
            h = self._apply_temporal(tb, h)
            x_bntqd = h.reshape(b, n, tq, d)
        return x_bntqd

