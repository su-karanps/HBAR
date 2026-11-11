from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import BottomChunkEncoder, CrossAttentionPool, FactorizedSpaceTimeEncoder
from .bsq import BinarySphericalQuantizer
from .quant import ResidualVQEMA
from .hierarchy import masked_chunk_reconstruction_to_voxel


def _shared_1d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 1 else x[0]


def _shared_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x[0]


def _scatter_mean_to_parents(child_bnd: torch.Tensor, child_to_parent: torch.Tensor, n_parents: int) -> torch.Tensor:
    """Mean-pool [B, N_child, D] up to [B, N_parents, D] using a fixed atlas mapping."""
    b, _, d = child_bnd.shape
    device = child_bnd.device
    ids = child_to_parent.to(device=device, dtype=torch.long)
    out = torch.zeros((b, n_parents, d), device=device, dtype=child_bnd.dtype)
    counts = torch.bincount(ids, minlength=n_parents).to(device=device, dtype=child_bnd.dtype).clamp_min(1.0)
    idx = ids.view(1, -1, 1).expand(b, -1, d)
    out.scatter_add_(dim=1, index=idx, src=child_bnd)
    return out / counts.view(1, n_parents, 1)


def _scatter_mean_to_parents_time(
    child_bntqd: torch.Tensor, child_to_parent: torch.Tensor, n_parents: int
) -> torch.Tensor:
    """Mean-pool [B, N_child, Tq, D] up to [B, N_parents, Tq, D] by folding Tq into the batch."""
    b, _, tq, d = child_bntqd.shape
    # [B, N_child, Tq, D] -> [B*Tq, N_child, D]
    x = child_bntqd.permute(0, 2, 1, 3).contiguous().reshape(b * tq, -1, d)
    pooled = _scatter_mean_to_parents(x, child_to_parent, n_parents)  # [B*Tq, N_parents, D]
    return pooled.reshape(b, tq, n_parents, d).permute(0, 2, 1, 3).contiguous()


def _cross_attn_pool_time(
    pool_module: CrossAttentionPool, x: torch.Tensor
) -> torch.Tensor:
    """Apply CrossAttentionPool over the spatial dimension while preserving time tokens.

    Input:  [B, N_child, Tq, D]
    Output: [B, N_parent, Tq, D]  (N_parent = pool_module.queries.shape[0])
    """
    b, _, tq, d = x.shape
    x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(b * tq, -1, d)
    out_flat = pool_module(x_flat)  # [B*Tq, N_parent, D]
    n_parent = out_flat.shape[1]
    return out_flat.reshape(b, tq, n_parent, d).permute(0, 2, 1, 3).contiguous()


def _broadcast_parent_series_to_children(
    parent_series_btp: torch.Tensor,
    child_to_parent: torch.Tensor,
) -> torch.Tensor:
    """Given [B, T, N_parents] and a child->parent map, return [B, T, N_children]."""
    ids = child_to_parent.to(device=parent_series_btp.device, dtype=torch.long)
    return parent_series_btp.index_select(dim=2, index=ids)


def parcel_mean_input(
    bottom_bntk: torch.Tensor,
    chunk_parcels: torch.Tensor,
    chunk_indices: torch.Tensor,
    n_parcels: int,
) -> torch.Tensor:
    """Replace every voxel's input signal with its parcel's mean signal (same shape).

    Built from the model's own cross-scale operators so that "parcel mean" means
    exactly what the fixed hierarchy computes: valid-voxel mean within each chunk,
    then `_scatter_mean_to_parents` over chunks (unweighted by chunk size, as in
    the encoder), then `_broadcast_parent_series_to_children` back to chunks.
    Padded voxel slots (chunk_indices < 0) stay zero.

    Used by `input_mode="parcel_mean"` (the sub-parcel-information control) and,
    via `within_parcel_residual`, by the evaluation scripts.
    """
    valid = (chunk_indices >= 0).to(device=bottom_bntk.device, dtype=bottom_bntk.dtype)  # [N0, K]
    valid_b = valid.view(1, valid.shape[0], 1, valid.shape[1])
    cnt = valid.sum(dim=1).clamp_min(1.0)  # [N0]
    chunk_mean_bnt = (bottom_bntk * valid_b).sum(dim=-1) / cnt.view(1, -1, 1)  # [B, N0, T]
    parcel_mean_bpt = _scatter_mean_to_parents(chunk_mean_bnt, chunk_parcels, n_parcels)  # [B, P, T]
    at_chunks_btn = _broadcast_parent_series_to_children(parcel_mean_bpt.permute(0, 2, 1), chunk_parcels)
    return at_chunks_btn.permute(0, 2, 1).unsqueeze(-1) * valid_b  # [B, N0, T, K]


def within_parcel_residual(
    bottom_bntk: torch.Tensor,
    chunk_parcels: torch.Tensor,
    chunk_indices: torch.Tensor,
    n_parcels: int,
) -> torch.Tensor:
    """Voxel signal minus its parcel mean, using the same operators as `parcel_mean_input`."""
    valid = (chunk_indices >= 0).to(device=bottom_bntk.device, dtype=bottom_bntk.dtype)
    return (bottom_bntk - parcel_mean_input(bottom_bntk, chunk_parcels, chunk_indices, n_parcels)) * valid.view(
        1, valid.shape[0], 1, valid.shape[1]
    )


def _zero_like_scalar(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=x.device, dtype=x.dtype)


def _null_bsq_stats(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Stats placeholders for `quantizer="none"` so every logging path still finds a tensor.

    All entries are zeros: nothing is quantized, so codebook usage / perplexity /
    commitment / entropy are undefined.  Zeros (rather than NaN) keep metrics.jsonl,
    the wandb logs and the eval-summary tables finite.
    """
    z = _zero_like_scalar(x)
    return {
        "vq_loss": z,
        "commit_loss": z,
        "entropy_penalty": z,
        "cb_entropy": z,
        "persample_entropy": z,
        "codebook_usage": z,
        "codebook_perplexity": z,
    }


def _size_matched_shuffle(assignment: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute a child->parent assignment vector, preserving group sizes.

    `assignment[i]` is the parent of child `i`.  Permuting the *entries* of that
    vector rearranges which children belong to which parent while leaving the
    multiset of labels -- hence every parent's child count -- exactly unchanged.
    Deterministic given `seed` (CPU generator, so it does not depend on device).
    """
    ids = assignment.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(ids.numel(), generator=g)
    return ids[perm].contiguous()


class _LevelBSQ(nn.Module):
    """
    One level of the hierarchy:
        [B, N_level, Tq, D] -> quant_proj -> BSQ -> post_quant_proj -> recon_head
    producing a full [B, N_level, T, out_per_token] time-series.

    Each of the (N_level * Tq) tokens goes through BSQ independently, so the
    quantizer sees Tq separate codes per spatial location -- one per time-slot
    of length T / Tq.  The per-token recon head reconstructs only that slot,
    and the outputs are concatenated along the time axis.
    """

    def __init__(
        self,
        token_dim: int,
        out_t: int,
        out_per_token: int,
        n_time_tokens: int,
        quant_dim: int,
        beta: float,
        gamma0: float,
        gamma: float,
        zeta: float,
        group_size: int,
        l2_norm: bool,
        inv_temperature: float,
        persample_entropy_compute: str,
        cb_entropy_scope: str = "batch",
        recon_head_layers: int = 1,
        use_voxel_slots: bool = False,
        use_parcel_cond: bool = False,
        quantizer: str = "bsq",
        n_spatial: Optional[int] = None,
        use_spatial_slots: bool = False,
        rvq_codebook_sizes: Optional[list[int]] = None,
        rvq_commitment: float = 0.25,
        rvq_ema_decay: float = 0.99,
        rvq_dead_code_steps: int = 200,
    ) -> None:
        super().__init__()
        if quantizer not in ("bsq", "none", "rvq"):
            raise ValueError(f"quantizer must be 'bsq', 'none' or 'rvq'; got {quantizer!r}")
        # "none" is the continuous-bottleneck control: the latent keeps exactly the
        # same shape (quant_dim per (spatial, time-token)) but is passed through
        # without sign-quantization, so the level costs quant_dim * 16 bits (fp16)
        # per token instead of quant_dim bits.  The BSQ module is still constructed
        # (it holds no parameters) so group_size / embed_dim metadata stays readable.
        self.quantizer = str(quantizer)
        self.out_t = int(out_t)
        self.out_per_token = int(out_per_token)
        self.n_time_tokens = int(n_time_tokens)
        if self.out_t % self.n_time_tokens != 0:
            raise ValueError(
                f"out_t={self.out_t} must be divisible by n_time_tokens={self.n_time_tokens}"
            )
        self.out_t_per_token = self.out_t // self.n_time_tokens
        self.quant_proj = nn.Linear(token_dim, quant_dim)
        self.bsq = BinarySphericalQuantizer(
            embed_dim=quant_dim,
            beta=beta,
            gamma0=gamma0,
            gamma=gamma,
            zeta=zeta,
            group_size=group_size,
            l2_norm=l2_norm,
            inv_temperature=inv_temperature,
            persample_entropy_compute=persample_entropy_compute,
            cb_entropy_scope=cb_entropy_scope,
        )
        # "rvq": residual VQ with EMA codebooks (flat VQ-VAE baseline).  quant_proj is the
        # code projection (quant_dim = VQ code dim); bits per code = sum_i log2(K_i).
        if self.quantizer == "rvq":
            if not rvq_codebook_sizes:
                raise ValueError("quantizer='rvq' requires rvq_codebook_sizes")
            self.rvq = ResidualVQEMA(
                dim=quant_dim,
                codebook_sizes=list(rvq_codebook_sizes),
                commitment=rvq_commitment,
                ema_decay=rvq_ema_decay,
                dead_code_steps=rvq_dead_code_steps,
            )
        self.post_quant_proj = nn.Linear(quant_dim, token_dim)

        # Optional: condition the decoder on the parent-level (parcel) reconstruction.
        # A single scalar per time-token (the parcel's mean signal for that window) is
        # projected to token_dim and added to r before the slot/recon decoder.  This
        # lets the bottom decoder see "what the parcel predicts" while deciding what
        # per-voxel residual to emit.
        self.use_parcel_cond = bool(use_parcel_cond)
        if self.use_parcel_cond:
            self.parcel_cond_proj = nn.Linear(1, token_dim)

        # Optional: a learned embedding per spatial slot (e.g. per parcel) added to the
        # quantized token before decoding, so the same code can decode to a different
        # series at different locations (the spatial analogue of voxel_slots).
        self.use_spatial_slots = bool(use_spatial_slots)
        if self.use_spatial_slots:
            if n_spatial is None:
                raise ValueError("use_spatial_slots requires n_spatial")
            self.spatial_slots = nn.Embedding(int(n_spatial), token_dim)
            nn.init.trunc_normal_(self.spatial_slots.weight, std=0.02)

        self.use_voxel_slots = bool(use_voxel_slots) and (self.out_per_token > 1)
        recon_head_layers = max(1, int(recon_head_layers))

        if self.use_voxel_slots:
            # Each of the K voxel positions gets a learned D-dim slot embedding.
            # In forward, the shared quantized token is biased by slot[k] before
            # decoding, so each voxel can extract a different amplitude/pattern
            # from the same BSQ code.  The slot_decoder predicts only out_t_per_token
            # values (one voxel's time window) rather than all K*out_t_per_token.
            self.voxel_slots = nn.Embedding(self.out_per_token, token_dim)
            nn.init.trunc_normal_(self.voxel_slots.weight, std=0.02)
            def _make_head(in_d: int, out_d: int, n_layers: int) -> nn.Module:
                if n_layers == 1:
                    return nn.Linear(in_d, out_d)
                layers: list[nn.Module] = []
                cur = in_d
                for _ in range(n_layers - 1):
                    layers += [nn.Linear(cur, token_dim), nn.GELU()]
                    cur = token_dim
                layers.append(nn.Linear(cur, out_d))
                return nn.Sequential(*layers)
            self.slot_decoder: nn.Module = _make_head(token_dim, self.out_t_per_token, recon_head_layers)
        else:
            out_dim = self.out_t_per_token * self.out_per_token
            if recon_head_layers == 1:
                self.recon_head: nn.Module = nn.Linear(token_dim, out_dim)
            else:
                layers_list: list[nn.Module] = []
                in_dim = token_dim
                for _ in range(recon_head_layers - 1):
                    layers_list += [nn.Linear(in_dim, token_dim), nn.GELU()]
                    in_dim = token_dim
                layers_list.append(nn.Linear(in_dim, out_dim))
                self.recon_head = nn.Sequential(*layers_list)

    def forward(
        self,
        h_bntqd: torch.Tensor,
        parcel_cond: Optional[torch.Tensor] = None,
        decode: bool = True,
    ):
        """
        Args:
            h_bntqd:     [B, N, Tq, D]  encoder features for this level.
            parcel_cond: [B, N, Tq, 1]  optional parent-level mean signal per
                         time-token, used to condition the reconstruction head.
            decode:      If False, skip the (expensive) reconstruction decoder and
                         return series=None.  Useful for inference/tokenization where
                         only the quantized codes q are needed.
        """
        b, n, tq, d = h_bntqd.shape
        if tq != self.n_time_tokens:
            raise ValueError(f"level got tq={tq}, expected n_time_tokens={self.n_time_tokens}")
        # Treat each (spatial, time-token) pair as an independent token for BSQ.
        h_flat = h_bntqd.reshape(b, n * tq, d)
        z = self.quant_proj(h_flat)
        if self.quantizer == "none":
            q, vq_loss, stats = z, _zero_like_scalar(z), _null_bsq_stats(z)
        elif self.quantizer == "rvq":
            q, vq_loss, stats = self.rvq(z)
        else:
            q, vq_loss, stats = self.bsq(z)

        # Reshape q to [B, N, Tq, quant_dim] before returning/decoding.
        q = q.reshape(b, n, tq, -1)

        if not decode:
            return q, vq_loss, stats, None

        r = self.post_quant_proj(q.reshape(b, n * tq, -1))  # [B, N*Tq, D]

        if self.use_parcel_cond and parcel_cond is not None:
            # parcel_cond: [B, N, Tq, 1] -> project to D, flatten, add to r
            cond_d = self.parcel_cond_proj(parcel_cond)          # [B, N, Tq, D]
            r = r + cond_d.reshape(b, n * tq, d)

        if self.use_spatial_slots:
            if n != self.spatial_slots.num_embeddings:
                raise ValueError(f"level got n={n}, expected {self.spatial_slots.num_embeddings} spatial slots")
            # h_flat index = n_idx * tq + t, so repeat each slot embedding tq times.
            r = r + self.spatial_slots.weight.repeat_interleave(tq, dim=0).unsqueeze(0)

        if self.use_voxel_slots:
            k = self.out_per_token
            # Broadcast token to K voxel slots and bias each by its slot embedding.
            # r: [B, N*Tq, D]  →  [B, N*Tq, K, D]
            r_exp = r.unsqueeze(2).expand(b, n * tq, k, d)
            slots = self.voxel_slots.weight  # [K, D]
            r_slotted = r_exp + slots         # [B, N*Tq, K, D]  (broadcast)
            # Flatten for the shared slot decoder.
            r_slotted = r_slotted.reshape(b * n * tq * k, d)
            per_vox = self.slot_decoder(r_slotted)        # [B*N*Tq*K, out_t_per_token]
            per_vox = per_vox.reshape(b, n, tq, k, self.out_t_per_token)
            # Reorder to [B, N, Tq, out_t_per_token, K] then merge time tokens.
            per_vox = per_vox.permute(0, 1, 2, 4, 3).contiguous()
            series = per_vox.reshape(b, n, tq * self.out_t_per_token, k)
        else:
            flat = self.recon_head(r)    # [B, N*Tq, out_t_per_token * out_per_token]
            flat = flat.view(b, n, tq, self.out_t_per_token, self.out_per_token)
            series = flat.reshape(b, n, tq * self.out_t_per_token, self.out_per_token)

        return q, vq_loss, stats, series


class NPQHierResidualTokenizerModel(nn.Module):
    """
    Residual hierarchical BSQ tokenizer with a fixed atlas hierarchy and
    per-level temporal tokenization.

    Levels (encoder side):
      bottom (chunks)  -> middle (parcels via mean-pool by token_parcels)
                       -> top    (networks via mean-pool by parcel_to_network)

    Each level has its own BSQ codebook *and* its own time-token count Tq_i.
    Internally the level tensor is [B, N_level, Tq_i, D].  BSQ is applied per
    (spatial, time-token) pair, giving the model Tq_i discrete codes per
    spatial location instead of one-per-window.

    Each level has its own reconstruction target:
      top    -> network time series  [B, T, N_net]
      middle -> parcel residual      [B, T, N_parcel]   (parcel_pred = base_from_top + residual)
      bottom -> chunk residual       [B, N0, T, K]      (chunk_pred  = base_from_mid + residual)

    No continuous bottom features (z0/h*) ever feed any decoder; the only path
    from encoder to decoder is through the per-level discrete codes.
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        n_parcels: int = 1024,
        n_networks: int = 17,
        token_dim: int = 192,
        bottom_depth: int = 3,
        middle_depth: int = 2,
        top_depth: int = 1,
        num_heads: int = 8,
        bottom_quant_dim: int = 16,
        middle_quant_dim: int = 10,
        top_quant_dim: int = 6,
        bottom_group_size: int = 8,
        middle_group_size: int = 5,
        top_group_size: int = 3,
        bsq_beta: float = 0.1,
        bsq_gamma0: float = 1.0,
        bsq_gamma: float = 1.0,
        bsq_zeta: float = 1.0,
        bsq_l2_norm: bool = False,
        bsq_inv_temperature: float = 1.0,
        bsq_persample_entropy_compute: str = "analytical",
        bsq_cb_entropy_scope: str = "batch",
        n_time_tokens_bottom: int = 1,
        n_time_tokens_middle: int = 1,
        n_time_tokens_top: int = 1,
        bottom_encoder_hidden: int = 64,
        bottom_temporal_summary: str = "attn_pool",
        bottom_temporal_pool_slots: Optional[int] = None,
        bottom_recon_head_layers: int = 1,
        bottom_use_voxel_slots: bool = False,
        bottom_use_parcel_cond: bool = False,
        middle_recon_head_layers: int = 1,
        middle_use_parcel_slots: bool = False,
        middle_bsq_inv_temperature: Optional[float] = None,
        middle_resid_loss: bool = False,
        middle_detach_from_bottom: bool = False,
        cross_scale_mode: str = "fixed",
        cross_scale_pool_num_heads: Optional[int] = None,
        learned_atlas_aux: bool = False,
        input_mode: str = "voxel",
        n_levels: int = 3,
        shuffle_parcel_to_network_seed: Optional[int] = None,
        shuffle_chunk_to_parcel_seed: Optional[int] = None,
        quantizer: str = "bsq",
        rvq_codebook_sizes: Optional[list[int]] = None,
        rvq_commitment: float = 0.25,
        rvq_ema_decay: float = 0.99,
        rvq_dead_code_steps: int = 200,
    ) -> None:
        super().__init__()
        if input_mode not in ("voxel", "parcel_mean"):
            raise ValueError(f"input_mode must be 'voxel' or 'parcel_mean'; got {input_mode!r}")
        # "parcel_mean": every voxel's input is replaced by its parcel's mean signal
        # (see `parcel_mean_input`). Targets, architecture and bit budget are
        # unchanged, so this isolates the value of within-parcel input variation.
        self.input_mode = str(input_mode)
        # ── hierarchy depth ─────────────────────────────────────────────────
        # 3 (default, paper model): chunks -> parcels -> networks.
        # 2: chunks -> parcels; the middle level decodes the parcel series
        #    directly (parcel_pred = mid_series, no network base).
        # 1: chunks only; the bottom level decodes the chunk series directly
        #    (chunk_pred = bot_series, no parcel base).
        if int(n_levels) not in (1, 2, 3):
            raise ValueError(f"n_levels must be 1, 2 or 3; got {n_levels!r}")
        self.n_levels = int(n_levels)
        if self.n_levels == 1 and bool(bottom_use_parcel_cond):
            raise ValueError(
                "bottom_use_parcel_cond requires a parcel level; set it to false when n_levels=1"
            )
        if shuffle_parcel_to_network_seed is not None and self.n_levels < 3:
            raise ValueError(
                "shuffle_parcel_to_network_seed requires the network level (n_levels=3)"
            )
        if shuffle_chunk_to_parcel_seed is not None and self.n_levels < 2:
            raise ValueError("shuffle_chunk_to_parcel_seed requires the parcel level (n_levels>=2)")
        # ── size-matched shuffled grouping control ──────────────────────────
        # When a seed is given, the corresponding atlas child->parent map is
        # replaced (once, on the first forward) by a random permutation of the
        # assignment vector.  Group sizes are preserved exactly; only *which*
        # children belong to which parent changes.  `chunk_indices`
        # (voxel -> chunk) is never touched.
        self.shuffle_parcel_to_network_seed = (
            int(shuffle_parcel_to_network_seed) if shuffle_parcel_to_network_seed is not None else None
        )
        self.shuffle_chunk_to_parcel_seed = (
            int(shuffle_chunk_to_parcel_seed) if shuffle_chunk_to_parcel_seed is not None else None
        )
        # Buffers are registered as None here and filled lazily on the first
        # forward (the layout tensors only arrive with the batch).  Registering
        # them up front means `_load_from_state_dict` can materialise a saved
        # permutation before any forward has run (see `_load_from_state_dict`).
        if self.shuffle_chunk_to_parcel_seed is not None:
            self.register_buffer("shuffled_chunk_to_parcel", None)
        if self.shuffle_parcel_to_network_seed is not None:
            self.register_buffer("shuffled_parcel_to_network", None)
        if quantizer not in ("bsq", "none", "rvq"):
            raise ValueError(f"quantizer must be 'bsq', 'none' or 'rvq'; got {quantizer!r}")
        self.quantizer = str(quantizer)
        self.t_window = int(t_window)
        self.k_voxels = int(k_voxels)
        self.n_parcels = int(n_parcels)
        self.n_networks = int(n_networks)
        self.n_time_tokens_bottom = int(n_time_tokens_bottom)
        self.n_time_tokens_middle = int(n_time_tokens_middle)
        self.n_time_tokens_top = int(n_time_tokens_top)

        if self.n_time_tokens_middle != self.n_time_tokens_bottom:
            # We scatter-mean the bottom time-token features up to parcels and
            # then up to networks; the simplest path keeps Tq identical across
            # levels.  Different per-level Tq is possible (via temporal pooling)
            # but not implemented here -- keep it symmetric for now.
            raise NotImplementedError(
                "n_time_tokens_middle must equal n_time_tokens_bottom in this version"
            )
        if self.n_time_tokens_top != self.n_time_tokens_bottom:
            raise NotImplementedError(
                "n_time_tokens_top must equal n_time_tokens_bottom in this version"
            )

        self.bottom_encoder = BottomChunkEncoder(
            t_window=t_window,
            k_voxels=k_voxels,
            dim=token_dim,
            hidden=bottom_encoder_hidden,
            n_time_tokens=self.n_time_tokens_bottom,
            temporal_summary=bottom_temporal_summary,
            temporal_pool_slots=bottom_temporal_pool_slots,
        )
        self.bottom_transformer = FactorizedSpaceTimeEncoder(
            dim=token_dim,
            depth=bottom_depth,
            num_heads=num_heads,
            max_tq=max(self.n_time_tokens_bottom, 1),
        )
        if self.n_levels >= 2:
            self.middle_transformer = FactorizedSpaceTimeEncoder(
                dim=token_dim,
                depth=middle_depth,
                num_heads=num_heads,
                max_tq=max(self.n_time_tokens_middle, 1),
            )
        if self.n_levels >= 3:
            self.top_transformer = FactorizedSpaceTimeEncoder(
                dim=token_dim,
                depth=top_depth,
                num_heads=num_heads,
                max_tq=max(self.n_time_tokens_top, 1),
            )

        bsq_kwargs = dict(
            beta=bsq_beta,
            gamma0=bsq_gamma0,
            gamma=bsq_gamma,
            zeta=bsq_zeta,
            l2_norm=bsq_l2_norm,
            inv_temperature=bsq_inv_temperature,
            persample_entropy_compute=bsq_persample_entropy_compute,
            cb_entropy_scope=bsq_cb_entropy_scope,
            # "rvq" applies to the bottom level only; parcel/network levels keep BSQ.
            quantizer="bsq" if self.quantizer == "rvq" else self.quantizer,
        )
        if self.n_levels >= 3:
            self.level_top = _LevelBSQ(
                token_dim=token_dim,
                out_t=t_window,
                out_per_token=1,
                n_time_tokens=self.n_time_tokens_top,
                quant_dim=top_quant_dim,
                group_size=top_group_size,
                **bsq_kwargs,
            )
        if self.n_levels >= 2:
            self.level_middle = _LevelBSQ(
                token_dim=token_dim,
                out_t=t_window,
                out_per_token=1,
                n_time_tokens=self.n_time_tokens_middle,
                quant_dim=middle_quant_dim,
                group_size=middle_group_size,
                recon_head_layers=middle_recon_head_layers,
                n_spatial=n_parcels,
                use_spatial_slots=middle_use_parcel_slots,
                # Middle-only BSQ temperature override (None = shared bsq_inv_temperature).
                # At inv_temperature 1 with l2-normed latents the soft code distribution is
                # near-uniform, so the codebook-entropy term cannot see a collapsed codebook.
                **{
                    **bsq_kwargs,
                    **(
                        {"inv_temperature": float(middle_bsq_inv_temperature)}
                        if middle_bsq_inv_temperature is not None
                        else {}
                    ),
                },
            )
        if cross_scale_mode not in ("fixed", "learned", "soft"):
            raise ValueError(f"cross_scale_mode must be 'fixed', 'learned', or 'soft'; got {cross_scale_mode!r}")
        self.cross_scale_mode = cross_scale_mode
        self.learned_atlas_aux = bool(learned_atlas_aux)
        pool_heads = int(cross_scale_pool_num_heads) if cross_scale_pool_num_heads is not None else int(num_heads)
        if token_dim % pool_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by cross_scale_pool_num_heads ({pool_heads})"
            )
        if cross_scale_mode in ("learned", "soft"):
            if self.n_levels >= 2:
                self.chunk_to_parcel_pool = CrossAttentionPool(
                    dim=token_dim, num_queries=n_parcels, num_heads=pool_heads
                )
            if self.n_levels >= 3:
                self.parcel_to_network_pool = CrossAttentionPool(
                    dim=token_dim, num_queries=n_networks, num_heads=pool_heads
                )

        self.bottom_use_parcel_cond = bool(bottom_use_parcel_cond)
        # middle_resid_loss: also emit the middle level's residual and its own target
        #   (true parcel series - true network series broadcast to parcels), so the trainer
        #   can supervise the middle code directly ("recon_parcel_resid").
        # middle_detach_from_bottom: the bottom level uses the parcel prediction (as chunk base
        #   and as parcel_cond) without back-propagating chunk/voxel losses into the middle and
        #   top levels, which are then trained only by their own losses.
        self.middle_resid_loss = bool(middle_resid_loss)
        self.middle_detach_from_bottom = bool(middle_detach_from_bottom)
        self.level_bottom = _LevelBSQ(
            token_dim=token_dim,
            out_t=t_window,
            out_per_token=k_voxels,
            n_time_tokens=self.n_time_tokens_bottom,
            quant_dim=bottom_quant_dim,
            group_size=bottom_group_size,
            recon_head_layers=bottom_recon_head_layers,
            use_voxel_slots=bottom_use_voxel_slots,
            use_parcel_cond=bottom_use_parcel_cond,
            **{**bsq_kwargs, "quantizer": self.quantizer},
            rvq_codebook_sizes=rvq_codebook_sizes,
            rvq_commitment=rvq_commitment,
            rvq_ema_decay=rvq_ema_decay,
            rvq_dead_code_steps=rvq_dead_code_steps,
        )

    # ── shuffled-grouping control ───────────────────────────────────────────

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[override]
        """Materialise lazily-registered shuffle buffers before the normal copy.

        The permutation is built on the first forward, so a checkpoint loaded
        into a fresh model (evaluation scripts load before ever calling forward)
        would otherwise see `shuffled_*` as an unexpected key and silently drop
        it -- the model would then re-derive a permutation from its seed.  That
        happens to agree here, but relying on it is fragile, so the saved
        permutation is restored explicitly.
        """
        for name in ("shuffled_chunk_to_parcel", "shuffled_parcel_to_network"):
            key = prefix + name
            if key in state_dict and name in self._buffers and self._buffers[name] is None:
                self._buffers[name] = state_dict[key].detach().clone().long()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _resolve_maps(
        self, chunk_parcels: torch.Tensor, parcel_to_network: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the child->parent maps the hierarchy should actually use.

        Identity unless a shuffle seed was configured, in which case the
        permutation is built once (first forward) and cached as a buffer so it
        is saved in the checkpoint and reused at evaluation time.
        """
        if self.shuffle_chunk_to_parcel_seed is not None:
            if self._buffers["shuffled_chunk_to_parcel"] is None:
                self._buffers["shuffled_chunk_to_parcel"] = _size_matched_shuffle(
                    chunk_parcels, self.shuffle_chunk_to_parcel_seed
                ).to(device=chunk_parcels.device)
            chunk_parcels = self.shuffled_chunk_to_parcel.to(device=chunk_parcels.device)
        if self.shuffle_parcel_to_network_seed is not None:
            if self._buffers["shuffled_parcel_to_network"] is None:
                self._buffers["shuffled_parcel_to_network"] = _size_matched_shuffle(
                    parcel_to_network, self.shuffle_parcel_to_network_seed
                ).to(device=parcel_to_network.device)
            parcel_to_network = self.shuffled_parcel_to_network.to(device=parcel_to_network.device)
        return chunk_parcels, parcel_to_network

    @property
    def maps_are_shuffled(self) -> bool:
        return (
            self.shuffle_chunk_to_parcel_seed is not None
            or self.shuffle_parcel_to_network_seed is not None
        )

    @torch.no_grad()
    def _shuffled_coarse_targets(
        self,
        batch: Dict[str, torch.Tensor],
        chunk_parcels: torch.Tensor,
        parcel_to_network: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruction targets for the coarse levels under a shuffled hierarchy.

        With a shuffled map the dataset's `middle_tokens` / `top_tokens` (atlas
        parcel / network means) are no longer the aggregate of the model's
        groups, so slot p of the model is not atlas parcel p.  The fair target is
        the aggregate of the *fine* signals under the model's own mapping:

          parcel_target  = scatter_mean over chunks (of the valid-voxel chunk
                           mean of `bottom_tokens`) under the chunk->parcel map
                           in use.  Only produced when chunk->parcel is shuffled.
          network_target = scatter_mean of the parcel series under the
                           parcel->network map in use; the parcel series is
                           `parcel_target` when chunk->parcel is shuffled, and
                           the dataset's `middle_tokens` otherwise.

        With the *true* maps, `scatter_mean(middle_tokens, parcel_to_network)`
        reproduces `top_tokens` exactly (the preprocessor builds the network
        series the same way, as an unweighted mean of parcel series).  The
        analogous identity does NOT hold for the parcel level: the preprocessor
        averages *voxels* within a parcel, whereas the model's operators average
        *chunk means* (unweighted by chunk size), so `parcel_target` differs
        from `middle_tokens` whenever a parcel's chunks have unequal voxel counts.
        """
        out: Dict[str, torch.Tensor] = {}
        parcel_series_btp = batch["middle_tokens"].float()
        if self.shuffle_chunk_to_parcel_seed is not None:
            bottom = batch["bottom_tokens"]
            valid = (chunk_indices >= 0).to(device=bottom.device, dtype=bottom.dtype)  # [N0, K]
            cnt = valid.sum(dim=1).clamp_min(1.0)  # [N0]
            chunk_mean_bnt = (bottom * valid.view(1, valid.shape[0], 1, valid.shape[1])).sum(dim=-1)
            chunk_mean_bnt = chunk_mean_bnt / cnt.view(1, -1, 1)  # [B, N0, T]
            parcel_series_btp = _scatter_mean_to_parents(
                chunk_mean_bnt, chunk_parcels, self.n_parcels
            ).permute(0, 2, 1).contiguous().float()  # [B, T, P]
            out["parcel_target"] = parcel_series_btp
        if self.n_levels >= 3:
            out["network_target"] = (
                _scatter_mean_to_parents(
                    parcel_series_btp.permute(0, 2, 1).contiguous(), parcel_to_network, self.n_networks
                )
                .permute(0, 2, 1)
                .contiguous()
            )  # [B, T, N_net]
        return out

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        encode_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:       Standard data batch dict.
            encode_only: If True, skip all decoder computation and return only the
                         codes of the levels this model has: {q0, q1, q2} at
                         n_levels=3, {q0, q1} at 2, {q0} at 1.  Speeds up
                         inference/tokenization by skipping the expensive
                         slot-embedding decoder.
        """
        bottom = batch["bottom_tokens"]  # [B, N0, T, K]
        chunk_parcels = _shared_1d(batch["token_parcels"])  # [N0]
        parcel_to_network = _shared_1d(batch["parcel_to_network"])  # [P]
        chunk_indices = _shared_2d(batch["chunk_indices"])  # [N0, K]
        n_voxels = int(batch["voxel_window"].shape[-1])
        if self.input_mode == "parcel_mean":
            bottom = parcel_mean_input(bottom, chunk_parcels, chunk_indices, self.n_parcels)
        # Everything below (encoder pooling, atlas references, decoder broadcast)
        # uses the resolved maps; `input_mode` above deliberately keeps the true
        # atlas, because it defines the *input signal*, not the hierarchy.
        chunk_parcels, parcel_to_network = self._resolve_maps(chunk_parcels, parcel_to_network)

        z_chunk = self.bottom_encoder(bottom)  # [B, N0, Tq_b, D]
        z_chunk = self.bottom_transformer(z_chunk)

        align_scale_loss: Optional[torch.Tensor] = None
        atlas_latent_mse: Optional[torch.Tensor] = None
        z_parcel_atlas: Optional[torch.Tensor] = None
        z_net_atlas: Optional[torch.Tensor] = None
        z_parcel: Optional[torch.Tensor] = None
        z_net: Optional[torch.Tensor] = None
        if self.n_levels >= 2 and self.cross_scale_mode == "fixed":
            z_parcel = _scatter_mean_to_parents_time(z_chunk, chunk_parcels, self.n_parcels)
            z_parcel = self.middle_transformer(z_parcel)
            if self.n_levels >= 3:
                z_net = _scatter_mean_to_parents_time(z_parcel, parcel_to_network, self.n_networks)
        elif self.n_levels >= 2:
            # Learned (or soft) cross-scale aggregation via cross-attention.
            z_parcel = _cross_attn_pool_time(self.chunk_to_parcel_pool, z_chunk)
            z_parcel = self.middle_transformer(z_parcel)
            if self.n_levels >= 3:
                z_net = _cross_attn_pool_time(self.parcel_to_network_pool, z_parcel)
            # Atlas reference features (for soft / learned-aux loss and for Pareto logging).
            if self.cross_scale_mode in ("learned", "soft"):
                with torch.no_grad():
                    z_parcel_atlas = _scatter_mean_to_parents_time(z_chunk, chunk_parcels, self.n_parcels)
                    if self.n_levels >= 3:
                        z_net_atlas = _scatter_mean_to_parents_time(z_parcel_atlas, parcel_to_network, self.n_networks)
            use_align = self.cross_scale_mode == "soft" or (
                self.cross_scale_mode == "learned" and self.learned_atlas_aux
            )
            if use_align and z_parcel_atlas is not None:
                if z_net_atlas is not None:
                    align_scale_loss = F.mse_loss(z_parcel, z_parcel_atlas) + F.mse_loss(z_net, z_net_atlas)
                else:
                    # n_levels=2: no network level, so the align loss is on z_parcel only.
                    align_scale_loss = F.mse_loss(z_parcel, z_parcel_atlas)

        if self.cross_scale_mode in ("learned", "soft") and z_parcel_atlas is not None:
            # Pareto / logging: deviation from atlas mean-pool (same construction as align_scale loss).
            if z_net_atlas is not None:
                atlas_latent_mse = (
                    F.mse_loss(z_parcel.detach(), z_parcel_atlas) + F.mse_loss(z_net.detach(), z_net_atlas)
                ).detach()
            else:
                atlas_latent_mse = F.mse_loss(z_parcel.detach(), z_parcel_atlas).detach()

        q_top = vq_top = st_top = top_series = None
        q_mid = vq_mid = st_mid = mid_series = None
        if self.n_levels >= 3:
            z_net = self.top_transformer(z_net)
            q_top, vq_top, st_top, top_series = self.level_top(z_net, decode=not encode_only)
        if self.n_levels >= 2:
            q_mid, vq_mid, st_mid, mid_series = self.level_middle(z_parcel, decode=not encode_only)
        q_bot, vq_bot, st_bot, bot_series = self.level_bottom(
            z_chunk, parcel_cond=None, decode=not encode_only
        )

        if encode_only:
            enc: Dict[str, torch.Tensor] = {"q0": q_bot}
            if self.n_levels >= 2:
                enc["q1"] = q_mid
            if self.n_levels >= 3:
                enc["q2"] = q_top
            return enc

        # ── Full forward: build reconstructions ──────────────────────────────
        network_pred_btn: Optional[torch.Tensor] = None
        parcel_pred_btp: Optional[torch.Tensor] = None
        if self.n_levels >= 3:
            network_pred_btn = top_series.squeeze(-1).permute(0, 2, 1)        # [B, T, N_net]
        if self.n_levels >= 2:
            parcel_residual_btp = mid_series.squeeze(-1).permute(0, 2, 1)     # [B, T, P]
            if network_pred_btn is not None:
                parcel_base_btp = _broadcast_parent_series_to_children(network_pred_btn, parcel_to_network)
                parcel_pred_btp = parcel_base_btp + parcel_residual_btp        # [B, T, P]
            else:
                # n_levels=2: no top base, the middle level decodes the parcel series directly.
                parcel_pred_btp = parcel_residual_btp

        # Optionally re-run bottom decoder with parcel conditioning.
        # (Only reached when encode_only=False and bottom_use_parcel_cond=True.)
        if self.bottom_use_parcel_cond:
            b_sz = parcel_pred_btp.shape[0]
            tq = self.n_time_tokens_bottom
            t_per = self.t_window // tq
            parcel_at_chunks = _broadcast_parent_series_to_children(
                parcel_pred_btp.detach() if self.middle_detach_from_bottom else parcel_pred_btp, chunk_parcels
            )  # [B, T, N0]
            parcel_cond = (
                parcel_at_chunks.permute(0, 2, 1)  # [B, N0, T]
                .reshape(b_sz, -1, tq, t_per)
                .mean(dim=-1, keepdim=True)
            )  # [B, N0, Tq, 1]
            # Re-run bottom level with parcel conditioning (decode=True).
            q_bot, vq_bot, st_bot, bot_series = self.level_bottom(
                z_chunk, parcel_cond=parcel_cond, decode=True
            )

        chunk_residual_bntk = bot_series  # [B, N0, T, K]
        if parcel_pred_btp is not None:
            chunk_base_btn = _broadcast_parent_series_to_children(
                parcel_pred_btp.detach() if self.middle_detach_from_bottom else parcel_pred_btp, chunk_parcels
            )  # [B, T, N0]
            chunk_base_bntk = chunk_base_btn.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.k_voxels)
            chunk_pred_bntk = chunk_base_bntk + chunk_residual_bntk
        else:
            # n_levels=1: no parcel base, the bottom level decodes the chunk series directly.
            chunk_pred_bntk = chunk_residual_bntk

        voxel_recon = masked_chunk_reconstruction_to_voxel(chunk_pred_bntk, chunk_indices, n_voxels=n_voxels)

        stat_levels = [st_bot]
        if self.n_levels >= 2:
            stat_levels.append(st_mid)
        if self.n_levels >= 3:
            stat_levels.append(st_top)

        def _stat_sum(key: str) -> torch.Tensor:
            acc = stat_levels[0][key]
            for st in stat_levels[1:]:
                acc = acc + st[key]
            return acc

        out = {
            "z_chunk": z_chunk,
            "q0": q_bot,
            "chunk_recon": chunk_pred_bntk,
            "voxel_recon": voxel_recon,
            "vq_bottom": vq_bot,
            "codebook_usage_bottom": st_bot["codebook_usage"],
            "codebook_perplexity_bottom": st_bot["codebook_perplexity"],
            "bsq_commit_loss_bottom": st_bot["commit_loss"],
            "bsq_cb_entropy_bottom": st_bot["cb_entropy"],
            "bsq_persample_entropy_bottom": st_bot["persample_entropy"],
            "bsq_commit_loss": _stat_sum("commit_loss"),
            "bsq_cb_entropy": _stat_sum("cb_entropy"),
            "bsq_persample_entropy": _stat_sum("persample_entropy"),
        }
        if "codebook_usage_stage1" in st_bot:  # rvq bottom level: second-stage codebook stats
            out["codebook_usage_bottom_stage1"] = st_bot["codebook_usage_stage1"]
            out["codebook_perplexity_bottom_stage1"] = st_bot["codebook_perplexity_stage1"]
        if self.n_levels >= 2:
            out.update(
                {
                    "z_parcel": z_parcel,
                    "q1": q_mid,
                    "parcel_pred": parcel_pred_btp,
                    "vq_middle": vq_mid,
                    "codebook_usage_middle": st_mid["codebook_usage"],
                    "codebook_perplexity_middle": st_mid["codebook_perplexity"],
                    "bsq_commit_loss_middle": st_mid["commit_loss"],
                    "bsq_cb_entropy_middle": st_mid["cb_entropy"],
                    "bsq_persample_entropy_middle": st_mid["persample_entropy"],
                }
            )
        if self.n_levels >= 3:
            out.update(
                {
                    "z_net": z_net,
                    "q2": q_top,
                    "network_pred": network_pred_btn,
                    "vq_top": vq_top,
                    "codebook_usage_top": st_top["codebook_usage"],
                    "codebook_perplexity_top": st_top["codebook_perplexity"],
                    "bsq_commit_loss_top": st_top["commit_loss"],
                    "bsq_cb_entropy_top": st_top["cb_entropy"],
                    "bsq_persample_entropy_top": st_top["persample_entropy"],
                }
            )
        if self.maps_are_shuffled:
            out.update(
                self._shuffled_coarse_targets(batch, chunk_parcels, parcel_to_network, chunk_indices)
            )
        if self.middle_resid_loss and self.n_levels >= 3:
            pt = out.get("parcel_target", batch["middle_tokens"]).float()
            nt = out.get("network_target", batch["top_tokens"]).float()
            out["parcel_resid_pred"] = parcel_residual_btp
            out["parcel_resid_target"] = pt - _broadcast_parent_series_to_children(nt, parcel_to_network).float()
        if align_scale_loss is not None:
            out["align_scale"] = align_scale_loss
        if atlas_latent_mse is not None:
            out["atlas_latent_mse"] = atlas_latent_mse
        return out
