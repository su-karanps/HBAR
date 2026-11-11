from typing import Dict

import torch
import torch.nn as nn

from .blocks import CrossAttentionPool
from .model_learned import LearnedHierarchyTokenizerModel


def _shared_1d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 1 else x[0]


def _pool_chunk_series_to_parcel(
    chunk_series_bnt: torch.Tensor,
    chunk_parcels_n: torch.Tensor,
    n_parcels: int,
) -> torch.Tensor:
    # chunk_series: [B,N0,T] -> [B,T,P]
    b, _, t = chunk_series_bnt.shape
    device = chunk_series_bnt.device
    ids = chunk_parcels_n.to(device)
    out = torch.zeros((b, t, n_parcels), device=device, dtype=chunk_series_bnt.dtype)
    cnt = torch.zeros((n_parcels,), device=device, dtype=chunk_series_bnt.dtype)
    cnt.index_add_(0, ids, torch.ones_like(ids, dtype=chunk_series_bnt.dtype))
    cnt = cnt.clamp_min(1.0)
    index = ids.view(1, 1, -1).expand(b, t, -1)
    out.scatter_add_(dim=2, index=index, src=chunk_series_bnt.permute(0, 2, 1))
    return out / cnt.view(1, 1, -1)


def _pool_parcel_series_to_network(
    parcel_series_btp: torch.Tensor,
    parcel_to_network_p: torch.Tensor,
    n_networks: int,
) -> torch.Tensor:
    b, t, p = parcel_series_btp.shape
    device = parcel_series_btp.device
    ids = parcel_to_network_p.to(device=device, dtype=torch.long)
    out = torch.zeros((b, t, n_networks), device=device, dtype=parcel_series_btp.dtype)
    index = ids.view(1, 1, p).expand(b, t, p)
    out.scatter_add_(dim=2, index=index, src=parcel_series_btp)
    cnt = torch.bincount(ids, minlength=n_networks).to(device=device, dtype=parcel_series_btp.dtype).clamp_min(1.0)
    return out / cnt.view(1, 1, -1)


class BrainStructuredTokenizerModel(LearnedHierarchyTokenizerModel):
    """
    Model 3:
      - same learned hierarchy as Model 2
      - plus soft atlas-alignment/consistency/dependency prediction heads
    """

    def __init__(
        self,
        t_window: int,
        k_voxels: int,
        n_middle: int = 1024,
        n_top: int = 17,
        n_parcels_target: int = 1024,
        n_networks_target: int = 17,
        token_dim: int = 256,
        bottom_depth: int = 3,
        middle_depth: int = 2,
        top_depth: int = 1,
        num_heads: int = 8,
        middle_codebook_size: int = 1024,
        top_codebook_size: int = 256,
        commitment_weight: float = 0.25,
        decoder_use_bottom_skip: bool = True,
    ) -> None:
        super().__init__(
            t_window=t_window,
            k_voxels=k_voxels,
            n_middle=n_middle,
            n_top=n_top,
            token_dim=token_dim,
            bottom_depth=bottom_depth,
            middle_depth=middle_depth,
            top_depth=top_depth,
            num_heads=num_heads,
            middle_codebook_size=middle_codebook_size,
            top_codebook_size=top_codebook_size,
            commitment_weight=commitment_weight,
            decoder_use_bottom_skip=decoder_use_bottom_skip,
        )
        self.t_window = int(t_window)
        self.n_parcels_target = int(n_parcels_target)
        self.n_networks_target = int(n_networks_target)
        self.parcel_summary_pool = CrossAttentionPool(token_dim, num_queries=n_parcels_target, num_heads=num_heads)
        self.network_summary_pool = CrossAttentionPool(token_dim, num_queries=n_networks_target, num_heads=num_heads)
        self.parcel_time_head = nn.Linear(token_dim, t_window)
        self.network_time_head = nn.Linear(token_dim, t_window)
        self.dep_parcel_head = nn.Linear(token_dim, t_window)

    def forward(self, batch: Dict[str, torch.Tensor], return_attn: bool = False) -> Dict[str, torch.Tensor]:
        out = super().forward(batch, return_attn=return_attn)
        q1 = out["q1"]  # [B,N1,D]
        q2 = out["q2"]  # [B,N2,D]
        chunk_recon = out["chunk_recon"]  # [B,N0,T,K]
        chunk_parcels = _shared_1d(batch["token_parcels"])
        parcel_to_network = _shared_1d(batch["parcel_to_network"])

        parcel_tokens = self.parcel_summary_pool(q1)  # [B,P,D]
        network_tokens = self.network_summary_pool(q2)  # [B,N,D]

        pred_parcel_summary = self.parcel_time_head(parcel_tokens).transpose(1, 2)  # [B,T,P]
        pred_network_summary = self.network_time_head(network_tokens).transpose(1, 2)  # [B,T,N]
        dep_parcel_pred = self.dep_parcel_head(parcel_tokens).transpose(1, 2)  # [B,T,P]

        # reconstruction-derived parcel/network summaries from chunk recon
        chunk_scalar = chunk_recon.mean(dim=-1)  # [B,N0,T]
        recon_parcel_summary = _pool_chunk_series_to_parcel(
            chunk_series_bnt=chunk_scalar,
            chunk_parcels_n=chunk_parcels,
            n_parcels=self.n_parcels_target,
        )

        recon_network_summary = _pool_parcel_series_to_network(
            parcel_series_btp=recon_parcel_summary,
            parcel_to_network_p=parcel_to_network,
            n_networks=self.n_networks_target,
        )

        out.update(
            {
                "pred_parcel_summary": pred_parcel_summary,
                "pred_network_summary": pred_network_summary,
                "dep_parcel_pred": dep_parcel_pred,
                "recon_parcel_summary": recon_parcel_summary,
                "recon_network_summary": recon_network_summary,
            }
        )
        return out

