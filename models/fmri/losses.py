from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .hierarchy import aggregate_mean_by_ids


def recon_mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_true)


def recon_delta_mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    if x_true.shape[1] < 2:
        return torch.tensor(0.0, device=x_true.device, dtype=x_true.dtype)
    true_delta = x_true[:, 1:, :] - x_true[:, :-1, :]
    pred_delta = x_pred[:, 1:, :] - x_pred[:, :-1, :]
    return F.mse_loss(pred_delta, true_delta)


def alignment_losses(
    pred_parcel_btp: torch.Tensor,
    pred_network_btn: torch.Tensor,
    true_parcel_btp: torch.Tensor,
    true_network_btn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    l_parcel = F.mse_loss(pred_parcel_btp, true_parcel_btp)
    l_network = F.mse_loss(pred_network_btn, true_network_btn)
    return l_parcel, l_network


def consistency_losses(
    pred_parcel_btp: torch.Tensor,
    pred_network_btn: torch.Tensor,
    recon_voxel_btv: torch.Tensor,
    voxel_to_parcel_v: torch.Tensor,
    parcel_to_network_p: torch.Tensor,
    n_parcels: int,
    n_networks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rec_parcel = aggregate_mean_by_ids(recon_voxel_btv, voxel_to_parcel_v, n_parcels)
    rec_network = aggregate_mean_by_ids(rec_parcel, parcel_to_network_p, n_networks)
    return F.mse_loss(pred_parcel_btp, rec_parcel), F.mse_loss(pred_network_btn, rec_network)


def dependency_loss_masked_parcel(
    pred_parcel_btp: torch.Tensor,
    true_parcel_btp: torch.Tensor,
    mask_ratio: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    b, t, p = true_parcel_btp.shape
    device = true_parcel_btp.device
    mask = torch.rand((b, p), device=device) < float(mask_ratio)
    if mask.sum() == 0:
        mask[0, 0] = True
    mask_btp = mask.unsqueeze(1).expand(-1, t, -1)
    loss = F.mse_loss(pred_parcel_btp[mask_btp], true_parcel_btp[mask_btp])
    return loss, {"dep_mask_ratio_realized": mask.float().mean().detach()}


def weighted_total_loss(losses: Dict[str, torch.Tensor], lambdas: Dict[str, float]) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for k, v in losses.items():
        total = total + float(lambdas.get(k, 1.0)) * v
    return total

