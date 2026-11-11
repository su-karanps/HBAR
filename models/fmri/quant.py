import math
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    Standard VQ module with straight-through estimator.
    """

    def __init__(self, codebook_size: int, dim: int, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.commitment_weight = float(commitment_weight)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.dim) * 0.02)

    def forward(self, z_bnd: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, d = z_bnd.shape
        flat = z_bnd.reshape(-1, d)  # [B*N, D]
        # squared L2 distance
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = torch.argmin(dist, dim=1)
        quant = self.codebook[idx].reshape(b, n, d)

        codebook_loss = F.mse_loss(quant, z_bnd.detach())
        commit_loss = F.mse_loss(z_bnd, quant.detach())
        vq_loss = codebook_loss + self.commitment_weight * commit_loss

        # straight-through
        quant_st = z_bnd + (quant - z_bnd).detach()

        with torch.no_grad():
            counts = torch.bincount(idx, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            usage = (counts > 0).float().mean()
            perplexity = torch.exp(-(probs * (probs + 1e-12).log()).sum())

        stats = {
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss.detach(),
            "commit_loss": commit_loss.detach(),
            "codebook_usage": usage.detach(),
            "codebook_perplexity": perplexity.detach(),
            "indices": idx.reshape(b, n),
        }
        return quant_st, stats



class _EMACodebook(nn.Module):
    """One stage of ResidualVQEMA: an EMA-updated codebook with cosine lookup.

    Buffers (saved/restored with the model state_dict):
        embed            [K, D]  code vectors = ema_embed_sum / Laplace-smoothed cluster size
        ema_cluster_size [K]
        ema_embed_sum    [K, D]
        unused_steps     [K]     consecutive training forwards in which a code was not selected
    """

    def __init__(self, codebook_size: int, dim: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        embed = torch.randn(self.codebook_size, int(dim))
        self.register_buffer("embed", embed)
        # Unit prior count per code: with zeros, the Laplace-smoothed size of a never-selected
        # code is ~eps, its EMA mean blows up to init/eps, and the magnitude-blind cosine lookup
        # happily selects it (commit loss ~1e9 at step 1). Ones keep embed == ema_embed_sum /
        # ema_cluster_size == init until the code is used; the dead-code restart also sets 1.
        self.register_buffer("ema_cluster_size", torch.ones(self.codebook_size))
        self.register_buffer("ema_embed_sum", embed.clone())
        self.register_buffer("unused_steps", torch.zeros(self.codebook_size, dtype=torch.long))

    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        # Cosine nearest neighbour: both the inputs and the code rows are L2-normalised for the
        # search only; the selected (unnormalised) EMA mean is what gets emitted.
        sim = F.normalize(x, p=2.0, dim=-1, eps=1e-6) @ F.normalize(self.embed, p=2.0, dim=-1, eps=1e-6).t()
        return sim.argmax(dim=-1)


class ResidualVQEMA(nn.Module):
    """Residual vector quantizer with EMA codebooks (flat VQ-VAE baseline quantizer).

    Stage i quantizes the residual `z - sum(q_0..q_{i-1})`; the output is `sum_i q_i` with a
    straight-through estimator. Codebooks are trained by EMA (no codebook loss); the only loss is
    `rvq_commitment * sum_i MSE(resid_i, q_i.detach())`. A code unused for `dead_code_steps`
    consecutive training forwards is re-initialised to a random input vector of the current
    batch. Bits per code = sum_i log2(K_i).

    The attribute is deliberately called `rvq_commitment` (not `commitment_weight` / `beta`):
    `FMRITokenizerTrainer._apply_vq_commitment_warmup` rewrites those attributes on every
    module, and the RVQ weight is meant to stay at its configured value.
    """

    def __init__(
        self,
        dim: int,
        codebook_sizes: List[int],
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_steps: int = 200,
        laplace_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if not codebook_sizes:
            raise ValueError("ResidualVQEMA needs at least one codebook size")
        self.dim = int(dim)
        self.codebook_sizes = [int(k) for k in codebook_sizes]
        self.stages = nn.ModuleList([_EMACodebook(k, self.dim) for k in self.codebook_sizes])
        self.rvq_commitment = float(commitment)
        self.ema_decay = float(ema_decay)
        self.dead_code_steps = int(dead_code_steps)
        self.laplace_eps = float(laplace_eps)

    @property
    def bits_per_code(self) -> float:
        return float(sum(math.log2(k) for k in self.codebook_sizes))

    @staticmethod
    def _dist_active() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @torch.no_grad()
    def _ema_update(self, st: _EMACodebook, x: torch.Tensor, idx: torch.Tensor) -> None:
        k = st.codebook_size
        onehot = F.one_hot(idx, k).to(x.dtype)                 # [M, K]
        counts = onehot.sum(dim=0)                             # [K]
        embed_sum = onehot.t() @ x                             # [K, D]
        if self._dist_active():
            dist.all_reduce(counts)
            dist.all_reduce(embed_sum)
        d = self.ema_decay
        st.ema_cluster_size.mul_(d).add_(counts, alpha=1.0 - d)
        st.ema_embed_sum.mul_(d).add_(embed_sum, alpha=1.0 - d)
        n = st.ema_cluster_size.sum()
        smoothed = (st.ema_cluster_size + self.laplace_eps) / (n + k * self.laplace_eps) * n
        st.embed.copy_(st.ema_embed_sum / smoothed.unsqueeze(1))
        # Dead-code restart (no host sync: masked where-ops).
        st.unused_steps.copy_(torch.where(counts > 0, torch.zeros_like(st.unused_steps), st.unused_steps + 1))
        dead = st.unused_steps >= self.dead_code_steps          # [K]
        ridx = torch.randint(0, x.shape[0], (k,), device=x.device)
        new = x[ridx]                                          # [K, D]
        if self._dist_active():
            dist.broadcast(new, src=0)
        dead_col = dead.unsqueeze(1)
        st.embed.copy_(torch.where(dead_col, new, st.embed))
        st.ema_embed_sum.copy_(torch.where(dead_col, new, st.ema_embed_sum))
        st.ema_cluster_size.copy_(torch.where(dead, torch.ones_like(st.ema_cluster_size), st.ema_cluster_size))
        st.unused_steps.copy_(torch.where(dead, torch.zeros_like(st.unused_steps), st.unused_steps))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # z: [B, N, D] (any leading shape). Codebook math in fp32; output cast back to z.dtype.
        orig_shape = z.shape
        z32 = z.reshape(-1, self.dim).float()
        resid = z32
        q_sum = torch.zeros_like(z32)
        commit = z32.new_zeros(())
        indices: List[torch.Tensor] = []
        for st in self.stages:
            idx = st.lookup(resid.detach())
            q_i = st.embed[idx]                                 # [M, D], no grad (buffer)
            commit = commit + F.mse_loss(resid, q_i.detach())
            if self.training:
                self._ema_update(st, resid.detach(), idx)
            q_sum = q_sum + q_i
            resid = resid - q_i
            indices.append(idx)
        vq_loss = self.rvq_commitment * commit
        q_st = z32 + (q_sum - z32).detach()

        with torch.no_grad():
            def _hist_stats(idx: torch.Tensor, k: int):
                probs = torch.bincount(idx, minlength=k).float()
                probs = probs / probs.sum().clamp_min(1.0)
                ent = -(probs * (probs + 1e-12).log()).sum()
                return ent, (probs > 0).float().mean(), torch.exp(ent)
            ent0, usage0, ppl0 = _hist_stats(indices[0], self.codebook_sizes[0])
            if len(self.stages) > 1:
                _, usage1, ppl1 = _hist_stats(indices[1], self.codebook_sizes[1])
            else:
                usage1, ppl1 = torch.zeros_like(usage0), torch.zeros_like(ppl0)
            zero = torch.zeros((), device=z.device, dtype=z.dtype)
            stats = {
                "vq_loss": vq_loss.detach().to(z.dtype),
                "commit_loss": commit.detach().to(z.dtype),
                "entropy_penalty": zero,
                "cb_entropy": ent0.to(z.dtype),
                "persample_entropy": zero,
                "codebook_usage": usage0.to(z.dtype),
                "codebook_perplexity": ppl0.to(z.dtype),
                "codebook_usage_stage1": usage1.to(z.dtype),
                "codebook_perplexity_stage1": ppl1.to(z.dtype),
                "indices": indices,
            }
        return q_st.reshape(orig_shape).to(z.dtype), vq_loss, stats
