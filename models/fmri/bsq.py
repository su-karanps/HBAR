from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySphericalQuantizer(nn.Module):
    """
    Collapse-proof non-parametric quantizer.

    Quantization is sign-based:
      zhat = sign(z) in {-1, +1}^D
    with straight-through gradients.
    """

    def __init__(
        self,
        embed_dim: int = 18,
        beta: float = 0.1,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 1.0,
        group_size: int = 9,
        l2_norm: bool = False,
        inv_temperature: float = 1.0,
        persample_entropy_compute: str = "analytical",
        cb_entropy_scope: str = "batch",
    ) -> None:
        super().__init__()
        if cb_entropy_scope not in ("batch", "sample"):
            raise ValueError(f"cb_entropy_scope must be 'batch' or 'sample'; got {cb_entropy_scope!r}")
        # "batch" (default, paper runs): the codebook-entropy term maximises the entropy
        # of the code distribution averaged over EVERY token in the batch, so with
        # mixed-subject batches it can be satisfied by cross-subject variation while
        # each subject's tokens collapse onto few codes.  "sample": the average is taken
        # over the tokens of each sample (window) separately and the entropies are
        # averaged over samples, which rewards within-window code diversity regardless
        # of how batches are composed.
        self.cb_entropy_scope = str(cb_entropy_scope)
        self._n_samples = 1
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.gamma0 = float(gamma0)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.group_size = int(group_size)
        self.l2_norm = bool(l2_norm)
        self.inv_temperature = float(inv_temperature)
        self.persample_entropy_compute = str(persample_entropy_compute)

    def _scoped_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Entropy of the mean code distribution: over the whole batch, or per sample then averaged.

        probs: [N, ...] token-wise probabilities (bits: [N, D]; groups: [N, G, C]).
        Returns the summed entropy over the trailing distribution axes (bits: sum over D of
        binary entropies; groups: sum over G of categorical entropies).
        """
        if self.cb_entropy_scope == "sample" and self._n_samples > 1 and probs.shape[0] % self._n_samples == 0:
            probs = probs.reshape(self._n_samples, -1, *probs.shape[1:])
            mean_probs = probs.mean(dim=1)  # [B, ...]
        else:
            mean_probs = probs.mean(dim=0, keepdim=True)  # [1, ...]
        if mean_probs.ndim == 2:  # bit-wise: binary entropies
            mp = mean_probs.clamp(1e-4, 1.0 - 1e-4)
            om = (1.0 - mp).clamp_min(1e-4)
            ent = (-(mp * mp.log() + om * om.log())).sum(dim=-1)  # [B]
        else:  # group-wise categorical: [B, G, C]
            mp = mean_probs.clamp_min(1e-8)
            ent = (-(mp * mp.log()).sum(dim=-1)).sum(dim=-1)  # [B]
        return ent.mean()

    @staticmethod
    def _sign_no_zeros(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        zhat = self._sign_no_zeros(z)
        return z + (zhat - z).detach()

    def _soft_entropy_loss(self, z_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # "group" (v1 default): both per-sample and codebook entropy computed from the
        #   group softmax over 2^group_size codes per group. Rewards cross-sample
        #   diversity in code space directly.
        # "analytical": bit-wise per-sample entropy + group-based codebook entropy.
        if self.persample_entropy_compute == "group":
            return self._group_entropy_loss(z_flat)

        persample_entropy, _, avg_prob_bits = self._analytical_entropy_loss(z_flat)
        cb_entropy = self._group_codebook_entropy(z_flat)
        return persample_entropy, cb_entropy, avg_prob_bits

    def _analytical_entropy_loss(self, z_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cast to fp32 so the clamp + log are numerically safe under bf16/fp16 autocast.
        z32 = z_flat.float()
        probs = torch.sigmoid(2.0 * z32 / max(self.inv_temperature, 1e-8))
        # 1e-4 is safe in bf16 (1.0 - 1e-4 != 1.0) and small enough to preserve gradients.
        probs = probs.clamp(1e-4, 1.0 - 1e-4)
        one_minus = (1.0 - probs).clamp_min(1e-4)
        bit_entropy = -(probs * probs.log() + one_minus * one_minus.log())  # [N,D]
        persample_entropy = bit_entropy.sum(dim=-1).mean()

        # Encourage diverse average bit usage across the batch (or per sample, see cb_entropy_scope).
        avg_prob = probs.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)  # [D]
        cb_entropy = self._scoped_entropy(probs)
        return persample_entropy.to(z_flat.dtype), cb_entropy.to(z_flat.dtype), avg_prob.to(z_flat.dtype)

    def _group_codebook_entropy(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Group-wise codebook entropy: for each group of bits, treat as a categorical
        over 2^group_size codes. Encourages cross-sample diversity in code space.
        Falls back to analytical bit-wise codebook entropy if grouping is invalid.
        """
        if self.group_size <= 0 or (self.embed_dim % self.group_size) != 0:
            _, cb_entropy, _ = self._analytical_entropy_loss(z_flat)
            return cb_entropy

        n_groups = self.embed_dim // self.group_size
        z_groups = z_flat.reshape(-1, n_groups, self.group_size).float()  # [N,G,S], fp32
        group_codes = self._group_codebook(self.group_size, z_flat.device, torch.float32)  # [C,S]

        logits = torch.einsum("ngs,cs->ngc", z_groups, group_codes) / max(self.inv_temperature, 1e-8)
        probs = F.softmax(logits, dim=-1).clamp_min(1e-8)

        cb_entropy = self._scoped_entropy(probs)
        return cb_entropy.to(z_flat.dtype)

    def _group_entropy_loss(self, z_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full group-wise entropy: both per-sample and codebook entropy from group softmax.
        Used when persample_entropy_compute == "group" (e.g., v2/v4 sweep configs).
        """
        if self.group_size <= 0 or (self.embed_dim % self.group_size) != 0:
            return self._analytical_entropy_loss(z_flat)

        n_groups = self.embed_dim // self.group_size
        z_groups = z_flat.reshape(-1, n_groups, self.group_size).float()  # [N,G,S], fp32
        group_codes = self._group_codebook(self.group_size, z_flat.device, torch.float32)  # [C,S]

        logits = torch.einsum("ngs,cs->ngc", z_groups, group_codes) / max(self.inv_temperature, 1e-8)
        probs = F.softmax(logits, dim=-1).clamp_min(1e-8)

        per_group_entropy = -(probs * probs.log()).sum(dim=-1)  # [N,G]
        persample_entropy = per_group_entropy.sum(dim=-1).mean()

        cb_entropy = self._scoped_entropy(probs)

        avg_prob_bits = torch.sigmoid(2.0 * z_flat.float() / max(self.inv_temperature, 1e-8)).mean(dim=0)
        return persample_entropy.to(z_flat.dtype), cb_entropy.to(z_flat.dtype), avg_prob_bits.to(z_flat.dtype)

    @staticmethod
    def _group_codebook(group_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_codes = 2 ** group_size
        bits = ((torch.arange(n_codes, device=device).unsqueeze(1) >> torch.arange(group_size, device=device)) & 1).float()
        return (bits * 2.0 - 1.0).to(dtype=dtype)

    @torch.no_grad()
    def codes_to_indexes(self, zq_flat: torch.Tensor) -> torch.Tensor:
        # Codes wider than 30 bits cannot be packed into one int64 bincount index (overflow above 62 bits,
        # infeasible bincount size above ~30). For those, index the first 20 bits only; the result then feeds
        # logging stats (usage/perplexity over that 20-bit sub-code), not the quantization. Every model with
        # embed_dim <= 30 is unchanged.
        d = self.embed_dim if self.embed_dim <= 30 else 20
        bits = (zq_flat[..., :d] > 0).to(torch.long)  # [N, d]
        # pack to integer index
        powers = (2 ** torch.arange(d, device=zq_flat.device, dtype=torch.long)).view(1, -1)
        return (bits * powers).sum(dim=-1)  # [N]

    @torch.no_grad()
    def _compute_stats(
        self,
        zq_flat: torch.Tensor,
        device: torch.device,
        orig_shape: torch.Size,
        total_loss: torch.Tensor,
        commit_loss: torch.Tensor,
        entropy_penalty: torch.Tensor,
        cb_entropy: torch.Tensor,
        persample_entropy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        indices = self.codes_to_indexes(zq_flat)
        n_possible = min(2 ** self.embed_dim, 2 ** 20)
        probs = torch.bincount(indices.long(), minlength=n_possible).float()
        probs = probs / probs.sum().clamp_min(1.0)
        usage = (probs > 0).float().sum() / float(max(n_possible, 1))
        perplexity = torch.exp(-(probs * (probs + 1e-12).log()).sum())
        return {
            "vq_loss": total_loss,
            "commit_loss": commit_loss,
            "entropy_penalty": entropy_penalty,
            "cb_entropy": cb_entropy,
            "persample_entropy": persample_entropy,
            "codebook_usage": usage.to(device=device, dtype=perplexity.dtype),
            "codebook_perplexity": perplexity,
            "indices": indices.reshape(orig_shape[:-1]),
        }

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # z: [B, N, D]  (N may already include any extra axes flattened by the caller)
        orig_shape = z.shape
        self._n_samples = int(orig_shape[0]) if z.ndim >= 2 else 1
        z_flat = z.reshape(-1, self.embed_dim)
        # When l2_norm is on we put z onto the unit (hyper)sphere *before* taking the
        # sign.  Without this step z_flat can drift to arbitrary magnitudes, which
        # (a) makes the STE gradient scale-dependent and (b) blows up commit_loss as
        # ||z_flat||^2 rather than pulling z toward the codeword set.
        if self.l2_norm:
            z_flat = F.normalize(z_flat, p=2.0, dim=-1, eps=1e-6)
        zq = self.quantize(z_flat)

        q_scale = 1.0 / (self.embed_dim ** 0.5) if self.l2_norm else 1.0
        zq_scaled = zq * q_scale

        persample_entropy, cb_entropy, _ = self._soft_entropy_loss(z_flat)
        entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        commit_loss = self.beta * torch.mean(((zq_scaled.detach() - z_flat) ** 2).sum(dim=-1))
        total_loss = commit_loss + self.zeta * entropy_penalty / max(self.inv_temperature, 1e-8)

        zq_out = zq_scaled.reshape(orig_shape)
        stats = self._compute_stats(
            zq.detach(),
            z.device,
            orig_shape,
            total_loss.detach(),
            commit_loss.detach(),
            entropy_penalty.detach(),
            cb_entropy.detach(),
            persample_entropy.detach(),
        )
        return zq_out, total_loss, stats
