"""
Map tokenizer activations to flat discrete token id sequences (for storage + AR).

- VQ-VAE: EMA VQ `indices` (codebook id per slot).
- BSQ: per group, pack `sign(q)` to integers in [0, 2^group_size), shared across groups;
  add a **level offset** so the same int at different levels does not share an embedding
  (unless you want to — we offset by 2^max_g * level for simplicity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.fmri import LearnedHierarchyTokenizerModel, NPQChunkTokenizerModel, NPQParcelTokenizerModel, SingleLevelVQVAEModel

try:
    from models.fmri import NPQHierResidualTokenizerModel
except Exception:  # pragma: no cover
    NPQHierResidualTokenizerModel = None  # type: ignore


@dataclass
class TokenSequenceMeta:
    model_type: str
    seq_len: int
    vocab_size: int
    layout: Optional[Dict[str, Any]] = None


def bsq_group_codes_to_int(q: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    q: [..., D]  (sign / straight-through) — use sign bit
    Returns: [..., n_groups] in [0, 2^gs)
    """
    bits = (q > 0).to(torch.int64)
    d = int(bits.shape[-1])
    if group_size <= 0 or d % group_size != 0:
        raise ValueError(f"bad group_size={group_size} for D={d}")
    n_g = d // group_size
    g = bits.reshape(*bits.shape[:-1], n_g, group_size)
    p = 2 ** torch.arange(group_size, device=q.device, dtype=torch.int64)
    return (g * p).sum(dim=-1)


def bsq_uneven_group_codes_to_int(q: torch.Tensor, groups: List[int]) -> Tuple[torch.Tensor, int]:
    """
    q: [..., D] with D == sum(groups).  Each group packs its sign bits to an
    integer in [0, 2^g_i).  When the groups are not all the same size, group i
    is offset by sum(2^g_j for j < i) so ids live in one vocabulary of size
    sum(2^g_i).  Equal-size groups share one vocabulary (no offset), which keeps
    this identical to `bsq_group_codes_to_int` for e.g. [6, 6, 6].
    Returns ([..., n_groups], vocab_size).
    """
    groups = [int(g) for g in groups]
    d = int(q.shape[-1])
    if any(g <= 0 for g in groups) or sum(groups) != d:
        raise ValueError(f"bad bottom_groups={groups} for D={d}")
    bits = (q > 0).to(torch.int64)
    uneven = len(set(groups)) > 1
    cols: List[torch.Tensor] = []
    start, offset = 0, 0
    for g in groups:
        blk = bits[..., start : start + g]
        p = 2 ** torch.arange(g, device=q.device, dtype=torch.int64)
        cols.append((blk * p).sum(dim=-1) + (offset if uneven else 0))
        start += g
        offset += 2**g
    vocab = offset if uneven else 2 ** groups[0]
    return torch.stack(cols, dim=-1), int(vocab)


def _time_major_flatten(t: torch.Tensor) -> torch.Tensor:
    """
    Reorder [B, N_spatial, Tq, n_groups] → [B, Tq*N_spatial*n_groups] (time-major).

    This places all spatial tokens for time-slot 0 first, then time-slot 1, etc.,
    matching the user's desired layout: (t0 tok0…tokN) (t1 tok0…tokN) …
    """
    b = t.shape[0]
    # [B, N, Tq, ng] → [B, Tq, N, ng] → [B, Tq*N*ng]
    return t.permute(0, 2, 1, 3).contiguous().reshape(b, -1)


def discretize_vqvae_single(
    m: SingleLevelVQVAEModel,
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, int, TokenSequenceMeta]:
    """
    Uses pre-computed model output.  Indices are recovered by re-quantising the
    post-quant tensor (straight-through is injective for codebook lookup).
    We re-run only the quantiser step on the already-encoded z0 from `out`.
    """
    z = out["z0"]  # [B, N0, D]  (encoder output, pre-quant)
    _q, st = m.quant(z)
    idx = st["indices"]  # [B, N0]  — spatial, no time axis, already "time-major" trivially
    cb = int(m.quant.codebook_size)
    meta = TokenSequenceMeta("vqvae_single", int(idx.shape[1]), int(cb), {"n_chunks": int(idx.shape[1])})
    return idx, cb, meta


def discretize_learned_hier(
    m: LearnedHierarchyTokenizerModel,
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, int, TokenSequenceMeta]:
    """
    Uses already-computed h1, h2 (pre-quant encoder outputs) to get codebook indices.
    Note: out["q1"]/out["q2"] are post-quant (straight-through), so we feed the
    pre-quant embeddings h1/h2 into the quantiser to recover integer indices.
    """
    h1, h2 = out["h1"], out["h2"]  # pre-quant encoder outputs [B, n_m, D], [B, n_t, D]
    stages = getattr(m.core.quant_middle, "codebook_sizes", None)
    if stages is not None:
        # RVQ middle level (ms_vqvae415_rvq): finest-level codes only, as for every other AR row.
        # Per middle token [stage-0 id, stage-1 id + K0, ...], the flat RVQ AR layout.
        # h1 is autocast-dtype; re-quantize under the same autocast as the forward pass.
        with torch.autocast(device_type=h1.device.type, dtype=h1.dtype, enabled=h1.dtype != torch.float32):
            _qq1, s1 = m.core.quant_middle(h1)
        ids, off = [], 0
        for k, i in zip(stages, s1["stage_indices"]):
            ids.append(i + off)
            off += int(k)
        f = torch.stack(ids, dim=-1).reshape(h1.shape[0], -1)  # [B, n_m * n_stages]
        meta = TokenSequenceMeta("model2", int(f.shape[1]), int(off), {"middle_rvq_sizes": list(stages), "n_middle": int(h1.shape[1]), "levels": "middle only"})
        return f, int(off), meta
    _qq1, s1 = m.core.quant_middle(h1)
    _qq2, s2 = m.core.quant_top(h2)
    i1, i2 = s1["indices"], s2["indices"]  # [B, n_m], [B, n_t]
    mid = int(m.core.quant_middle.codebook_size)
    # Spatial only (no Tq), concat middle then top (coarser first is natural)
    f = torch.cat([i1, i2 + mid], dim=1)
    vocab = int(m.core.quant_middle.codebook_size + m.core.quant_top.codebook_size)
    meta = TokenSequenceMeta("model2", int(f.shape[1]), int(vocab), {"middle_cb": mid, "n_middle": int(i1.shape[1]), "n_top": int(i2.shape[1])})
    return f, vocab, meta


def discretize_npq_chunk(
    m: Union[NPQChunkTokenizerModel, NPQParcelTokenizerModel],
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, int, TokenSequenceMeta]:
    """Uses already-computed q0 from model output (chunk or parcel BSQ)."""
    q = out["q0"]  # [B, N0, Dq]
    gs = int(m.bsq.group_size)
    g = bsq_group_codes_to_int(q, gs)  # [B, N0, n_g]
    # N0 spatial chunks, no time axis — trivially spatial == time (single TR-window)
    f = g.reshape(g.shape[0], -1)
    v = int(2**gs)
    label = "npq_parcel" if isinstance(m, NPQParcelTokenizerModel) else "npq_chunk"
    meta = TokenSequenceMeta(label, int(f.shape[1]), int(v), {"group_size": gs, "n_chunks": int(q.shape[1])})
    return f, v, meta


def discretize_npq_hier_residual(
    m: "NPQHierResidualTokenizerModel",
    out: Dict[str, torch.Tensor],
    bottom_only: bool = True,
    bottom_groups: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int, TokenSequenceMeta]:
    """
    Convert hierarchical BSQ codes to flat integer token IDs.

    bottom_groups (optional, bottom_only=True only): explicit bit-group sizes for
        the bottom level, e.g. [10, 9] for a 19-bit code, used instead of the
        model's group_size.  See `bsq_uneven_group_codes_to_int` for the id
        layout.  Omitted -> byte-identical to the default path.

    bottom_only=True (default):
        Returns ONLY the bottom (voxel-chunk) level codes in time-major order.
        vocab = 2^g0  (e.g. 64 for group_size=6).
        Sequence: (t0: all chunk codes) (t1: all chunk codes) …
        This gives a clean, single-level AR task comparable to npq_bsq_chunk.

    bottom_only=False:
        All three levels interleaved per time-slot with disjoint vocab offsets.
        vocab = 2^g0 + 2^g1 + 2^g2.  Kept for completeness; not recommended for AR.
    """
    if NPQHierResidualTokenizerModel is None:
        raise RuntimeError("NPQHierResidualTokenizerModel not available")

    if str(getattr(m, "quantizer", "bsq")) != "bsq":
        raise ValueError(
            "discretize_npq_hier_residual requires a quantized model; this model was "
            "built with quantizer='none' (continuous-bottleneck control), so q* are "
            "continuous floats and have no integer token IDs."
        )

    g0 = int(m.level_bottom.bsq.group_size)
    q0 = out["q0"]  # [B, N0, Tq, D]
    tq = int(q0.shape[2])

    # A 1- or 2-level model has no q1 / q2; only the bottom_only path is available.
    q1 = out.get("q1")
    q2 = out.get("q2")
    if not bottom_only:
        if q1 is None or q2 is None:
            raise ValueError(
                "bottom_only=False needs all three levels; this model produced "
                f"{sorted(k for k in ('q0', 'q1', 'q2') if out.get(k) is not None)}."
            )
        g1 = int(m.level_middle.bsq.group_size)
        g2 = int(m.level_top.bsq.group_size)
        tq1, tq2 = int(q1.shape[2]), int(q2.shape[2])
        if not (tq == tq1 == tq2):
            raise ValueError(
                f"Time-major interleaving requires equal Tq across levels; "
                f"got tq0={tq}, tq1={tq1}, tq2={tq2}."
            )

    if bottom_groups is not None and not bottom_only:
        raise ValueError("bottom_groups is only supported with bottom_only=True")
    if bottom_groups is not None:
        t0, m0 = bsq_uneven_group_codes_to_int(q0, bottom_groups)  # [B, N0, Tq, len(groups)]
    else:
        t0 = bsq_group_codes_to_int(q0, g0)  # [B, N0, Tq, ng0]
        m0 = int(2**g0)
    b = int(q0.shape[0])
    n0, ng0 = int(t0.shape[1]), int(t0.shape[3])

    if bottom_only:
        # [B, N0, Tq, ng0] → [B, Tq, N0, ng0] → [B, Tq*N0*ng0]
        f = t0.permute(0, 2, 1, 3).contiguous().reshape(b, -1)
        vs = m0
        toks_per_window = int(f.shape[1])
        meta = TokenSequenceMeta(
            "npq_hier_residual",
            toks_per_window,
            vs,
            {
                "m0": m0, "tq": tq,
                "n_bottom": n0, "ng0": ng0,
                "toks_per_time_slot": n0 * ng0,
                "layout": "time_major_bottom_only",
                "bottom_groups": (
                    [int(g) for g in bottom_groups] if bottom_groups is not None else None
                ),
                "vocab_size": vs,
            },
        )
        return f, vs, meta

    # All-levels path (kept for completeness).
    t1 = bsq_group_codes_to_int(q1, g1)  # [B, P,    Tq, ng1]
    t2 = bsq_group_codes_to_int(q2, g2)  # [B, Nnet, Tq, ng2]
    np_, ng1 = int(t1.shape[1]), int(t1.shape[3])
    nn_, ng2 = int(t2.shape[1]), int(t2.shape[3])
    m1, m2 = int(2**g1), int(2**g2)

    t0r = t0.permute(0, 2, 1, 3).contiguous().reshape(b, tq, n0 * ng0)
    t1r = t1.permute(0, 2, 1, 3).contiguous().reshape(b, tq, np_ * ng1) + m0
    t2r = t2.permute(0, 2, 1, 3).contiguous().reshape(b, tq, nn_ * ng2) + (m0 + m1)
    f = torch.cat([t0r, t1r, t2r], dim=2).reshape(b, -1)

    vs = m0 + m1 + m2
    toks_per_window = int(f.shape[1])
    toks_per_slot = toks_per_window // tq
    meta = TokenSequenceMeta(
        "npq_hier_residual",
        toks_per_window,
        vs,
        {
            "m0": m0, "m1": m1, "m2": m2,
            "tq": tq,
            "toks_per_time_slot": toks_per_slot,
            "n_bottom": n0, "n_middle": np_, "n_top": nn_,
            "ng0": ng0, "ng1": ng1, "ng2": ng2,
            "layout": "time_major",
        },
    )
    return f, int(vs), meta


def discretize_model(
    model: nn.Module,
    model_type: str,
    batch: Dict[str, torch.Tensor],
    out: Optional[Dict[str, torch.Tensor]] = None,
    bottom_groups: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int, TokenSequenceMeta]:
    """
    Convert one batch to flat integer token ids.  `out` must be the already-computed
    model forward output (avoids a second pass).  For npq_hier_residual `out` is
    mandatory since all three q levels are needed.
    """
    if out is None:
        raise ValueError("discretize_model: `out` (model forward output) is required to avoid a second forward pass")
    if model_type == "vqvae_single" and isinstance(model, SingleLevelVQVAEModel):
        return discretize_vqvae_single(model, out)
    if model_type == "model2" and isinstance(model, LearnedHierarchyTokenizerModel):
        return discretize_learned_hier(model, out)
    if model_type == "npq_chunk" and isinstance(model, NPQChunkTokenizerModel):
        return discretize_npq_chunk(model, out)
    if model_type == "npq_parcel" and isinstance(model, NPQParcelTokenizerModel):
        return discretize_npq_chunk(model, out)
    if model_type == "npq_hier_residual" and NPQHierResidualTokenizerModel is not None:
        assert isinstance(model, NPQHierResidualTokenizerModel)
        return discretize_npq_hier_residual(model, out, bottom_groups=bottom_groups)
    if bottom_groups is not None:
        raise ValueError(f"bottom_groups is only supported for npq_hier_residual (type={model_type!r})")
    raise ValueError(f"discretize_model: unsupported (type={model_type!r}, model={type(model)})")


def _mean_code(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 3:
        return q.mean(dim=1)
    return q.reshape(q.shape[0], -1, q.shape[-1]).mean(dim=1)


def pooled_representation(
    out: Dict[str, torch.Tensor],
    model_type: str,
) -> torch.Tensor:
    """Pooled (mean) post-quant vectors for linear probe, concatenated.  [B, Dtot]"""
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        if "q0" in out and out.get("q0") is not None and model_type in {
            "npq_hier_residual",
            "vqvae_single",
            "npq_chunk",
            "npq_parcel",
        }:
            parts.append(_mean_code(out["q0"]))
        if "q1" in out and out.get("q1") is not None and model_type == "npq_hier_residual":
            parts.append(_mean_code(out["q1"]))
        if "q2" in out and out.get("q2") is not None and model_type == "npq_hier_residual":
            parts.append(_mean_code(out["q2"]))
        if model_type == "model2":
            if "q1" in out and out.get("q1") is not None:
                parts.append(_mean_code(out["q1"]))
            if "q2" in out and out.get("q2") is not None:
                parts.append(_mean_code(out["q2"]))
    if not parts:
        raise ValueError("pooled_representation: no q* tensors in model output")
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
