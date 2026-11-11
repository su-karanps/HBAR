#!/usr/bin/env python3
"""
Causal next-token LM on precomputed per-window token ids.  Trains a small
causal Transformer; reports cross-entropy.  Optional: subject readout
(mlp on mean-pooled last hidden over shuffled subject chunks) for age/sex
when HCP metadata is available.
"""

from __future__ import annotations

import argparse
import json
import gc
import math
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from downstream.hcp_metadata import _norm_key, load_hcp_csv, infer_subject_id_from_record


def load_token_pack(d: str, tag: str) -> Dict[str, Any]:
    keys = f"meta_{tag}.json", f"token_ids_{tag}.npy", f"token_indptr_{tag}.npy", f"rec_idx_{tag}.npy", f"win_start_{tag}.npy"
    paths = [os.path.join(d, k) for k in keys]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing: {p}")
    with open(paths[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "meta": meta,
        "token_ids": np.load(paths[1], mmap_mode="r"),
        "indptr": np.load(paths[2]),
        "rec_idx": np.load(paths[3]),
        "win_start": np.load(paths[4]),
    }


def per_subject_chrono_sequences(
    rec_idx: np.ndarray,
    win_st: np.ndarray,
    ind: np.ndarray,
    tok: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Build per-subject chronological token sequences.  Windows of one subject are
    normally stored back-to-back in chronological order in the dump, in which
    case the subject sequence is a single slice of `tok` (a memmap view when
    `tok` is memory-mapped, so nothing is copied into RAM); otherwise the
    windows are concatenated.
    """
    key = rec_idx.astype(np.int64) * 1_000_000 + win_st.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    by: Dict[int, List[int]] = defaultdict(list)
    for oi_np in order:
        oi = int(oi_np)
        by[int(rec_idx[oi])].append(oi)  # window rows in chronological order
    out: Dict[int, np.ndarray] = {}
    for r, rows in by.items():
        contiguous = all(rows[i + 1] == rows[i] + 1 for i in range(len(rows) - 1))
        if contiguous:
            out[int(r)] = tok[int(ind[rows[0]]) : int(ind[rows[-1] + 1])]
        else:
            out[int(r)] = np.concatenate([tok[int(ind[oi]) : int(ind[oi + 1])] for oi in rows])
    return out


def concat_corpus(sequences: Dict[int, np.ndarray], m_vocab: int) -> Tuple[np.ndarray, int]:
    """
    1+offset token ids, zero reserved for padding in CE.  One SEP between subjects.
    Returns a numpy int32 array (28x smaller than Python list of ints).
    """
    sep = 2 + int(m_vocab)
    parts: List[np.ndarray] = []
    sep_arr = np.array([sep], dtype=np.int32)
    for r in sorted(sequences):
        shifted = (sequences[int(r)].astype(np.int32) + 1)
        parts.append(shifted)
        parts.append(sep_arr)
    corpus = np.concatenate(parts) if parts else np.array([], dtype=np.int32)
    return corpus, int(sep)


class ChunkDataset(Dataset):
    def __init__(self, data: "Union[List[int], np.ndarray]", chunk: int, patch_size: int = 1) -> None:
        # Keep the corpus as int32 (12e9 ids for the hierarchical dumps); each
        # item is converted to int64 on access.
        self.t = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.int32)
        if self.t.dtype != np.int32:
            self.t = self.t.astype(np.int32)
        ps = max(1, int(patch_size))
        self.ps = ps
        # Round chunk down to a multiple of patch_size so patches align cleanly.
        chunk = max(ps, (int(chunk) // ps) * ps)
        # We need chunk + patch_size tokens to produce one input patch sequence
        # (chunk tokens as input, predicting the patch_size tokens that follow).
        c = chunk + ps
        self.chunk = chunk
        self.c = c
        n = int(self.t.shape[0])
        # Stride = chunk (non-overlapping inputs)
        self.m = max(0, (n - c) // chunk + 1) if n >= c else 0

    def __len__(self) -> int:
        return int(self.m)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = i * self.chunk
        w = self.t[s : s + self.c]
        # x: the chunk_len input tokens; y: next patch_size tokens per position
        x = torch.from_numpy(np.asarray(w[: self.chunk], dtype=np.int64))
        # y is the target for each patch position: patch p predicts tokens
        # starting at (p+1)*patch_size ... (p+2)*patch_size-1
        # Shift by patch_size, then view as [n_patches, patch_size]
        y_flat = torch.from_numpy(np.asarray(w[self.ps : self.chunk + self.ps], dtype=np.int64))
        return {"x": x, "y": y_flat}


class CausalTfm(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        dim: int,
        nhead: int,
        nlayers: int,
        max_pos: int = 12_000,
        patch_size: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = max(1, int(patch_size))
        self.nv = n_vocab
        self.dim = int(dim)
        # +1: index 0 reserved for padding / CE ignore
        # Each of the patch_size sub-tokens gets its own embedding of dim // patch_size,
        # then they are concatenated to produce a dim-dimensional patch embedding.
        self.sub_dim = max(1, dim // self.patch_size)
        # Actual dim used (may be slightly less than requested if not divisible)
        self.actual_dim = self.sub_dim * self.patch_size
        self.emb = nn.Embedding(n_vocab + 1, self.sub_dim, padding_idx=0)
        if self.actual_dim != dim:
            # Project to requested dim when patch_size does not divide dim evenly
            self.patch_proj: Optional[nn.Module] = nn.Linear(self.actual_dim, dim, bias=False)
        else:
            self.patch_proj = None
        self.max_pos = max(int(max_pos), 64)
        self.pos = nn.Parameter(torch.zeros(1, self.max_pos, dim))
        el = nn.TransformerEncoderLayer(
            dim, nhead, 4 * dim, batch_first=True, norm_first=True, activation="gelu"
        )
        self.nlayers = int(nlayers)
        self.tr = nn.TransformerEncoder(el, num_layers=nlayers)
        # Tri-state cache:
        # None  -> unknown (try flash-friendly causal path first)
        # True  -> causal path without explicit mask works
        # False -> must use explicit square mask fallback
        self._supports_causal_nomask: Optional[bool] = None
        # Output head predicts all patch_size sub-tokens for the *next* patch.
        # Logits: [B, n_patches, patch_size * (n_vocab+1)]
        self.head = nn.Linear(dim, self.patch_size * (n_vocab + 1), bias=True)

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]  flat token ids (T must be a multiple of patch_size)
        returns: [B, T // patch_size, dim]
        """
        b, t = x.shape
        ps = self.patch_size
        if t % ps != 0:
            # Truncate to last complete patch
            x = x[:, : t - (t % ps)]
            t = x.shape[1]
        # [B, n_patches, patch_size]
        xp = x.reshape(b, t // ps, ps)
        # [B, n_patches, patch_size, sub_dim]
        e = self.emb(xp.clamp_min(0))
        # [B, n_patches, patch_size * sub_dim]
        e = e.reshape(b, t // ps, ps * self.sub_dim)
        if self.patch_proj is not None:
            e = self.patch_proj(e)
        return e  # [B, n_patches, dim]

    def forward(
        self, x: torch.Tensor, need_hidden: bool = False, apply_head: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """apply_head=False returns (None, hidden) so the caller can run the head in chunks."""
        b, t = x.shape
        h = self._embed_patches(x)          # [B, n_patches, dim]
        n_p = h.shape[1]
        if n_p > self.max_pos:
            raise ValueError(f"n_patches {n_p} > max_pos {self.max_pos}")
        h = h + self.pos[:, :n_p, :]
        h2: torch.Tensor
        used_nomask = False
        if self._supports_causal_nomask is not False:
            # Prefer no explicit attention mask; this is the path PyTorch uses
            # for flash/memory-efficient SDPA kernels when available.
            try:
                h2 = self.tr(h, mask=None, is_causal=True)  # type: ignore[call-arg]
                used_nomask = True
                if self._supports_causal_nomask is None:
                    self._supports_causal_nomask = True
            except TypeError:
                self._supports_causal_nomask = False
            except RuntimeError:
                # Some torch versions/configs reject is_causal without a mask.
                self._supports_causal_nomask = False

        if not used_nomask:
            mask = nn.Transformer.generate_square_subsequent_mask(n_p, device=x.device)
            try:
                h2 = self.tr(h, mask=mask, is_causal=True)  # type: ignore[call-arg]
            except TypeError:
                h2 = self.tr(h, mask=mask)
        if not apply_head:
            return None, h2
        logits = self.head(h2)   # [B, n_patches, patch_size * (n_vocab+1)]
        return (logits, h2) if need_hidden else (logits, None)


def _collate(batch: List) -> Dict[str, torch.Tensor]:
    return {"x": torch.stack([b["x"] for b in batch], 0), "y": torch.stack([b["y"] for b in batch], 0)}


def _ce_sum_chunked(
    model: CausalTfm, h2: torch.Tensor, y: torch.Tensor, max_logit_bytes: float = 4e9
) -> Tuple[torch.Tensor, int]:
    """
    Summed next-patch CE (ignore_index=0) with the output head applied in chunks of
    positions, each under activation checkpointing, so peak memory is bounded by
    one chunk's fp32 logits (<= max_logit_bytes) instead of the full
    [B, n_patches, patch_size*(n_vocab+1)] tensor (18 GB for a 1,536-vocab
    tokenizer at batch 4).  Mathematically identical to the unchunked loss; a
    single chunk is used whenever the full tensor fits (e.g. the 64-vocab case).
    Returns (loss_sum, n_valid_targets).
    """
    from torch.utils.checkpoint import checkpoint

    ps, nv1 = model.patch_size, model.nv + 1
    b, n_p = h2.shape[0], h2.shape[1]
    y_p = y[:, : n_p * ps]
    full_bytes = float(b * n_p * ps * nv1 * 4)
    n_chunks = max(1, int(math.ceil(full_bytes / max_logit_bytes)))
    step = int(math.ceil(n_p / n_chunks))

    def _chunk_loss(hs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        logits = model.head(hs).reshape(-1, nv1)
        return F.cross_entropy(logits.float(), ys.reshape(-1), ignore_index=0, reduction="sum")

    total = h2.new_zeros((), dtype=torch.float32)
    for c0 in range(0, n_p, step):
        c1 = min(n_p, c0 + step)
        hs, ys = h2[:, c0:c1], y_p[:, c0 * ps : c1 * ps]
        if torch.is_grad_enabled() and n_chunks > 1:
            total = total + checkpoint(_chunk_loss, hs, ys, use_reentrant=False)
        else:
            total = total + _chunk_loss(hs, ys)
    return total, int((y_p > 0).sum().item())


def _token_entropy(corpus: List[int], vocab_size: int) -> Tuple[float, float, int]:
    """Returns (entropy_bits, frac_vocab_used, n_unique_tokens)."""
    arr = np.asarray(corpus).reshape(-1)
    counts = np.zeros(int(vocab_size) + 3, dtype=np.int64)
    step = 200_000_000  # bincount casts to int64 internally; chunk to bound the copy
    for i in range(0, arr.shape[0], step):
        c = np.bincount(arr[i : i + step], minlength=counts.shape[0])
        counts[: c.shape[0]] += c[: counts.shape[0]]
    counts = counts[1:]  # skip padding id 0
    counts = counts[counts > 0]
    total = int(counts.sum())
    if total == 0:
        return 0.0, 0.0, 0
    probs = counts.astype(np.float64) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    frac = len(counts) / max(1, vocab_size)
    return entropy, frac, int(len(counts))


def _fmt_dur(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(round(seconds)))))


def _load_split_sets(split_json: str) -> Dict[str, set[str]]:
    """
    Returns {split_name: normalised subject ids} for every key among
    train/val/test present in the file.  A canonical split file has all three;
    a legacy file (or a bare list) only defines `val`, in which case the caller
    treats every other subject as train.
    """
    with open(split_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    raw: Dict[str, List[Any]] = {}
    if isinstance(obj, dict):
        for name in ("train", "val", "test"):
            if isinstance(obj.get(name), list):
                raw[name] = obj[name]
    elif isinstance(obj, list):
        raw["val"] = obj
    out: Dict[str, set[str]] = {}
    for name, vals in raw.items():
        out[name] = {_norm_key(str(v)) for v in vals if v is not None}
    return out


@torch.no_grad()
def causality_test(model: CausalTfm, batch: Dict[str, torch.Tensor], dev: torch.device, tol: float = 1e-4) -> Dict[str, Any]:
    """
    Untrained-model unit test, run before training:
      1. hidden states at patch positions <= p are unchanged (fp32, max abs diff
         < tol) when every id at positions > p is replaced, in eval mode (the
         readout condition) and in train mode (seeded dropout);
      2. the CE target at position p is exactly the ids of patch p+1.
    Returns a dict with PASS/FAIL and the measured differences.
    """
    ps = model.patch_size
    x, y = batch["x"].to(dev), batch["y"].to(dev)
    n_p = x.shape[1] // ps
    g = torch.Generator(device="cpu").manual_seed(1234)
    res: Dict[str, Any] = {"n_patches": int(n_p), "patch_size": int(ps), "tol": tol, "checks": []}
    ok = True
    was_training = model.training
    for p in sorted({0, 1, n_p // 2, n_p - 2}):
        x2 = x.clone()
        x2[:, (p + 1) * ps :] = torch.randint(1, model.nv + 1, x2[:, (p + 1) * ps :].shape, generator=g).to(dev)
        for mode in ("eval", "train"):
            model.eval() if mode == "eval" else model.train()
            model._supports_causal_nomask = None  # exercise the mask-selection logic each time
            torch.manual_seed(0)
            _, h1 = model(x, need_hidden=True)
            torch.manual_seed(0)
            _, h2 = model(x2, need_hidden=True)
            assert h1 is not None and h2 is not None
            past = float((h1[:, : p + 1].float() - h2[:, : p + 1].float()).abs().max())
            future = float((h1[:, p + 1 :].float() - h2[:, p + 1 :].float()).abs().max())
            passed = past < tol and future > tol
            ok = ok and passed
            res["checks"].append({"p": int(p), "mode": mode, "max_abs_diff_past": past, "max_abs_diff_future": future, "pass": passed})
    model.train(was_training)
    # Target check: y[:, p*ps:(p+1)*ps] must equal x patch p+1 for every p < n_p-1
    # (the last target patch lies beyond the input and comes from ChunkDataset).
    tgt_ok = bool(torch.equal(y[:, : (n_p - 1) * ps], x[:, ps : n_p * ps]))
    res["target_is_next_patch"] = tgt_ok
    ok = ok and tgt_ok
    res["result"] = "PASS" if ok else "FAIL"
    print(f"[causality_test] {res['result']}  target_is_next_patch={tgt_ok}  "
          + "  ".join(f"p={c['p']}/{c['mode']}:past={c['max_abs_diff_past']:.2e},future={c['max_abs_diff_future']:.2e}" for c in res["checks"]),
          flush=True)
    return res


def _train_loop(
    model: CausalTfm,
    dl: DataLoader,
    dev: torch.device,
    epochs: int,
    lr: float,
    tag: str,
    warmup_steps: int = 200,
    lr_decay: str = "cosine",
    log_every: int = 20,
    val_dl: Optional[DataLoader] = None,
    use_amp: bool = True,
) -> List[Dict[str, float]]:
    """Returns per-epoch metrics list.

    lr_decay: 'cosine' (recommended) | 'none'
    warmup_steps: linear LR warm-up from 0 → lr over this many steps.
    log_every: print within-epoch progress every this many batches.
    val_dl: optional validation DataLoader; if provided, val CE is reported each epoch.

    Cosine decay spans exactly one epoch's worth of steps after warmup, then
    holds at the minimum LR for remaining epochs.  This prevents the optimizer
    from running on near-zero LR for the majority of training when many epochs
    are used (previously decay spanned all epochs, leaving lr≈0 from epoch 2
    onwards for long token sequences like bsq_hier).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ps = model.patch_size
    nv1 = model.nv + 1
    steps_per_epoch = len(dl)
    total_steps = epochs * steps_per_epoch
    # Decay over one epoch after warmup rather than all epochs, so the schedule
    # is not effectively dead for epochs 2-N.
    decay_steps = max(1, steps_per_epoch - warmup_steps)

    def _get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        if lr_decay == "cosine":
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            # Floor at 20% of peak so epochs 2-N still train meaningfully.
            # (1% floor caused LR to crater to ~0 after ep1, wasting 80% of training.)
            return lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr

    def _set_lr(step: int) -> float:
        new_lr = _get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = new_lr
        return new_lr

    amp_enabled = use_amp and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    history: List[Dict[str, float]] = []
    global_step = 0
    run_t0 = time.perf_counter()

    for e in range(epochs):
        epoch_t0 = time.perf_counter()
        loss_sum, tok_sum, n, step = 0.0, 0, 0, 0
        interval_loss, interval_tok = 0.0, 0

        for batch in dl:
            cur_lr = _set_lr(global_step)
            x, y = batch["x"].to(dev), batch["y"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
                _, h2 = model(x, apply_head=False)
                b = int(x.shape[0])
                ce_sum, valid = _ce_sum_chunked(model, h2, y)
                loss = ce_sum / max(1, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.item()) * valid
            loss_sum += loss_val
            tok_sum += valid
            interval_loss += loss_val
            interval_tok += valid
            n += int(b)
            step += 1
            global_step += 1

            if step % log_every == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - epoch_t0
                sps = step / max(elapsed, 1e-6)
                eta_epoch = (steps_per_epoch - step) / max(sps, 1e-6)
                eta_run = (total_steps - global_step) / max(global_step / max(time.perf_counter() - run_t0, 1e-6), 1e-6)
                interval_ce = interval_loss / max(1, interval_tok)
                print(
                    f"  [{tag}][ep {e+1}/{epochs}] "
                    f"step {step}/{steps_per_epoch} ({100*step//steps_per_epoch}%)  "
                    f"ce={interval_ce:.4f}  lr={cur_lr:.2e}  "
                    f"ep_eta={_fmt_dur(eta_epoch)}  run_eta={_fmt_dur(eta_run)}",
                    flush=True,
                )
                interval_loss, interval_tok = 0.0, 0

        epoch_secs = time.perf_counter() - epoch_t0
        ce = loss_sum / max(1, tok_sum)
        ppl = float(np.exp(min(30, ce)))
        rec = {"epoch": e + 1, "ce": ce, "ppl": ppl, "epoch_secs": epoch_secs, "lr": _get_lr(global_step)}

        # Validation CE (no gradient, no LR update).
        val_ce_str = ""
        if val_dl is not None:
            model.eval()
            val_loss, val_tok = 0.0, 0
            for vbatch in val_dl:
                vx, vy = vbatch["x"].to(dev), vbatch["y"].to(dev)
                with torch.no_grad(), torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
                    _, vh2 = model(vx, apply_head=False)
                    vloss, valid_v = _ce_sum_chunked(model, vh2, vy)
                val_loss += float(vloss.item())
                val_tok += valid_v
            model.train()
            val_ce = val_loss / max(1, val_tok)
            rec["val_ce"] = val_ce
            val_ce_str = f"  val_ce={val_ce:.4f}"

        history.append(rec)
        print(
            f"[{tag}] epoch {e+1}/{epochs}  ce={ce:.4f}  ppl={ppl:.3f}{val_ce_str}  "
            f"time={_fmt_dur(epoch_secs)}  lr={_get_lr(global_step):.2e}"
            + (f"  (patch_size={ps})" if ps > 1 else ""),
            flush=True,
        )

    return history


@torch.inference_mode()
def _pooled_hidden_features(
    per_sub: Dict[int, np.ndarray],
    model: CausalTfm,
    rec_subj: List[str],
    pheno: Dict[str, Tuple[float, int]],
    dev: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Mean-pooled last hidden state over each subject's whole chronological sequence."""
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    hiddens: List[np.ndarray] = []
    lab_age: List[float] = []
    lab_sex: List[int] = []
    hid_subj: List[str] = []
    ps = model.patch_size
    max_toks = model.max_pos * ps  # max tokens that fit in one forward pass
    for r in sorted(per_sub):
        seq = per_sub[r]
        if r < 0 or r >= len(rec_subj):
            continue
        sid = str(rec_subj[int(r)])
        if len(seq) < 8:
            continue
        k = _norm_key(sid)
        if k not in pheno:
            continue
        a, s = pheno[k]
        # +1 offset matches `concat_corpus` + training.  Process the sequence in
        # max_pos-sized chunks, then mean-pool hidden states.
        tarray = torch.from_numpy(np.asarray(seq, dtype=np.int64) + 1).to(dev)
        chunk_hiddens: List[torch.Tensor] = []
        for start in range(0, len(tarray), max_toks):
            chunk = tarray[start : start + max_toks].unsqueeze(0)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=(use_amp and dev.type == "cuda")):
                _, h_c = model(chunk, need_hidden=True)
            if h_c is not None:
                chunk_hiddens.append(h_c.float().mean(dim=1))
        if not chunk_hiddens:
            continue
        pooled = torch.stack(chunk_hiddens).mean(dim=0).float().cpu().numpy()[0]
        hiddens.append(pooled)
        lab_age.append(float(a))
        lab_sex.append(int(s))
        hid_subj.append(k)
    if not hiddens:
        return np.zeros((0, model.dim), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64), []
    return np.stack(hiddens, axis=0), np.array(lab_age, np.float32), np.array(lab_sex, np.int64), hid_subj


def readout_pooled(
    fit_per: Dict[int, np.ndarray],
    eval_per: Dict[int, np.ndarray],
    max_tok: int,
    model: CausalTfm,
    rec_subj: List[str],
    hcp_csv: str,
    hcp_id: str,
    hcp_age: str,
    hcp_sex: str,
    device: str,
    seed: int = 0,
    n_boot: int = 10000,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    predictions_npz: Optional[str] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Frozen AR model -> mean-pooled hidden features per subject -> logistic probe
    fit on `fit_per` subjects, evaluated on `eval_per` subjects via
    downstream.probe_eval.evaluate_task (same protocol as linear_probe_hcp.py).
    """
    if (not hcp_csv) or (not os.path.isfile(hcp_csv)) or max_tok < 1:
        print("[readout] skip: no metadata file")
        return {}
    from downstream.probe_eval import evaluate_task, save_predictions

    dev = torch.device(device)
    was_training = model.training
    model.eval()
    pheno = { _norm_key(s.subject_id): (s.age, s.sex) for s in load_hcp_csv(hcp_csv, hcp_id, hcp_age, hcp_sex) }
    t0 = time.perf_counter()
    Xtr, age_tr, sex_tr, subj_tr = _pooled_hidden_features(fit_per, model, rec_subj, pheno, dev, use_amp)
    Xv, age_va, sex_va, subj_va = _pooled_hidden_features(eval_per, model, rec_subj, pheno, dev, use_amp)
    model.train(was_training)
    print(f"[readout] features: fit={len(subj_tr)} eval={len(subj_va)} subjects  dim={Xtr.shape[1] if Xtr.size else 0}  "
          f"time={_fmt_dur(time.perf_counter() - t0)}")
    if len(subj_tr) < 4 or len(subj_va) < 1:
        print(f"[readout] too few subjects: fit={len(subj_tr)} eval={len(subj_va)}")
        return {}
    if set(subj_tr) & set(subj_va):
        raise RuntimeError("[readout] fit and eval subject sets overlap")

    # Age classification: HCP-YA age is 3 bucket labels (23, 28, 33); map to class indices.
    buckets = np.unique(np.concatenate([age_tr[age_tr > 0], age_va[age_va > 0]]))
    buckets.sort()
    lut = {float(b): i for i, b in enumerate(buckets)}

    def _age_to_cls(a: np.ndarray) -> np.ndarray:
        out_arr = np.full(len(a), -1, dtype=np.int64)
        for i, v in enumerate(a):
            if v > 0 and float(v) in lut:
                out_arr[i] = lut[float(v)]
        return out_arr

    yc_tr = _age_to_cls(age_tr)
    yc_va = _age_to_cls(age_va)
    m_tr = yc_tr >= 0
    m_va = yc_va >= 0

    out: Dict[str, Any] = {
        "readout_n_fit_subjects": int(len(subj_tr)),
        "readout_n_eval_subjects": int(len(subj_va)),
        "readout_feature_dim": int(Xtr.shape[1]),
        "readout_age_classes": [float(b) for b in buckets],
    }
    pred_tasks: Dict[str, Dict[str, np.ndarray]] = {}
    pred_subjects: Dict[str, List[str]] = {}
    subj_va_arr = np.asarray(subj_va, dtype=object)

    def _run(task: str, Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
             subs: List[str]) -> None:
        res, preds = evaluate_task(
            Xa, ya, Xb, yb,
            seed=seed, n_boot=n_boot, cv_splits=cv_splits, cv_repeats=cv_repeats,
        )
        out[f"readout_{task}"] = res
        out[f"readout_{task}_acc"] = res.get("acc")
        out[f"readout_{task}_f1"] = res.get("f1_macro")
        out[f"readout_{task}_auroc"] = res.get("auroc")
        ci = (res.get("ci95") or {}).get("auroc")
        auroc = res.get("auroc")
        auroc_str = "n/a" if auroc is None else f"{auroc:.3f}"
        ci_str = "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        print(
            f"[readout] {task}  acc={res.get('acc', float('nan')):.3f}  "
            f"macro-F1={res.get('f1_macro', float('nan')):.3f}  AUROC={auroc_str}{ci_str}"
        )
        pred_tasks[task] = preds
        pred_subjects[task] = subs

    if m_tr.sum() >= 2 and len(np.unique(yc_tr[m_tr])) >= 2 and m_va.sum() >= 1:
        _run(
            "age", Xtr[m_tr, :], yc_tr[m_tr], Xv[m_va, :], yc_va[m_va],
            [str(s) for s in subj_va_arr[m_va]],
        )
    trm = sex_tr >= 0
    vam = sex_va >= 0
    if trm.sum() < 2 or vam.sum() < 1:
        print("[readout] sex: skip")
    else:
        _run(
            "sex", Xtr[trm, :], sex_tr[trm], Xv[vam, :], sex_va[vam],
            [str(s) for s in subj_va_arr[vam]],
        )

    if predictions_npz and pred_tasks:
        save_predictions(predictions_npz, pred_tasks, pred_subjects)
        print(f"[readout] saved predictions: {predictions_npz}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="a")
    p.add_argument("--chunk_len", type=int, default=0,
                   help="Raw context length in tokens.  If 0 (default), --n_context_windows is used instead.")
    p.add_argument("--n_context_windows", type=int, default=4,
                   help="Number of consecutive windows to feed as context (chunk_len = n_context_windows * seq_len_per_window). "
                        "Ignored when --chunk_len > 0.")
    p.add_argument("--patch_size", type=int, default=1,
                   help="Pack this many consecutive token IDs into one transformer position.  "
                        "Each ID gets its own embedding of dim//patch_size; they are concatenated "
                        "to form a dim-dimensional patch embedding.  The head predicts all "
                        "patch_size next-position IDs simultaneously.  "
                        "Use e.g. --patch_size 16 for hier-v10 to reduce 222k → ~14k positions/window.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Peak learning rate for AdamW (cosine-decayed from this value). "
                        "Default 2e-4; consider 5e-5 for very long sequences (hier models).")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="Linear LR warm-up from 0 → --lr over this many steps.")
    p.add_argument("--lr_decay", type=str, default="cosine", choices=["cosine", "none"],
                   help="LR schedule after warmup. 'cosine' decays to lr/100; 'none' holds constant.")
    p.add_argument("--log_every", type=int, default=20,
                   help="Print within-epoch progress every this many batches.")
    p.add_argument("--val_subjects", type=float, default=0.1,
                   help="Fraction of subjects held out for validation CE (0 to disable).")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision (bf16/fp16) for training. Default: on.")
    p.add_argument("--no_amp", dest="amp", action="store_false",
                   help="Disable AMP (fall back to fp32).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--hcp_metadata_csv", type=str, default=os.path.join(REPO, "data", "hcp_metadata.csv"))
    p.add_argument("--readout", action="store_true")
    p.add_argument("--readout_only", action="store_true",
                   help="Skip training; load existing checkpoint and run readout only.")
    p.add_argument("--max_subjects", type=int, default=0,
                   help="Cap the number of subjects loaded for AR training (0 = use all). "
                        "Useful for large token sets like bsq_hier that would OOM if fully loaded.")
    p.add_argument("--subject_split_json", type=str, default="",
                   help="Subject split JSON. Canonical files ({'train','val','test'}): AR trains on "
                        "train only, val is the held-out CE set and test is never loaded into the corpus. "
                        "Legacy files with only 'val' keep the old behaviour (every other subject is train).")
    p.add_argument("--readout_fit_split", type=str, default="train", choices=["train", "val", "test"],
                   help="Split whose subjects the readout probe is fit on.")
    p.add_argument("--readout_eval_split", type=str, default="val", choices=["val", "test"],
                   help="Split the readout probe is evaluated on.")
    p.add_argument("--causality_test", action="store_true",
                   help="Before training, check on one batch that the untrained model is causal and "
                        "that targets are the next patch; result is stored in metrics.")
    p.add_argument("--hcp_id_column", type=str, default="subject_id")
    p.add_argument("--hcp_age_column", type=str, default="age")
    p.add_argument("--hcp_sex_column", type=str, default="sex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=10000,
                   help="Bootstrap resamples for readout metric CIs.")
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--cv_repeats", type=int, default=5,
                   help="Repeats for stratified K-fold on the readout (0 disables CV).")
    p.add_argument("--predictions_npz", type=str, default="",
                   help="If set, dump per-subject readout predictions for paired significance tests.")
    a = p.parse_args()

    if torch.cuda.is_available():
        # Keep all SDPA backends enabled so PyTorch can pick flash/mem-efficient
        # kernels when the runtime/dtype/hardware allow it.
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    # The nn.TransformerEncoderLayer eval/no-grad "fast path" ignores the is_causal
    # hint when no explicit mask is given, which would make readout features
    # non-causal.  Disable it; the SDPA is_causal path is used instead.
    torch.backends.mha.set_fastpath_enabled(False)
    run_t_start = time.perf_counter()

    pack = load_token_pack(a.token_dir, a.tag)
    meta = pack["meta"]
    m_vocab = int(meta.get("vocab_size", 1024))
    seq_len_per_window = int(meta.get("seq_len_per_window", 1))
    rec_idx = pack["rec_idx"]
    wst = pack["win_start"]
    ind = pack["indptr"]
    tids = pack["token_ids"]

    # For hierarchical models (npq_hier_residual) saved with all levels interleaved,
    # strip middle and top codes and keep only bottom-level (voxel-chunk) tokens.
    # Bottom codes have values in [0, m0), middle in [m0, m0+m1), top in [m0+m1, ...).
    # The vocab offset is always present regardless of which precompute version was used.
    # This gives a clean single-level AR task (vocab=m0=64) comparable to npq_bsq_chunk.
    disc_meta = meta.get("discrete_meta", {})
    layout_info = disc_meta.get("layout", {}) if isinstance(disc_meta, dict) else {}
    if (
        meta.get("model_type") == "npq_hier_residual"
        and isinstance(layout_info, dict)
        and "m0" in layout_info
        and "m1" in layout_info  # multi-level saved file; bottom_only files only have m0
    ):
        m0 = int(layout_info["m0"])
        # Rebuild tids per-window, keeping only values < m0 (bottom-level codes).
        n_windows = len(ind) - 1
        new_ids_list: List[np.ndarray] = []
        new_ind = np.zeros(n_windows + 1, dtype=np.int64)
        for wi in range(n_windows):
            a_i, b_i = int(ind[wi]), int(ind[wi + 1])
            win_toks = tids[a_i:b_i]
            bot_toks = win_toks[win_toks < m0]
            new_ind[wi + 1] = new_ind[wi] + len(bot_toks)
            new_ids_list.append(bot_toks)
        old_tids = tids
        tids = np.concatenate(new_ids_list).astype(np.int32) if new_ids_list else np.array([], dtype=np.int32)
        del old_tids, new_ids_list
        gc.collect()
        ind = new_ind
        m_vocab = m0
        bot_per_window = int(new_ind[1]) if n_windows > 0 else 0  # infer from first window
        seq_len_per_window = bot_per_window
        print(
            f"[AR] npq_hier_residual: stripped to bottom-only "
            f"(~{seq_len_per_window} toks/window, vocab={m_vocab})"
        )

    per = per_subject_chrono_sequences(rec_idx, wst, ind, tids)
    del tids, ind  # free raw token array before building corpus
    gc.collect()
    if a.max_subjects > 0 and len(per) > a.max_subjects:
        rng_sub = np.random.default_rng(42)
        keys = sorted(per.keys())
        chosen = rng_sub.choice(keys, size=a.max_subjects, replace=False)
        per = {k: per[k] for k in chosen}
        print(f"[AR] max_subjects={a.max_subjects}: using {len(per)} of {len(keys)} subjects")

    rec_subj: Optional[List[str]] = None
    split_sets: Optional[Dict[str, set[str]]] = None
    if a.subject_split_json and os.path.isfile(a.subject_split_json):
        try:
            split_sets = _load_split_sets(a.subject_split_json)
            with open(meta["index_path"], "r", encoding="utf-8") as f:
                idxrec = json.load(f)
            rec_subj = [infer_subject_id_from_record(r) for r in idxrec]
            print(f"[AR] loaded split_json {a.subject_split_json}: "
                  + ", ".join(f"{k}={len(v)}" for k, v in split_sets.items()))
        except Exception as e:
            print(f"[AR] warning: failed to load split_json ({e}); falling back to random val split")
            split_sets = None

    # Partition subjects: split_per[name] -> {rec_idx: sequence}.  The AR corpus is
    # built from split_per["train"] only; "val" gives held-out CE; "test" (if
    # present) is only ever used for the readout when --readout_eval_split test.
    split_per: Dict[str, Dict[int, np.ndarray]] = {}
    used_split_json = False
    if split_sets and rec_subj is not None and "val" in split_sets:
        rec_split: Dict[int, str] = {}
        for i, sid in enumerate(rec_subj):
            k = _norm_key(str(sid))
            for name, ss in split_sets.items():
                if k in ss:
                    rec_split[i] = name
        if "train" in split_sets:
            for name in split_sets:
                split_per[name] = {k: v for k, v in per.items() if rec_split.get(k) == name}
            n_unlisted = sum(1 for k in per if k not in rec_split)
            if n_unlisted:
                print(f"[AR] {n_unlisted} token subjects are not in any split and are dropped")
        else:  # legacy: only val is listed, every other subject is train
            split_per["val"] = {k: v for k, v in per.items() if rec_split.get(k) == "val"}
            split_per["train"] = {k: v for k, v in per.items() if rec_split.get(k) != "val"}
        if split_per.get("train") and split_per.get("val"):
            used_split_json = True
            print("[AR] subjects per split (from split_json): "
                  + ", ".join(f"{k}={len(v)}" for k, v in split_per.items()))
        else:
            print("[AR] warning: split_json overlap produced empty train/val; falling back to random val split")
            split_per = {}
    if not used_split_json:
        if a.readout_eval_split == "test":
            raise SystemExit("--readout_eval_split test requires a canonical --subject_split_json with a test key")
        split_per = {"train": per, "val": {}}
        if a.val_subjects > 0.0:
            all_subs = sorted(per.keys())
            n_val = max(1, int(round(len(all_subs) * a.val_subjects)))
            rng_val = np.random.default_rng(99)
            val_subs = set(rng_val.choice(all_subs, size=n_val, replace=False).tolist())
            split_per["val"] = {k: v for k, v in per.items() if k in val_subs}
            split_per["train"] = {k: v for k, v in per.items() if k not in val_subs}
            print(f"[AR] train subjects={len(split_per['train'])}  val subjects={len(split_per['val'])} ({100*a.val_subjects:.0f}% held out)")
    per = split_per["train"]
    val_per = split_per.get("val", {})
    split_counts = {k: len(v) for k, v in split_per.items()}
    # Keep only what the readout needs beyond train/val (e.g. drop test when evaluating on val).
    for name in list(split_per):
        if name not in ("train", "val", a.readout_fit_split, a.readout_eval_split):
            del split_per[name]
    gc.collect()

    all_ids, _sep = concat_corpus(per, m_vocab)
    # readout_pooled needs per-subject chronology; do not free `per` until after readout.
    keep_per_for_readout = bool(
        a.readout and a.hcp_metadata_csv and os.path.isfile(a.hcp_metadata_csv)
    )
    if not keep_per_for_readout:
        split_per = {}
    gc.collect()
    n_real = int(np.max(all_ids)) if len(all_ids) > 0 else 1

    # Determine context length: prefer explicit --chunk_len, else derive from n_context_windows.
    if a.chunk_len > 0:
        chunk_len = a.chunk_len
    else:
        chunk_len = max(1, a.n_context_windows) * seq_len_per_window
    window_trs = 32
    try:
        import yaml
        with open(meta["data_config"], "r", encoding="utf-8") as f:
            window_trs = int(yaml.safe_load(f)["data"].get("window_size", 32))
    except Exception:
        pass
    print(
        f"[AR] seq_len_per_window={seq_len_per_window}  "
        f"n_context_windows≈{chunk_len // max(1, seq_len_per_window)}  "
        f"chunk_len={chunk_len} tokens"
    )

    patch_size = max(1, a.patch_size)
    ds = ChunkDataset(all_ids, chunk_len, patch_size=patch_size)
    if len(ds) < 1:
        raise SystemExit("Empty chunk dataset; reduce chunk_len or n_context_windows?")
    n_patches_per_chunk = chunk_len // patch_size
    print(
        f"[AR] patch_size={patch_size}  n_patches/chunk={n_patches_per_chunk}  "
        f"dataset_size={len(ds)}"
    )
    dl = DataLoader(
        ds, batch_size=a.batch_size, shuffle=True, num_workers=0, collate_fn=_collate, drop_last=True
    )
    val_dl: Optional[DataLoader] = None
    if val_per:
        val_ids, _ = concat_corpus(val_per, m_vocab)
        val_ds = ChunkDataset(val_ids, chunk_len, patch_size=patch_size)
        if len(val_ds) > 0:
            val_dl = DataLoader(
                val_ds, batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn=_collate, drop_last=False
            )
            print(f"[AR] val dataset_size={len(val_ds)}")
        del val_ids
    model = CausalTfm(
        n_vocab=n_real,
        dim=a.hidden,
        nhead=a.heads,
        nlayers=a.layers,
        max_pos=n_patches_per_chunk + 64,
        patch_size=patch_size,
    ).to(a.device)
    # Token entropy (before training, reflects codec, not the model)
    t_entropy, t_frac, t_uniq = _token_entropy(all_ids, n_real)
    print(
        f"[{a.tag}] token entropy={t_entropy:.2f} bits  "
        f"vocab_used={t_uniq}/{n_real} ({100*t_frac:.1f}%)  "
        f"max_entropy={np.log2(max(1, n_real)):.2f} bits"
    )

    n_params = int(sum(p_.numel() for p_ in model.parameters()))
    print(f"[AR] n_params={n_params:,}  positions/sample={n_patches_per_chunk}  bits/id={math.log2(max(2, m_vocab)):.3f}")

    causality: Optional[Dict[str, Any]] = None
    if a.causality_test:
        causality = causality_test(model, next(iter(dl)), torch.device(a.device))
        if causality["result"] != "PASS":
            print("[causality_test] FAILED; aborting before training", flush=True)
            if a.output_dir:
                os.makedirs(a.output_dir, exist_ok=True)
                with open(os.path.join(a.output_dir, f"metrics_{a.tag}.json"), "w", encoding="utf-8") as f:
                    json.dump({"causality_test": causality, "args": vars(a)}, f, indent=2)
            raise SystemExit(2)

    if a.readout_only and a.output_dir:
        ckpt_path = os.path.join(a.output_dir, f"ar_{a.tag}.pth")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=a.device)
            model.load_state_dict(ckpt["state"])
            print(f"[readout_only] loaded checkpoint from {ckpt_path}")
            history = []
        else:
            raise SystemExit(f"--readout_only: no checkpoint at {ckpt_path}")
    else:
        history = _train_loop(
            model,
            dl,
            torch.device(a.device),
            a.epochs,
            a.lr,
            a.tag,
            warmup_steps=a.warmup_steps,
            lr_decay=a.lr_decay,
            log_every=a.log_every,
            val_dl=val_dl,
            use_amp=a.amp,
        )

    results: Dict[str, Any] = {
        "tag": a.tag,
        "model_type": meta.get("model_type", "?"),
        "token_dir": a.token_dir,
        "n_vocab": n_real,
        "seq_len_per_window": seq_len_per_window,
        "patch_size": patch_size,
        "chunk_len": chunk_len,
        "n_context_windows_approx": chunk_len // max(1, seq_len_per_window),
        "token_entropy_bits": t_entropy,
        "max_entropy_bits": float(np.log2(max(1, n_real))),
        "vocab_utilization": t_frac,
        "n_unique_tokens": t_uniq,
        "train_history": history,
        "final_ce": history[-1]["ce"] if history else None,
        "final_ppl": history[-1]["ppl"] if history else None,
        "final_val_ce": history[-1].get("val_ce") if history else None,
        "wall_time_per_epoch_s": [h["epoch_secs"] for h in history],
        "subject_split_json": a.subject_split_json if used_split_json else None,
        "n_subjects_per_split": split_counts,
        "n_train_subjects": split_counts.get("train", 0),
        "n_val_subjects": split_counts.get("val", 0),
        "n_params": n_params,
        "positions_per_sample": n_patches_per_chunk,
        "bits_per_id": float(math.log2(max(2, m_vocab))),
        "tokenizer_vocab_size": int(m_vocab),
        "context_windows": int(chunk_len // max(1, seq_len_per_window)),
        "context_trs": int(chunk_len // max(1, seq_len_per_window)) * window_trs,
        "window_trs": window_trs,
        "train_dataset_size": len(ds),
        "val_dataset_size": len(val_dl.dataset) if val_dl is not None else 0,
        "batch_size": a.batch_size,
        "causality_test": causality,
        "readout_fit_split": a.readout_fit_split if a.readout else None,
        "readout_eval_split": a.readout_eval_split if a.readout else None,
        "args": vars(a),
    }

    if a.output_dir and not a.readout_only:
        os.makedirs(a.output_dir, exist_ok=True)
        pth = os.path.join(a.output_dir, f"ar_{a.tag}.pth")
        torch.save({"state": model.state_dict(), "n_vocab": n_real, "chunk": chunk_len, "meta": meta}, pth)
        print(f"Saved {pth}")

    if a.readout and os.path.isfile(a.hcp_metadata_csv):
        fit_per = split_per.get(a.readout_fit_split) or {}
        eval_per = split_per.get(a.readout_eval_split) or {}
        if not fit_per or not eval_per:
            raise SystemExit(
                f"[readout] empty split: fit={a.readout_fit_split} ({len(fit_per)}) "
                f"eval={a.readout_eval_split} ({len(eval_per)})"
            )
        if a.readout_fit_split == a.readout_eval_split:
            raise SystemExit("[readout] fit and eval split must differ")
        print(f"[readout] fit split={a.readout_fit_split} ({len(fit_per)} subj)  "
              f"eval split={a.readout_eval_split} ({len(eval_per)} subj)")
        if rec_subj is None:
            with open(meta["index_path"], "r", encoding="utf-8") as f:
                idxrec = json.load(f)
            rec_subj = [infer_subject_id_from_record(r) for r in idxrec]
        ro = readout_pooled(
            fit_per,
            eval_per,
            n_real,
            model,
            rec_subj,
            a.hcp_metadata_csv,
            a.hcp_id_column,
            a.hcp_age_column,
            a.hcp_sex_column,
            a.device,
            seed=a.seed,
            n_boot=a.n_boot,
            cv_splits=a.cv_splits,
            cv_repeats=a.cv_repeats,
            predictions_npz=a.predictions_npz,
            use_amp=a.amp,
        )
        results.update(ro or {})
        split_per = {}
        gc.collect()
    elif a.readout:
        print(f"[readout] skip: no HCP CSV at {a.hcp_metadata_csv}")

    results["total_wall_time_s"] = time.perf_counter() - run_t_start
    if a.output_dir:
        metrics_path = os.path.join(a.output_dir, f"metrics_{a.tag}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {metrics_path}")



if __name__ == "__main__":
    main()
