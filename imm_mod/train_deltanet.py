#!/usr/bin/env python3
# train_delta.py
"""
DeltaNet for IMM-Mod STEPWISE dataset.

Dataset (from your new gen.py):
  src: "T|m|qk|mat1|...|matT"
  tgt: "v1|v2|...|vT"
    where v_t = (P_t)[qk] mod m, P_t = M1...Mt (mod m)

Tokens, teacher_force_state=False (default, L = 2 + T):
  [0] BOS
  [1] META   payload=(m,qk)
  then for t=1..T: MAT token carries M_t (flattened 3x3 residues)
  The model must derive P_t itself from the raw matrix stream via its own
  recurrent state -- matches the input-only tokenization used elsewhere.

Tokens, teacher_force_state=True (L = 2 + 2*T), leaky ablation only:
  [0] BOS
  [1] META   payload=(m,qk)
  then for t=1..T:
    STATE token carries P_{t-1} (flattened 3x3 residues)
    MAT   token carries M_t     (flattened 3x3 residues)
  STATE tokens carry the ground-truth previous state, so v_t=(P_{t-1}@M_t)[qk]
  becomes a single dot product of the given STATE and MAT tokens -- the model
  never needs to accumulate state across steps. Only use this for an explicit
  "given the true previous state, can the model do one correct step" ablation
  -- never for comparing task-solving ability against Mamba/Transformer.

Supervision:
  - at ALL MAT tokens (stepwise classification)

Optional aux loss:
  - aux_w * SmoothL1 to predict phi(P_t) at ALL MAT tokens, phi(x)=signed_log1p(x)
    (output-side target only, never fed back as input, so this adds no leak)
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# utils
# =========================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return torch.sign(x) * torch.log1p(torch.abs(x))


def infer_stats_from_src_path(src_path: str) -> Tuple[int, int]:
    max_T = 0
    max_m = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("|")
            if len(parts) < 4:
                continue
            T = int(parts[0])
            m = int(parts[1])
            max_T = max(max_T, T)
            max_m = max(max_m, m)
    return max_T, max_m


def infer_stats_from_dir(data_dir: str, splits: List[str]) -> Tuple[int, int]:
    max_T = 0
    max_m = 0
    found_any = False
    checked = []
    for sp in splits:
        src_path = os.path.join(data_dir, f"{sp}_src.txt")
        checked.append(src_path)
        if os.path.exists(src_path):
            found_any = True
            t, m = infer_stats_from_src_path(src_path)
            max_T = max(max_T, t)
            max_m = max(max_m, m)
    if not found_any:
        raise ValueError(
            f"Could not infer stats: no '*_src.txt' found in data_dir='{data_dir}' for splits={splits}.\n"
            f"Checked:\n  " + "\n  ".join(checked)
        )
    if max_T <= 0 or max_m <= 0:
        raise ValueError(f"Bad inferred stats: max_T={max_T}, max_m={max_m}")
    return max_T, max_m


# =========================
# parsing
# =========================
def parse_src_line(src_line: str, N: int = 3) -> Tuple[int, int, int, np.ndarray]:
    """
    src: "T|m|qk|mat1|...|matT"
    returns (T,m,qk,mats_raw) with mats_raw shape (T,9) int64 (raw ints from file)
    """
    parts = src_line.strip().split("|")
    if len(parts) < 4:
        raise ValueError(f"Bad src line (need >=4 fields): {src_line[:140]}")

    T = int(parts[0])
    m = int(parts[1])
    qk = int(parts[2])

    if T <= 0:
        raise ValueError(f"Bad T={T}")
    if m < 2:
        raise ValueError(f"Bad m={m}")
    if not (0 <= qk <= 8):
        raise ValueError(f"qk out of range: qk={qk}")

    mats_parts = parts[3:]
    if len(mats_parts) != T:
        raise ValueError(f"T mismatch: header T={T} but got {len(mats_parts)} matrices")

    D = N * N
    mats = np.empty((T, D), dtype=np.int64)
    for t in range(T):
        xs = mats_parts[t].split(",")
        if len(xs) != D:
            raise ValueError(f"Bad matrix len at t={t}: got {len(xs)} expected {D}")
        mats[t] = np.fromiter((int(v) for v in xs), dtype=np.int64, count=D)

    return T, m, qk, mats


def parse_tgt_line_stepwise(tgt_line: str, T: int, target_mode: str, m: int) -> np.ndarray:
    """
    tgt: "v1|...|vT" values are residues in [0..m-1]
    returns y_step: (T,) int64
      - multiclass: y_step[t]=v_{t+1}
      - binary0:    y_step[t]=1 iff v_{t+1}==0 else 0
    """
    parts = tgt_line.strip().split("|")
    if len(parts) != T:
        raise ValueError(f"Bad tgt length: got {len(parts)} but T={T}")
    v = np.fromiter((int(x) for x in parts), dtype=np.int64, count=T)
    if np.any(v < 0) or np.any(v >= m):
        bad = v[(v < 0) | (v >= m)][0]
        raise ValueError(f"Bad target residue {bad} for m={m}")

    if target_mode == "multiclass":
        return v.astype(np.int64, copy=False)
    else:
        return (v == 0).astype(np.int64, copy=False)


# =========================
# mod math for states
# =========================
def matmul3_mod(A: np.ndarray, B: np.ndarray, m: int) -> np.ndarray:
    return ((A % m) @ (B % m)) % m


def compute_states_prev_and_next_mod(mats_raw: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    mats_raw: (T,9) raw ints (pm1)
    returns prev[t]=P_{t-1}, nxt[t]=P_t (flattened residues) for t=0..T-1
    """
    T = mats_raw.shape[0]
    mats3 = mats_raw.reshape(T, 3, 3).astype(np.int64, copy=False)

    P = np.eye(3, dtype=np.int64) % m
    prev = np.empty((T, 9), dtype=np.int64)
    nxt = np.empty((T, 9), dtype=np.int64)

    for t in range(T):
        prev[t] = (P.reshape(-1) % m)
        P = matmul3_mod(P, mats3[t], m)
        nxt[t] = (P.reshape(-1) % m)

    return prev, nxt


# =========================
# token types
# =========================
TOK_PAD = 0
TOK_BOS = 1
TOK_META = 2
TOK_STATE = 3
TOK_MAT = 4
NUM_TOKEN_TYPES = 5


# =========================
# dataset (precompute tokens, optional cache)
# =========================
class PrecomputedIMMModStepwiseDataset(Dataset):
    """
    Per-sample variable-length arrays:
      tok_type_i: (L,) uint8
      tok_val_i : (L,9) int64 (residues for STATE/MAT; META uses first slots (m,qk))
      y_tgt_i   : (L,) int64 (labels at MAT positions, 0 elsewhere)
      y_mask_i  : (L,) bool (True at MAT positions)
      aux_tgt_i : (L,9) int64 (P_t residues at MAT positions)
      aux_mask_i: (L,) bool (True at MAT positions)
      plus m_i, L_i
    """

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        alphabet: str,
        target_mode: str,
        m_max: int,
        cache_path: Optional[str] = None,
        teacher_force_state: bool = False,
    ):
        self.alphabet = alphabet
        self.target_mode = target_mode
        self.m_max = int(m_max)
        self.teacher_force_state = bool(teacher_force_state)

        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            meta = dict(data["meta"].item())
            ok = (
                meta.get("alphabet") == alphabet
                and meta.get("target_mode") == target_mode
                and int(meta.get("m_max")) == int(m_max)
                and bool(meta.get("teacher_force_state", False)) == self.teacher_force_state
            )
            if ok:
                self.m_i = data["m_i"].astype(np.int64, copy=False)
                self.lengths = data["lengths"].astype(np.int64, copy=False)
                self.tok_type = list(data["tok_type"])
                self.tok_val = list(data["tok_val"])
                self.y_tgt = list(data["y_tgt"])
                self.y_mask = list(data["y_mask"])
                self.aux_tgt = list(data["aux_tgt"])
                self.aux_mask = list(data["aux_mask"])
                return
            else:
                print(f"[Cache] ignoring incompatible cache: {cache_path}")

        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = [ln.strip() for ln in f if ln.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [ln.strip() for ln in f if ln.strip()]
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src/tgt mismatch: {len(src_lines)} vs {len(tgt_lines)}")

        N = len(src_lines)
        self.m_i = np.empty((N,), dtype=np.int64)
        self.lengths = np.empty((N,), dtype=np.int64)

        self.tok_type: List[np.ndarray] = [None] * N  # type: ignore[list-item]
        self.tok_val: List[np.ndarray] = [None] * N   # type: ignore[list-item]
        self.y_tgt: List[np.ndarray] = [None] * N     # type: ignore[list-item]
        self.y_mask: List[np.ndarray] = [None] * N    # type: ignore[list-item]
        self.aux_tgt: List[np.ndarray] = [None] * N   # type: ignore[list-item]
        self.aux_mask: List[np.ndarray] = [None] * N  # type: ignore[list-item]

        it = list(zip(src_lines, tgt_lines))
        for i, (src, tgt) in enumerate(tqdm(it, desc=f"Precompute {os.path.basename(src_path)}", dynamic_ncols=True)):
            T, m, qk, mats_raw = parse_src_line(src, N=3)

            if m > self.m_max:
                raise ValueError(f"Found m={m} > m_max={self.m_max}. Increase --m_max.")

            if alphabet == "pm1":
                ok = np.all((mats_raw == -1) | (mats_raw == 0) | (mats_raw == 1))
                if not ok:
                    raise ValueError(f"Alphabet mismatch (pm1) at line {i}")
            elif alphabet == "any":
                pass
            else:
                raise ValueError(f"Unknown alphabet: {alphabet} (use pm1 or any)")

            y_step = parse_tgt_line_stepwise(tgt, T=T, target_mode=target_mode, m=m)  # (T,)

            mats_mod = np.remainder(mats_raw, m).astype(np.int64, copy=False)  # (T,9) residues
            # Always computed: states_next feeds the (output-only) aux loss in both modes,
            # states_prev additionally feeds the STATE token but only when teacher_force_state.
            states_prev, states_next = compute_states_prev_and_next_mod(mats_raw, m=m)  # (T,9) residues

            L = (2 + 2 * T) if self.teacher_force_state else (2 + T)
            tok_type = np.full((L,), TOK_PAD, dtype=np.uint8)
            tok_val = np.zeros((L, 9), dtype=np.int64)
            y_tgt = np.zeros((L,), dtype=np.int64)
            y_mask = np.zeros((L,), dtype=np.bool_)
            aux_tgt = np.zeros((L, 9), dtype=np.int64)
            aux_mask = np.zeros((L,), dtype=np.bool_)

            tok_type[0] = TOK_BOS
            tok_type[1] = TOK_META
            tok_val[1, 0] = m
            tok_val[1, 1] = qk

            pos = 2
            for t in range(1, T + 1):
                if self.teacher_force_state:
                    # STATE: P_{t-1} -- leaky ablation only, see module docstring
                    tok_type[pos] = TOK_STATE
                    tok_val[pos, :] = states_prev[t - 1]
                    pos += 1

                # MAT: M_t (residues)
                tok_type[pos] = TOK_MAT
                tok_val[pos, :] = mats_mod[t - 1]

                # stepwise supervision at ALL MAT tokens
                y_mask[pos] = True
                y_tgt[pos] = int(y_step[t - 1])

                # aux: predict P_t at MAT tokens (output-only target, never fed back as input)
                aux_mask[pos] = True
                aux_tgt[pos, :] = states_next[t - 1]

                pos += 1

            self.m_i[i] = m
            self.lengths[i] = L
            self.tok_type[i] = tok_type
            self.tok_val[i] = tok_val
            self.y_tgt[i] = y_tgt
            self.y_mask[i] = y_mask
            self.aux_tgt[i] = aux_tgt
            self.aux_mask[i] = aux_mask

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            meta = {"alphabet": alphabet, "target_mode": target_mode, "m_max": int(m_max),
                    "teacher_force_state": self.teacher_force_state}
            np.savez_compressed(
                cache_path,
                meta=np.array(meta, dtype=object),
                m_i=self.m_i,
                lengths=self.lengths,
                tok_type=np.array(self.tok_type, dtype=object),
                tok_val=np.array(self.tok_val, dtype=object),
                y_tgt=np.array(self.y_tgt, dtype=object),
                y_mask=np.array(self.y_mask, dtype=object),
                aux_tgt=np.array(self.aux_tgt, dtype=object),
                aux_mask=np.array(self.aux_mask, dtype=object),
            )
            print(f"[Cache] wrote: {cache_path}")

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "tok_type": self.tok_type[idx],
            "tok_val": self.tok_val[idx],
            "y_tgt": self.y_tgt[idx],
            "y_mask": self.y_mask[idx],
            "aux_tgt": self.aux_tgt[idx],
            "aux_mask": self.aux_mask[idx],
            "L": int(self.lengths[idx]),
            "m": int(self.m_i[idx]),
        }


# =========================
# collate (fast padding)
# =========================
@dataclass
class Batch:
    tok_type: torch.Tensor   # (B,Lmax) long
    tok_val: torch.Tensor    # (B,Lmax,9) long
    attn01: torch.Tensor     # (B,Lmax) int32
    lengths: torch.Tensor    # (B,) long
    m_i: torch.Tensor        # (B,) long
    y_tgt: torch.Tensor      # (B,Lmax) long
    y_mask: torch.Tensor     # (B,Lmax) bool
    aux_tgt: torch.Tensor    # (B,Lmax,9) long
    aux_mask: torch.Tensor   # (B,Lmax) bool


def collate_batch(items: List[dict]) -> Batch:
    B = len(items)
    lengths = torch.tensor([it["L"] for it in items], dtype=torch.long)
    L_max = int(lengths.max().item())

    tok_type = torch.full((B, L_max), TOK_PAD, dtype=torch.long)
    tok_val = torch.zeros((B, L_max, 9), dtype=torch.long)
    attn01 = torch.zeros((B, L_max), dtype=torch.int32)

    m_i = torch.tensor([it["m"] for it in items], dtype=torch.long)

    y_tgt = torch.zeros((B, L_max), dtype=torch.long)
    y_mask = torch.zeros((B, L_max), dtype=torch.bool)

    aux_tgt = torch.zeros((B, L_max, 9), dtype=torch.long)
    aux_mask = torch.zeros((B, L_max), dtype=torch.bool)

    for b, it in enumerate(items):
        L = int(it["L"])
        attn01[b, :L] = 1
        tok_type[b, :L] = torch.from_numpy(it["tok_type"].astype(np.int64, copy=False))
        tok_val[b, :L, :] = torch.from_numpy(it["tok_val"].astype(np.int64, copy=False))
        y_tgt[b, :L] = torch.from_numpy(it["y_tgt"].astype(np.int64, copy=False))
        y_mask[b, :L] = torch.from_numpy(it["y_mask"])
        aux_tgt[b, :L, :] = torch.from_numpy(it["aux_tgt"].astype(np.int64, copy=False))
        aux_mask[b, :L] = torch.from_numpy(it["aux_mask"])

    return Batch(tok_type, tok_val, attn01, lengths, m_i, y_tgt, y_mask, aux_tgt, aux_mask)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


# =========================
# encoders
# =========================
class Residue9Encoder(nn.Module):
    def __init__(self, d_model: int, m_max: int):
        super().__init__()
        self.val_emb = nn.Embedding(m_max, d_model)
        self.pos_emb = nn.Embedding(9, d_model)
        self.proj = nn.Linear(9 * d_model, d_model)
        self.register_buffer("pos_idx", torch.arange(9, dtype=torch.long), persistent=False)

    def forward(self, v9: torch.Tensor) -> torch.Tensor:
        v = self.val_emb(v9)                                    # (B,L,9,d)
        p = self.pos_emb(self.pos_idx).view(1, 1, 9, -1)        # (1,1,9,d)
        z = (v + p).reshape(v9.shape[0], v9.shape[1], -1)
        return self.proj(z)                                     # (B,L,d)


class MetaEncoder(nn.Module):
    def __init__(self, d_model: int, m_max: int):
        super().__init__()
        self.m_emb = nn.Embedding(m_max + 1, d_model)
        self.qk_emb = nn.Embedding(9, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, meta_payload: torch.Tensor) -> torch.Tensor:
        m = meta_payload[:, 0].clamp(min=0)
        qk = meta_payload[:, 1].clamp(min=0)
        return self.ln(self.m_emb(m) + self.qk_emb(qk))


class BaseTokenEncoder(nn.Module):
    # No sequence-level positional embedding: DeltaNet's recurrent state update
    # already encodes token order, and a learned nn.Embedding(max_len, ...)
    # would leave rows beyond the training length untrained, silently
    # corrupting bin1/bin2 length-generalization eval (those rows never see a
    # gradient update). Residue9Encoder's own pos_emb (size 9) is a different,
    # legitimate thing -- it encodes which of the 9 matrix entries a value is,
    # not a sequence position.
    def __init__(self, m_max: int, max_len: int, d_model: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.res9_enc = Residue9Encoder(d_model=d_model, m_max=m_max)
        self.meta_enc = MetaEncoder(d_model=d_model, m_max=m_max)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, tok_val: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len}")

        x = self.type_emb(tok_type)

        sm_pos = (tok_type == TOK_STATE) | (tok_type == TOK_MAT)          # (B,L) bool
        tok_val_safe = tok_val.clone()
        tok_val_safe[~sm_pos] = 0
        x = x + self.res9_enc(tok_val_safe) * sm_pos.unsqueeze(-1).to(x.dtype)

        meta_pos = (tok_type == TOK_META)
        if meta_pos.any():
            meta_idx = meta_pos.to(torch.int64).argmax(dim=1)
            meta_payload = tok_val[torch.arange(B, device=tok_type.device), meta_idx, :]  # (B,9)
            x[torch.arange(B, device=tok_type.device), meta_idx, :] += self.meta_enc(meta_payload)

        return self.drop(x)


# =========================
# model: DeltaNet
# =========================
def build_fla_deltanet_layer(d_model: int, num_heads: int, mode: str, layer_idx: int, allow_neg_eigval: bool):
    from fla.layers import DeltaNet
    return DeltaNet(
        mode=mode,  # training should use 'chunk'
        d_model=d_model,
        num_heads=num_heads,
        layer_idx=layer_idx,
        allow_neg_eigval=allow_neg_eigval,
    )


class ModelDeltaNet(nn.Module):
    def __init__(
        self,
        m_max: int,
        max_len: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
        deltanet_mode: str,
        allow_neg_eigval: bool,
        target_mode: str,
    ):
        super().__init__()
        self.m_max = m_max
        self.target_mode = target_mode

        self.enc = BaseTokenEncoder(m_max=m_max, max_len=max_len, d_model=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            build_fla_deltanet_layer(d_model, heads, deltanet_mode, i, allow_neg_eigval)
            for i in range(layers)
        ])
        self.ln = nn.LayerNorm(d_model)

        if target_mode == "multiclass":
            self.cls_head = nn.Linear(d_model, m_max)
        elif target_mode == "binary0":
            self.cls_head = nn.Linear(d_model, 1)
        else:
            raise ValueError("target_mode must be multiclass or binary0")

        self.aux_head = nn.Linear(d_model, 9)

    def forward(self, tok_type, tok_val, attn01):
        x = self.enc(tok_type, tok_val)
        for blk in self.blocks:
            x, _, _ = blk(x, attention_mask=attn01, past_key_values=None, use_cache=False)
        x = self.ln(x)
        return self.cls_head(x), self.aux_head(x)


# =========================
# loss / metrics (STEPWISE)
# =========================
def mask_logits_by_m(logits: torch.Tensor, m_i: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,L,m_max)
    masks classes >= m_i (per sample) to -1e9
    """
    B, L, m_max = logits.shape
    ar = torch.arange(m_max, device=logits.device).view(1, 1, m_max)
    mi = m_i.view(B, 1, 1)
    return logits.masked_fill(ar >= mi, -1e9)


def loss_stepwise_multiclass(
    logits: torch.Tensor,      # (B,L,m_max)
    y_tgt: torch.Tensor,       # (B,L)
    y_mask: torch.Tensor,      # (B,L) MAT positions
    m_i: torch.Tensor,         # (B,)
) -> torch.Tensor:
    idx = y_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return logits.new_tensor(0.0)
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]

    picked = logits[b_ix, l_ix, :]                 # (Npos, m_max)
    y = y_tgt[b_ix, l_ix]                          # (Npos,)
    m_pos = m_i[b_ix]                              # (Npos,)

    ar = torch.arange(picked.shape[-1], device=logits.device).view(1, -1).expand(picked.size(0), -1)
    picked = picked.masked_fill(ar >= m_pos.view(-1, 1), -1e9)

    return F.cross_entropy(picked, y)


def loss_stepwise_binary(
    logits: torch.Tensor,      # (B,L,1) or (B,L)
    y_tgt: torch.Tensor,       # (B,L)
    y_mask: torch.Tensor,      # (B,L)
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    idx = y_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return logits.new_tensor(0.0)
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]
    picked = logits[b_ix, l_ix]
    y = y_tgt[b_ix, l_ix].float()
    return F.binary_cross_entropy_with_logits(picked, y, reduction="mean")


def aux_state_loss_smoothl1(aux_pred_phi: torch.Tensor, aux_tgt: torch.Tensor, aux_mask: torch.Tensor) -> torch.Tensor:
    if aux_mask.sum().item() == 0:
        return aux_pred_phi.new_tensor(0.0)
    tgt_phi = signed_log1p(aux_tgt)
    m = aux_mask.unsqueeze(-1)
    pred = aux_pred_phi[m.expand_as(aux_pred_phi)].view(-1, 9)
    tgt = tgt_phi[m.expand_as(tgt_phi)].view(-1, 9)
    return F.smooth_l1_loss(pred, tgt, reduction="mean")


@torch.no_grad()
def metric_stepwise_acc(
    cls_logits: torch.Tensor,
    y_tgt: torch.Tensor,
    y_mask: torch.Tensor,
    target_mode: str,
    m_i: torch.Tensor,
) -> float:
    idx = y_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return 0.0
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]

    if target_mode == "multiclass":
        picked = cls_logits[b_ix, l_ix, :]                      # (Npos,m_max)
        y = y_tgt[b_ix, l_ix]                                   # (Npos,)
        m_pos = m_i[b_ix]                                       # (Npos,)

        ar = torch.arange(picked.shape[-1], device=cls_logits.device).view(1, -1).expand(picked.size(0), -1)
        picked = picked.masked_fill(ar >= m_pos.view(-1, 1), -1e9)

        pred = picked.argmax(dim=-1)
        return float((pred == y).float().mean().item())
    else:
        picked = cls_logits[b_ix, l_ix, 0] if cls_logits.dim() == 3 else cls_logits[b_ix, l_ix]
        pred = (torch.sigmoid(picked) >= 0.5).to(torch.long)
        y = y_tgt[b_ix, l_ix].to(torch.long)
        return float((pred == y).float().mean().item())


# =========================
# train / eval
# =========================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    amp: bool,
    amp_dtype: str,
    grad_clip: float,
    target_mode: str,
    aux_w: float,
    max_steps: int = 0,
    global_step: int = 0,
) -> Tuple[float, float, float, int, bool]:
    train = optimizer is not None
    model.train(train)

    use_amp = amp and (device.type == "cuda")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_fp16 = use_amp and (amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_loss = total_cls = total_acc = 0.0
    steps = 0
    hit_limit = False

    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for batch in pbar:
        if train and max_steps > 0 and global_step >= max_steps:
            hit_limit = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        tok_val = batch.tok_val.to(device, non_blocking=True)
        attn01 = batch.attn01.to(device, non_blocking=True)

        m_i = batch.m_i.to(device, non_blocking=True)
        y_tgt = batch.y_tgt.to(device, non_blocking=True)
        y_mask = batch.y_mask.to(device, non_blocking=True)

        aux_tgt = batch.aux_tgt.to(device, non_blocking=True)
        aux_mask = batch.aux_mask.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            cls_logits, aux_pred = model(tok_type, tok_val, attn01)

            if target_mode == "multiclass":
                cls_logits = mask_logits_by_m(cls_logits, m_i)
                cls_loss = loss_stepwise_multiclass(cls_logits, y_tgt, y_mask, m_i)
            else:
                cls_loss = loss_stepwise_binary(cls_logits, y_tgt, y_mask)

            if aux_w > 0:
                aux_loss = aux_state_loss_smoothl1(aux_pred, aux_tgt, aux_mask)
                loss = cls_loss + float(aux_w) * aux_loss
            else:
                loss = cls_loss

        if train:
            if use_fp16:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                hit_limit = True

        acc = metric_stepwise_acc(cls_logits, y_tgt, y_mask, target_mode=target_mode, m_i=m_i)

        steps += 1
        total_loss += float(loss.item())
        total_cls += float(cls_loss.item())
        total_acc += float(acc)

        pbar.set_postfix(loss=total_loss / steps, cls=total_cls / steps, acc=100 * total_acc / steps, gstep=global_step)

        if hit_limit:
            break

    denom = max(1, steps)
    return (total_loss / denom, total_cls / denom, total_acc / denom, global_step, hit_limit)


# =========================
# main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--alphabet", type=str, default="pm1", choices=["pm1", "any"])
    ap.add_argument("--target_mode", type=str, default="multiclass", choices=["multiclass", "binary0"])
    ap.add_argument("--splits", type=str, default="train,val_bin0,test_bin0,test_bin1,test_bin2")

    ap.add_argument("--m_max", type=int, default=0, help="0=infer max m from dataset")
    ap.add_argument("--max_len", type=int, default=0, help="0=infer (2+max_T, or 2+2*max_T if --teacher_force_state)")
    ap.add_argument("--teacher_force_state", action="store_true",
        help="Feed the ground-truth previous state P_{t-1} as an extra STATE token before "
             "each M_t (train AND eval). This leaks the answer -- v_t becomes a single dot "
             "product of the STATE and MAT tokens -- so the model never needs to accumulate "
             "state across steps. OFF by default (input-only tokenization). Only enable for "
             "an explicit single-step-given-true-state ablation.")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--deltanet_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--allow_neg_eigval", action="store_true")

    ap.add_argument("--aux_w", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--save_path", type=str, default="ckpt_delta_stepwise.pt")
    ap.add_argument("--early_stop", type=str, default="acc", choices=["loss", "acc"])
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--eval_log", type=str, default="delta_stepwise_eval.log")

    ap.add_argument("--cache_dir", type=str, default="", help="cache precomputed split tensors as .npz here")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.deltanet_mode != "chunk":
        print("[Warn] DeltaNet training should use chunk. Forcing.")
        args.deltanet_mode = "chunk"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    inferred_max_T, inferred_max_m = infer_stats_from_dir(args.data_dir, split_list)
    if args.m_max <= 0:
        args.m_max = inferred_max_m
        print(f"[Auto] inferred m_max(all splits)={args.m_max}")
    if args.max_len <= 0:
        args.max_len = (2 + 2 * inferred_max_T) if args.teacher_force_state else (2 + inferred_max_T)
        print(f"[Auto] inferred max_T(all splits)={inferred_max_T} => max_len={args.max_len}")

    max_T = (args.max_len - 2) // 2 if args.teacher_force_state else (args.max_len - 2)

    tf_desc = "teacher forcing STATE tokens (leaks P_{t-1}, train+eval)" if args.teacher_force_state else "input-only (no teacher forcing)"
    print(f"[Device] {device}")
    print(f"[Task] Stepwise: predict v_t=(P_t)[qk] in Z_m at ALL MAT tokens; target_mode={args.target_mode} | {tf_desc}")
    print(f"[Data] {args.data_dir} alphabet={args.alphabet} m_max={args.m_max} max_T~{max_T}")
    print(f"[Model] DeltaNet max_len={args.max_len} d_model={args.d_model} heads={args.heads} layers={args.layers} dropout={args.dropout}")
    print(f"[Aux ] aux_w={args.aux_w}")
    if args.max_steps > 0:
        print(f"[Train] hard cap max_steps={args.max_steps}")

    def split_paths(split: str) -> Tuple[str, str]:
        return (
            os.path.join(args.data_dir, f"{split}_src.txt"),
            os.path.join(args.data_dir, f"{split}_tgt.txt"),
        )

    def cache_path_for(split: str) -> Optional[str]:
        if not args.cache_dir:
            return None
        os.makedirs(args.cache_dir, exist_ok=True)
        return os.path.join(args.cache_dir, f"{split}.npz")

    train_src, train_tgt = split_paths("train")
    val_src, val_tgt = split_paths("val_bin0")
    t0_src, t0_tgt = split_paths("test_bin0")
    t1_src, t1_tgt = split_paths("test_bin1")
    t2_src, t2_tgt = split_paths("test_bin2")

    train_ds = PrecomputedIMMModStepwiseDataset(train_src, train_tgt, args.alphabet, args.target_mode, args.m_max, cache_path_for("train"), teacher_force_state=args.teacher_force_state)
    val_ds   = PrecomputedIMMModStepwiseDataset(val_src,   val_tgt,   args.alphabet, args.target_mode, args.m_max, cache_path_for("val_bin0"), teacher_force_state=args.teacher_force_state)
    test0_ds = PrecomputedIMMModStepwiseDataset(t0_src,    t0_tgt,    args.alphabet, args.target_mode, args.m_max, cache_path_for("test_bin0"), teacher_force_state=args.teacher_force_state)
    test1_ds = PrecomputedIMMModStepwiseDataset(t1_src,    t1_tgt,    args.alphabet, args.target_mode, args.m_max, cache_path_for("test_bin1"), teacher_force_state=args.teacher_force_state)
    test2_ds = PrecomputedIMMModStepwiseDataset(t2_src,    t2_tgt,    args.alphabet, args.target_mode, args.m_max, cache_path_for("test_bin2"), teacher_force_state=args.teacher_force_state)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)
    test0_loader = make_loader(test0_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test1_loader = make_loader(test1_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test2_loader = make_loader(test2_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ModelDeltaNet(
        m_max=args.m_max,
        max_len=args.max_len,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        deltanet_mode=args.deltanet_mode,
        allow_neg_eigval=args.allow_neg_eigval,
        target_mode=args.target_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Params] {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop == "loss":
        best_val = float("inf")
        better = lambda cur: cur < best_val - 1e-6
    else:
        best_val = -1.0
        better = lambda cur: cur > best_val + 1e-12

    bad = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cls, tr_acc, global_step, hit = run_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=args.grad_clip,
            target_mode=args.target_mode, aux_w=args.aux_w,
            max_steps=args.max_steps, global_step=global_step
        )
        va_loss, va_cls, va_acc, _, _ = run_epoch(
            model, val_loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=0.0,
            target_mode=args.target_mode, aux_w=args.aux_w,
        )

        cur = va_loss if args.early_stop == "loss" else va_acc
        improved = better(cur)
        if improved:
            best_val = cur
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "global_step": global_step}, args.save_path)
        else:
            bad += 1

        print(
            f"Epoch {epoch:03d} | step={global_step:06d} | "
            f"train loss={tr_loss:.4f} cls={tr_cls:.4f} acc={tr_acc*100:.2f}% | "
            f"val   loss={va_loss:.4f} cls={va_cls:.4f} acc={va_acc*100:.2f}% | "
            f"best({args.early_stop})={best_val:.6f} bad={bad}/{args.patience}"
            f"{' [saved]' if improved else ''}"
        )

        if hit:
            print(f"Reached max_steps={args.max_steps}. Stopping.")
            break
        if bad >= args.patience:
            print("Early stopping (patience).")
            break

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\n[Eval best checkpoint]")
    for name, loader in [("test_bin0", test0_loader), ("test_bin1", test1_loader), ("test_bin2", test2_loader)]:
        te_loss, te_cls, te_acc, _, _ = run_epoch(
            model, loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=0.0,
            target_mode=args.target_mode, aux_w=args.aux_w,
        )
        eval_str = f"{name:9s} | loss={te_loss:.4f} cls={te_cls:.4f} acc={te_acc*100:.2f}%"
        print(eval_str)
        with open(args.eval_log, "a", encoding="utf-8") as logf:
            logf.write(args.data_dir + " " + args.save_path + " " + eval_str + "\n")

    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
