#!/usr/bin/env python3
# train_rwkv7_query_stepwise.py
"""
RWKV-7 (FLA) for MOD prefix-query dataset with STEPWISE targets (NEW format).

Dataset:
  src: "T|m|qk|mat1|...|matT"
  tgt: "v1|v2|...|vT"
    where v_t = (P_t).flat[qk] mod m,  P_t = M1...Mt (mod m)

Tokens, teacher_force_state=False (default, L = 2 + T):
  [BOS], [META], then for t=1..T: [MAT] carries M_t flattened (9 residues)
  The model must derive P_t itself from the raw matrix stream via its own
  recurrent state -- matches the input-only tokenization used elsewhere.

Tokens, teacher_force_state=True (L = 2 + 2*T), leaky ablation only:
  [BOS], [META], then for t=1..T:
    [STATE] carries P_{t-1} flattened (9 residues)
    [MAT]   carries M_t     flattened (9 residues)
  STATE tokens carry the ground-truth previous state, so v_t=(P_{t-1}@M_t)[qk]
  becomes a single dot product of the given STATE and MAT tokens -- the model
  never needs to accumulate state across steps. Only use this for an explicit
  "given the true previous state, can the model do one correct step" ablation
  -- never for comparing task-solving ability against Mamba/Transformer.

Supervision:
  - Cross-entropy at every MAT token (stepwise).
  - We mask unused classes >= m_i per-sample.

Notes:
  - attention_mask passed to FLA RWKV7Attention is ALWAYS bool (B,L), contiguous, on device.
  - torch.lerp guard + optional fused_addcmul disable (env flags).
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


# =============================================================================
# FLA / RWKV7 stability patches (DROP-IN)
# =============================================================================

# ---- 0) hard disable torch.compile BEFORE any FLA import ----
if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1":
    if hasattr(torch, "compile"):
        def _no_compile(fn=None, *args, **kwargs):
            return fn if fn is not None else (lambda f: f)
        torch.compile = _no_compile  # type: ignore[attr-defined]
        print("[patch] torch.compile disabled (identity)")

# ---- 1) patch torch.lerp to guard end=None + dtype/device mismatch ----
if os.environ.get("PATCH_TORCH_LERP_DTYPE", "1") == "1":
    _orig_lerp = torch.lerp

    def _lerp_guard(input: torch.Tensor, end, weight, *args, **kwargs):
        if end is None:
            end = input
        if isinstance(end, torch.Tensor):
            if end.dtype != input.dtype or end.device != input.device:
                end = end.to(dtype=input.dtype, device=input.device)
        if isinstance(weight, torch.Tensor):
            if weight.dtype != input.dtype or weight.device != input.device:
                weight = weight.to(dtype=input.dtype, device=input.device)
        return _orig_lerp(input, end, weight, *args, **kwargs)

    torch.lerp = _lerp_guard  # type: ignore[assignment]
    print("[patch] torch.lerp guard enabled (end=None + dtype/device)")

# ---- 2) optionally disable RWKV7 fused_addcmul kernel ----
def patch_disable_fla_rwkv7_fused_addcmul() -> None:
    if os.environ.get("DISABLE_RWKV7_FUSED_ADDCMUL", "0") != "1":
        return
    try:
        import fla.ops.rwkv7.fused_addcmul as fac
        import fla.layers.rwkv7 as rwkv7_layer
    except Exception as e:
        print(f"[patch] disable fused_addcmul: import failed: {e}")
        return

    def fused_addcmul_fallback(hidden_states, delta, xr, xw, xk, xv, xa, xg):
        return (
            torch.addcmul(hidden_states, delta, xr),
            torch.addcmul(hidden_states, delta, xw),
            torch.addcmul(hidden_states, delta, xk),
            torch.addcmul(hidden_states, delta, xv),
            torch.addcmul(hidden_states, delta, xa),
            torch.addcmul(hidden_states, delta, xg),
        )

    fac.fused_addcmul_rwkv7 = fused_addcmul_fallback
    rwkv7_layer.fused_addcmul_rwkv7 = fused_addcmul_fallback
    print("[patch] DISABLE_RWKV7_FUSED_ADDCMUL=1 -> torch.addcmul fallback")

patch_disable_fla_rwkv7_fused_addcmul()


# =========================
# utils
# =========================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_autocast_dtype(device: torch.device, amp_dtype: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


def parse_src_line_stepwise_query(src_line: str, N: int = 3) -> Tuple[int, int, int, np.ndarray]:
    """
    src: "T|m|qk|mat1|...|matT"
    returns (T, m, qk, mats_raw[T,9])
    """
    parts = src_line.strip().split("|")
    if len(parts) < 4:
        raise ValueError(f"Bad src line (need >=4 fields): {src_line[:160]}")

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


def parse_tgt_steps(tgt_line: str, T: int) -> np.ndarray:
    blocks = [z for z in tgt_line.strip().split("|") if z != ""]
    if len(blocks) != T:
        raise ValueError(f"Bad tgt step count: got {len(blocks)} expected T={T}")
    y = np.fromiter((int(v) for v in blocks), dtype=np.int64, count=T)
    return y


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
    found = False
    for sp in splits:
        src_path = os.path.join(data_dir, f"{sp}_src.txt")
        if os.path.exists(src_path):
            found = True
            t, m = infer_stats_from_src_path(src_path)
            max_T = max(max_T, t)
            max_m = max(max_m, m)
    if not found:
        raise ValueError(f"No '*_src.txt' found in {data_dir}")
    if max_T <= 0 or max_m <= 0:
        raise ValueError(f"Bad inferred stats: max_T={max_T}, max_m={max_m}")
    return max_T, max_m


def matmul3_mod(A: np.ndarray, B: np.ndarray, m: int) -> np.ndarray:
    return ((A % m) @ (B % m)) % m


def compute_prev_states_mod(mats_raw: np.ndarray, m: int) -> np.ndarray:
    """
    mats_raw: (T,9) raw ints
    returns prev[t] = P_{t-1} flattened (0..m-1), shape (T,9)
    """
    T = mats_raw.shape[0]
    mats3 = mats_raw.reshape(T, 3, 3).astype(np.int64, copy=False)

    P = np.eye(3, dtype=np.int64) % m
    prev = np.empty((T, 9), dtype=np.int64)

    for t in range(T):
        prev[t] = P.reshape(-1)
        P = matmul3_mod(P, mats3[t], m)

    return prev


# =========================
# dataset
# =========================
class PreloadedModQueryStepwiseDataset(Dataset):
    """
    Loads:
      src: T|m|qk|mat...
      tgt: v1|...|vT
    Stores per sample:
      T, m, qk
      mats_mod: (T,9) residues in [0..m-1]
      states_prev: (T,9) residues for P_{t-1}
      y_steps: (T,) in [0..m-1]
    """
    def __init__(self, src_path: str, tgt_path: str, alphabet: str, m_max: int, quiet: bool = False):
        self.alphabet = alphabet
        self.m_max = m_max

        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = [ln.strip() for ln in f if ln.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [ln.strip() for ln in f if ln.strip()]
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src/tgt mismatch: {len(src_lines)} vs {len(tgt_lines)}")

        self.Ts: List[int] = []
        self.ms: List[int] = []
        self.qks: List[int] = []
        self.mats_mod: List[torch.Tensor] = []
        self.prev: List[torch.Tensor] = []
        self.y_steps: List[torch.Tensor] = []

        it = list(zip(src_lines, tgt_lines))
        if not quiet:
            it = tqdm(it, desc=f"Preload {os.path.basename(src_path)}", dynamic_ncols=True)

        for i, (src, tgt) in enumerate(it):
            T, m, qk, mats_raw = parse_src_line_stepwise_query(src, N=3)

            if not (2 <= m <= m_max):
                raise ValueError(f"m={m} out of range [2..m_max={m_max}] at line {i}")

            if alphabet == "pm1":
                ok = np.all((mats_raw == -1) | (mats_raw == 0) | (mats_raw == 1))
            elif alphabet == "01":
                ok = np.all((mats_raw == 0) | (mats_raw == 1))
            elif alphabet == "any":
                ok = True
            else:
                raise ValueError(f"Unknown alphabet: {alphabet}")
            if not ok:
                raise ValueError(f"Alphabet mismatch in src at line {i}")

            mats_mod = np.remainder(mats_raw, m).astype(np.int64, copy=False)  # (T,9)
            prev = compute_prev_states_mod(mats_raw, m=m)                       # (T,9)

            y = parse_tgt_steps(tgt, T=T)
            y = np.remainder(y, m).astype(np.int64, copy=False)                 # (T,)

            self.Ts.append(T)
            self.ms.append(m)
            self.qks.append(qk)

            self.mats_mod.append(torch.from_numpy(mats_mod).to(torch.int16))    # (T,9)
            self.prev.append(torch.from_numpy(prev).to(torch.int16))            # (T,9)
            self.y_steps.append(torch.from_numpy(y).to(torch.int16))            # (T,)

    def __len__(self) -> int:
        return len(self.Ts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "T": self.Ts[idx],
            "m": self.ms[idx],
            "qk": self.qks[idx],
            "mats_mod": self.mats_mod[idx],
            "prev": self.prev[idx],
            "y_steps": self.y_steps[idx],
        }


# =========================
# collate
# =========================
TOK_PAD = 0
TOK_BOS = 1
TOK_META = 2
TOK_STATE = 3
TOK_MAT = 4
NUM_TOKEN_TYPES = 5


@dataclass
class Batch:
    tok_type: torch.Tensor   # (B,L) long
    tok_val: torch.Tensor    # (B,L,9) long
    attn01: torch.Tensor     # (B,L) bool
    lengths: torch.Tensor    # (B,) long
    m_i: torch.Tensor        # (B,) long
    y_tok: torch.Tensor      # (B,L) long with -100 ignore at non-MAT
    qk: torch.Tensor         # (B,) long 0..8


def collate_batch(items: List[dict], m_max: int, teacher_force_state: bool = False) -> Batch:
    B = len(items)
    Ts = [int(it["T"]) for it in items]
    if teacher_force_state:
        lengths = torch.tensor([2 + 2 * t for t in Ts], dtype=torch.long)
    else:
        lengths = torch.tensor([2 + t for t in Ts], dtype=torch.long)
    L_max = int(lengths.max().item())

    tok_type = torch.full((B, L_max), TOK_PAD, dtype=torch.long)
    tok_val = torch.zeros((B, L_max, 9), dtype=torch.long)
    attn01 = torch.zeros((B, L_max), dtype=torch.bool)

    m_i = torch.tensor([int(it["m"]) for it in items], dtype=torch.long)
    qk = torch.tensor([int(it["qk"]) for it in items], dtype=torch.long)

    y_tok = torch.full((B, L_max), -100, dtype=torch.long)

    for b, it in enumerate(items):
        T = int(it["T"])
        m = int(it["m"])
        qk_b = int(it["qk"])

        mats_mod = it["mats_mod"]  # (T,9) int16
        prev = it["prev"]          # (T,9) int16
        ys = it["y_steps"]         # (T,)  int16

        if m > m_max:
            raise ValueError(f"Found m={m} > m_max={m_max}. Increase --m_max or use auto-infer.")
        if int(mats_mod.max().item()) >= m_max:
            raise ValueError(f"Found residue >= m_max in mats_mod (m_max={m_max})")
        if int(prev.max().item()) >= m_max:
            raise ValueError(f"Found residue >= m_max in prev (m_max={m_max})")

        L = (2 + 2 * T) if teacher_force_state else (2 + T)
        attn01[b, :L] = True

        tok_type[b, 0] = TOK_BOS
        tok_type[b, 1] = TOK_META
        tok_val[b, 1, 0] = m
        tok_val[b, 1, 1] = qk_b  # store qk; remaining zeros

        pos = 2
        for t in range(T):
            if teacher_force_state:
                # STATE: P_{t-1} -- leaky ablation only, see module docstring
                tok_type[b, pos] = TOK_STATE
                tok_val[b, pos] = prev[t].to(torch.long)
                pos += 1

            tok_type[b, pos] = TOK_MAT
            tok_val[b, pos] = mats_mod[t].to(torch.long)

            # stepwise supervision at MAT token
            y_tok[b, pos] = ys[t].to(torch.long)
            pos += 1

    return Batch(tok_type, tok_val, attn01, lengths, m_i, y_tok, qk)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, m_max: int,
                teacher_force_state: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda items: collate_batch(items, m_max=m_max, teacher_force_state=teacher_force_state),
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
        v = self.val_emb(v9)                             # (B,L,9,d)
        p = self.pos_emb(self.pos_idx).view(1, 1, 9, -1)
        z = (v + p).reshape(v9.shape[0], v9.shape[1], -1)
        return self.proj(z)                              # (B,L,d)


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
    # No sequence-level positional embedding: RWKV7's time-mixing recurrence
    # already encodes token order, and a learned nn.Embedding(max_len, ...)
    # would leave rows beyond the training length untrained, silently
    # corrupting bin1/bin2 length-generalization eval (those rows never see a
    # gradient update). Residue9Encoder's own pos_emb (size 9) is a different,
    # legitimate thing -- it encodes which of the 9 matrix entries a value is,
    # not a sequence position.
    def __init__(self, m_max: int, max_len: int, d_model: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.m_max = m_max
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)

        self.res9_enc = Residue9Encoder(d_model=d_model, m_max=m_max)
        self.meta_enc = MetaEncoder(d_model=d_model, m_max=m_max)

        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, tok_val: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len}")

        x = self.type_emb(tok_type)

        sm_pos = (tok_type == TOK_STATE) | (tok_type == TOK_MAT)
        sm_mask = sm_pos.unsqueeze(-1).to(x.dtype)
        tok_val_sm = tok_val * sm_pos.unsqueeze(-1).to(tok_val.dtype)
        x = x + self.res9_enc(tok_val_sm) * sm_mask

        # META fixed at position 1
        meta_payload = tok_val[:, 1, :]
        x[:, 1, :] = x[:, 1, :] + self.meta_enc(meta_payload)

        return self.drop(x)


# =========================
# RWKV-7 blocks
# =========================
class RWKVBlock(nn.Module):
    """
    x = x + RWKV7Attention(LN(x))
    x = x + MLP(LN(x))
    """
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        rwkv_mode: str,
        layer_idx: int,
        num_hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        from fla.layers import RWKV7Attention  # import after patches

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RWKV7Attention(
            mode=rwkv_mode,
            hidden_size=d_model,
            head_dim=head_dim,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn01: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)

        mask = attn01
        if mask.dim() == 2 and mask.shape[0] != h.shape[0] and mask.shape[1] == h.shape[0]:
            mask = mask.transpose(0, 1)
        mask = mask.to(dtype=torch.bool, device=h.device).contiguous()

        h = self.attn(h, attention_mask=mask, past_key_values=None, use_cache=False)[0]
        x = x + self.drop1(h)

        h2 = self.ln2(x)
        x = x + self.drop2(self.mlp(h2))
        return x


class ModelRWKVStepwise(nn.Module):
    def __init__(
        self,
        m_max: int,
        max_len: int,
        d_model: int,
        head_dim: int,
        layers: int,
        dropout: float,
        rwkv_mode: str,
    ):
        super().__init__()
        self.m_max = m_max

        self.enc = BaseTokenEncoder(m_max=m_max, max_len=max_len, d_model=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            RWKVBlock(
                d_model=d_model,
                head_dim=head_dim,
                rwkv_mode=rwkv_mode,
                layer_idx=i,
                num_hidden_layers=layers,
                dropout=dropout,
            )
            for i in range(layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, m_max)  # (B,L,m_max)

    def forward(self, tok_type, tok_val, attn01):
        x = self.enc(tok_type, tok_val)
        for blk in self.blocks:
            x = blk(x, attn01=attn01)
        x = self.ln(x)
        return self.head(x)


# =========================
# loss / metrics
# =========================
def mask_logits_by_m(logits: torch.Tensor, m_i: torch.Tensor) -> torch.Tensor:
    B, L, m_max = logits.shape
    ar = torch.arange(m_max, device=logits.device).view(1, 1, m_max)
    mi = m_i.view(B, 1, 1)
    # dtype-aware fill value: -1e9 overflows fp16's much narrower range
    # (max ~65504) even though it's fine for fp32/bf16, which share fp32's
    # exponent range.
    neg = torch.finfo(logits.dtype).min / 2
    return logits.masked_fill(ar >= mi, neg)


def loss_steps_ce(logits: torch.Tensor, y_tok: torch.Tensor) -> torch.Tensor:
    B, L, m_max = logits.shape
    return F.cross_entropy(logits.reshape(B * L, m_max), y_tok.reshape(B * L), ignore_index=-100)


@torch.no_grad()
def metrics_step_and_final(logits: torch.Tensor, y_tok: torch.Tensor, lengths: torch.Tensor) -> Tuple[float, float]:
    pred = logits.argmax(dim=-1)
    valid = (y_tok != -100)
    step_acc = ((pred == y_tok) & valid).sum().item() / max(1, valid.sum().item())

    B = logits.shape[0]
    idx = (lengths - 1).clamp_min(0)
    last_pred = pred[torch.arange(B, device=logits.device), idx]
    last_y = y_tok[torch.arange(B, device=y_tok.device), idx]
    final_acc = (last_pred == last_y).float().mean().item()
    return float(step_acc), float(final_acc)


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
    max_steps: int,
    global_steps: int,
    log_every_steps: int,
) -> Tuple[float, float, float, int, int, bool]:
    train = optimizer is not None
    model.train(train)

    use_amp = amp and (device.type == "cuda")
    autocast_dtype = pick_autocast_dtype(device, amp_dtype)

    use_fp16 = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_loss = total_step = total_final = 0.0
    n_batches = 0
    opt_steps = 0
    hit_cap = False

    for batch_idx, batch in enumerate(tqdm(loader, dynamic_ncols=True, leave=False), start=1):
        if train and max_steps > 0 and global_steps >= max_steps:
            hit_cap = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        tok_val = batch.tok_val.to(device, non_blocking=True)
        attn01 = batch.attn01.to(device, non_blocking=True).to(torch.bool).contiguous()

        lengths = batch.lengths.to(device, non_blocking=True)
        m_i = batch.m_i.to(device, non_blocking=True)
        y_tok = batch.y_tok.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits = model(tok_type, tok_val, attn01)
            logits = mask_logits_by_m(logits, m_i)
            loss = loss_steps_ce(logits, y_tok)

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

            opt_steps += 1
            global_steps += 1

            if log_every_steps > 0 and (global_steps == 1 or global_steps % log_every_steps == 0):
                cap = max_steps if max_steps > 0 else -1
                print(f"[Step {global_steps:06d}/{cap}] loss={loss.item():.4f}")

            if max_steps > 0 and global_steps >= max_steps:
                hit_cap = True

        step_acc, final_acc = metrics_step_and_final(logits.detach(), y_tok, lengths)

        n_batches += 1
        total_loss += float(loss.item())
        total_step += step_acc
        total_final += final_acc

        if hit_cap and train:
            break

    denom = max(1, n_batches)
    return (
        total_loss / denom,
        total_step / denom,
        total_final / denom,
        global_steps,
        opt_steps,
        hit_cap,
    )


# =========================
# main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alphabet", type=str, default="pm1", choices=["01", "pm1", "any"])

    ap.add_argument("--m_max", type=int, default=0, help="0=infer from dataset")
    ap.add_argument("--max_len", type=int, default=0, help="0=infer (2+max_T, or 2+2*max_T if --teacher_force_state)")
    ap.add_argument("--splits", type=str, default="train,val_bin0,test_bin0,test_bin1,test_bin2")
    ap.add_argument("--teacher_force_state", action="store_true",
        help="Feed the ground-truth previous state P_{t-1} as an extra STATE token before "
             "each M_t (train AND eval). This leaks the answer -- v_t becomes a single dot "
             "product of the STATE and MAT tokens -- so the model never needs to accumulate "
             "state across steps. OFF by default (input-only tokenization). Only enable for "
             "an explicit single-step-given-true-state ablation.")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--rwkv7_head_dim", type=int, default=64)
    ap.add_argument("--rwkv7_depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rwkv7_mode", type=str, default="chunk", choices=["chunk", "naive", "fused_recurrent"])

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--log_every_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--early_stop", type=str, default="loss", choices=["loss", "stepAcc", "finalAcc"])
    ap.add_argument("--save_path", type=str, default="ckpt_rwkv7_mmquery_stepwise.pt")
    ap.add_argument("--eval_log", type=str, default="final_rwkv7_stepwise_eval.log")
    ap.add_argument("--quiet_preload", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.rwkv7_mode != "chunk":
        print("[Warn] RWKV7 training recommended in chunk mode. Forcing chunk.")
        args.rwkv7_mode = "chunk"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    print(f"[Task] MOD query STEPWISE | src=T|m|qk|mat.. tgt=v1..vT | {tf_desc}")
    print(f"[Data] {args.data_dir} alphabet={args.alphabet} m_max={args.m_max} max_T~{max_T}")
    print(f"[Model] RWKV7 max_len={args.max_len} d_model={args.d_model} head_dim={args.rwkv7_head_dim} depth={args.rwkv7_depth} dropout={args.dropout} mode={args.rwkv7_mode}")
    print(f"[Perf] batch_size={args.batch_size} workers={args.num_workers} amp={args.amp} {args.amp_dtype}")

    def split_paths(split: str) -> Tuple[str, str]:
        return (
            os.path.join(args.data_dir, f"{split}_src.txt"),
            os.path.join(args.data_dir, f"{split}_tgt.txt"),
        )

    train_src, train_tgt = split_paths("train")
    val_src, val_tgt = split_paths("val_bin0")
    t0_src, t0_tgt = split_paths("test_bin0")
    t1_src, t1_tgt = split_paths("test_bin1")
    t2_src, t2_tgt = split_paths("test_bin2")

    train_ds = PreloadedModQueryStepwiseDataset(train_src, train_tgt, args.alphabet, args.m_max, quiet=args.quiet_preload)
    val_ds   = PreloadedModQueryStepwiseDataset(val_src,   val_tgt,   args.alphabet, args.m_max, quiet=True)
    test0_ds = PreloadedModQueryStepwiseDataset(t0_src,    t0_tgt,    args.alphabet, args.m_max, quiet=True)
    test1_ds = PreloadedModQueryStepwiseDataset(t1_src,    t1_tgt,    args.alphabet, args.m_max, quiet=True)
    test2_ds = PreloadedModQueryStepwiseDataset(t2_src,    t2_tgt,    args.alphabet, args.m_max, quiet=True)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers, m_max=args.m_max, teacher_force_state=args.teacher_force_state)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers, m_max=args.m_max, teacher_force_state=args.teacher_force_state)
    test0_loader = make_loader(test0_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, m_max=args.m_max, teacher_force_state=args.teacher_force_state)
    test1_loader = make_loader(test1_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, m_max=args.m_max, teacher_force_state=args.teacher_force_state)
    test2_loader = make_loader(test2_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, m_max=args.m_max, teacher_force_state=args.teacher_force_state)

    model = ModelRWKVStepwise(
        m_max=args.m_max,
        max_len=args.max_len,
        d_model=args.d_model,
        head_dim=args.rwkv7_head_dim,
        layers=args.rwkv7_depth,
        dropout=args.dropout,
        rwkv_mode=args.rwkv7_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Params] {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # early stop bookkeeping
    if args.early_stop == "loss":
        best_val = float("inf")
    else:
        best_val = -1.0
    bad = 0
    global_steps = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_step, tr_final, global_steps, _, hit = run_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            global_steps=global_steps,
            log_every_steps=args.log_every_steps,
        )

        do_eval = (epoch % max(1, args.eval_every) == 0) or hit or (epoch == args.epochs)

        if do_eval:
            va_loss, va_step, va_final, _, _, _ = run_epoch(
                model, val_loader, device, None,
                amp=args.amp, amp_dtype=args.amp_dtype,
                grad_clip=0.0,
                max_steps=0,
                global_steps=0,
                log_every_steps=0,
            )

            cur = va_loss if args.early_stop == "loss" else (va_step if args.early_stop == "stepAcc" else va_final)
            if args.early_stop == "loss":
                improved = cur < best_val - 1e-6
            else:
                improved = cur > best_val + 1e-12

            if improved:
                best_val = cur
                bad = 0
                torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
            else:
                bad += 1

            print(
                f"Epoch {epoch:03d} | steps={global_steps} | "
                f"train loss={tr_loss:.4f} stepAcc={tr_step*100:.2f}% finalAcc={tr_final*100:.2f}% | "
                f"val loss={va_loss:.4f} stepAcc={va_step*100:.2f}% finalAcc={va_final*100:.2f}% | "
                f"best({args.early_stop})={best_val:.6f} bad={bad}/{args.patience}"
                f"{' [saved]' if improved else ''}"
            )

            if bad >= args.patience:
                print("Early stopping.")
                break
        else:
            print(
                f"Epoch {epoch:03d} | steps={global_steps} | "
                f"train loss={tr_loss:.4f} stepAcc={tr_step*100:.2f}% finalAcc={tr_final*100:.2f}% | (val skipped)"
            )

        if args.max_steps > 0 and global_steps >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}. Stopping training.")
            break

    # load best and eval
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\n[Eval best checkpoint]")
    for name, loader in [("test_bin0", test0_loader), ("test_bin1", test1_loader), ("test_bin2", test2_loader)]:
        te_loss, te_step, te_final, _, _, _ = run_epoch(
            model, loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=0.0,
            max_steps=0, global_steps=0,
            log_every_steps=0,
        )
        eval_str = f"{name:9s} | loss={te_loss:.4f} stepAcc={te_step*100:.2f}% finalAcc={te_final*100:.2f}%"
        print(eval_str)
        with open(args.eval_log, "a", encoding="utf-8") as logf:
            logf.write(args.data_dir + " " + args.save_path + " " + eval_str + "\n")

    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
