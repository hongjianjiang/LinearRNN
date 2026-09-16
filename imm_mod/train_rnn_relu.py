#!/usr/bin/env python3
# train_rnn_query_stepwise.py
"""
RNN/GRU baseline for MOD prefix-query dataset with STEPWISE targets.

Dataset (new gen.py):
  src: "T|m|qk|mat1|...|matT"
  tgt: "v1|v2|...|vT"
    where v_t = (P_t).flat[qk] mod m,  P_t = M1...Mt (mod m)

Tokens (always, both modes):
  BOS then T MAT tokens => length L = T+1
  At MAT tokens we always feed:
    - matrix residues (9 entries)
    - query column j (0..2) as embedding (broadcast to MAT tokens)

teacher_force_state=False (default): that's it -- the model must derive
  v_t itself from the raw matrix stream via its own recurrent state, matching
  the input-only tokenization used elsewhere.

teacher_force_state=True, leaky ablation only: we additionally compute
  r_prev[t] = row_i(P_{t-1}) in Z_m (i=qk//3) and feed it at each MAT token.
  Since v_t=(P_{t-1}@M_t)[qk] mod m, this reduces to a dot product of the
  given r_prev[t] row and the given M_t column j -- the model never needs to
  accumulate state across steps. Only use this for an explicit "given the
  true previous state, can the model do one correct step" ablation -- never
  for comparing task-solving ability against Mamba/Transformer.

Loss:
  Cross-entropy at each MAT token (stepwise).
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# =========================
# utils
# =========================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_max_T_from_src_path(src_path: str) -> int:
    # new src: "T|m|qk|mat|...|mat" -> T is explicit
    max_T = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("|")
            if len(parts) < 4:
                continue
            T = int(parts[0])
            if T > max_T:
                max_T = T
    return max_T


def infer_max_T_from_dir(data_dir: str, splits: List[str]) -> int:
    max_T = 0
    for sp in splits:
        src_path = os.path.join(data_dir, f"{sp}_src.txt")
        if os.path.exists(src_path):
            max_T = max(max_T, infer_max_T_from_src_path(src_path))
    if max_T <= 0:
        raise ValueError(f"Could not infer max_T from {data_dir}")
    return max_T


def parse_src_line_T_m_qk(src_line: str, N: int = 3) -> Tuple[int, int, int, np.ndarray]:
    """
    src: "T|m|qk|mat1|...|matT"
    returns (T, m, qk, mats_raw[T,9])
    """
    parts = src_line.strip().split("|")
    if len(parts) < 4:
        raise ValueError("Bad src line: expected T|m|qk|mat|...|mat")

    T = int(parts[0])
    m = int(parts[1])
    qk = int(parts[2])

    if T <= 0:
        raise ValueError(f"Bad T={T}")
    if m < 2:
        raise ValueError(f"Bad m={m}")
    if not (0 <= qk <= 8):
        raise ValueError(f"Bad qk={qk}, expected 0..8")

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


def compute_row_prev_sequence_for_query(mats_raw: np.ndarray, m: int, qk: int, N: int = 3) -> np.ndarray:
    """
    r_prev[t] = row_i(P_{t-1}) where i=qk//3 and P_{t}=P_{t-1}*M_t mod m.
    returns r_prev shape (T,3) in [0..m-1]
    """
    T = mats_raw.shape[0]
    mats = (mats_raw % m).reshape(T, N, N).astype(np.int64, copy=False)

    i = qk // 3
    P = np.eye(N, dtype=np.int64) % m
    r_prev = np.empty((T, N), dtype=np.int64)

    for t in range(T):
        r_prev[t] = P[i]  # row i of P_{t-1}
        P = (P @ mats[t]) % m

    return r_prev


def compute_full_state_sequence(mats_raw: np.ndarray, m: int, N: int = 3) -> np.ndarray:
    """
    aux_tgt[t] = P_t (all 9 residues, flattened row-major), for t=1..T.
    Output-only auxiliary target -- never fed back as model input, so this
    adds no leak. Gives the model dense supervision on the full state it
    actually needs to track (v_t only reads one entry of it) instead of just
    the single queried scalar.
    """
    T = mats_raw.shape[0]
    mats = (mats_raw % m).reshape(T, N, N).astype(np.int64, copy=False)

    P = np.eye(N, dtype=np.int64) % m
    aux = np.empty((T, N * N), dtype=np.int64)

    for t in range(T):
        P = (P @ mats[t]) % m
        aux[t] = P.reshape(-1)

    return aux


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return torch.sign(x) * torch.log1p(torch.abs(x))


# =========================
# dataset (preloaded, torch tensors)
# =========================
class PreloadedQueryStepDataset(Dataset):
    """
    Stores per-sample tensors on CPU:
      mats_mod: (T,9) int16
      r_prev:   (T,3) int16
      y_steps:  (T,)  int16
    And scalars:
      T: int
      m: int
      qj: int   (j = qk%3)  used as query-col embedding
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
        self.qj: List[int] = []
        self.mats_mod: List[torch.Tensor] = []
        self.r_prev: List[torch.Tensor] = []
        self.y_steps: List[torch.Tensor] = []
        self.aux_tgt: List[torch.Tensor] = []

        it = list(zip(src_lines, tgt_lines))
        if not quiet:
            print(f"[Preload] {os.path.basename(src_path)} n={len(it)}")

        for i, (src, tgt) in enumerate(it):
            T, m_i, qk, mats_raw = parse_src_line_T_m_qk(src, N=3)
            if not (2 <= m_i <= m_max):
                raise ValueError(f"m_i={m_i} out of range [2..m_max={m_max}] at line {i}")

            if alphabet == "pm1":
                ok = np.all((mats_raw == -1) | (mats_raw == 0) | (mats_raw == 1))
            elif alphabet == "01":
                ok = np.all((mats_raw == 0) | (mats_raw == 1))
            else:
                raise ValueError(f"Unknown alphabet: {alphabet}")
            if not ok:
                raise ValueError(f"Alphabet mismatch in src at line {i}")

            mats_mod = np.remainder(mats_raw, m_i).astype(np.int64, copy=False)  # (T,9) in [0..m-1]

            y = parse_tgt_steps(tgt, T=T)
            y = np.remainder(y, m_i).astype(np.int64, copy=False)                # (T,) in [0..m-1]

            r_prev = compute_row_prev_sequence_for_query(mats_raw, m=m_i, qk=qk, N=3)  # (T,3)
            aux = compute_full_state_sequence(mats_raw, m=m_i, N=3)              # (T,9), output-only
            j = int(qk % 3)

            self.Ts.append(T)
            self.ms.append(m_i)
            self.qj.append(j)

            self.mats_mod.append(torch.from_numpy(mats_mod).to(torch.int16))    # (T,9)
            self.r_prev.append(torch.from_numpy(r_prev).to(torch.int16))        # (T,3)
            self.y_steps.append(torch.from_numpy(y).to(torch.int16))            # (T,)
            self.aux_tgt.append(torch.from_numpy(aux).to(torch.int16))          # (T,9)

    def __len__(self) -> int:
        return len(self.Ts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "T": self.Ts[idx],
            "m": self.ms[idx],
            "qj": self.qj[idx],
            "mats_mod": self.mats_mod[idx],
            "r_prev": self.r_prev[idx],
            "y_steps": self.y_steps[idx],
            "aux_tgt": self.aux_tgt[idx],
        }


# =========================
# length-bucket batch sampler
# =========================
class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths_T: List[int],
        batch_size: int,
        shuffle: bool,
        bucket_size: int = 4096,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.lengths_T = lengths_T
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.indices_sorted = list(range(len(lengths_T)))
        self.indices_sorted.sort(key=lambda i: lengths_T[i])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        idx = self.indices_sorted
        buckets = [idx[i : i + self.bucket_size] for i in range(0, len(idx), self.bucket_size)]

        if self.shuffle:
            for b in buckets:
                perm = torch.randperm(len(b), generator=g).tolist()
                b[:] = [b[j] for j in perm]
            perm_b = torch.randperm(len(buckets), generator=g).tolist()
            buckets = [buckets[j] for j in perm_b]

        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        # approximate; fine for progress bars / bookkeeping
        n = len(self.lengths_T)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# =========================
# collate
# =========================
TOK_PAD = 0
TOK_BOS = 1
TOK_MAT = 2
NUM_TOKEN_TYPES = 3


@dataclass
class Batch:
    tok_type: torch.Tensor     # (B,L) long
    mats_mod: torch.Tensor     # (B,L,9) int16
    r_prev: torch.Tensor       # (B,L,3) int16
    y_tok: torch.Tensor        # (B,L) long (with -100 ignore)
    lengths: torch.Tensor      # (B,) long  (L = T+1)
    m_i: torch.Tensor          # (B,) long
    qj: torch.Tensor           # (B,) long  (0..2)
    aux_tgt: torch.Tensor      # (B,L,9) long (P_t residues at MAT positions, output-only)


def collate_batch(items: List[dict], m_max: int) -> Batch:
    B = len(items)
    Ts = [int(it["T"]) for it in items]
    lengths = torch.tensor([t + 1 for t in Ts], dtype=torch.long)
    L_max = int(lengths.max().item())

    tok_type = torch.full((B, L_max), TOK_PAD, dtype=torch.long)
    mats_mod = torch.zeros((B, L_max, 9), dtype=torch.int16)
    r_prev = torch.zeros((B, L_max, 3), dtype=torch.int16)
    y_tok = torch.full((B, L_max), -100, dtype=torch.long)
    aux_tgt = torch.zeros((B, L_max, 9), dtype=torch.long)

    m_i = torch.tensor([int(it["m"]) for it in items], dtype=torch.long)
    qj = torch.tensor([int(it["qj"]) for it in items], dtype=torch.long)

    for b, it in enumerate(items):
        T = int(it["T"])
        L = T + 1
        tok_type[b, 0] = TOK_BOS
        tok_type[b, 1:L] = TOK_MAT

        mats = it["mats_mod"]    # (T,9) int16
        rows = it["r_prev"]      # (T,3) int16
        ys = it["y_steps"]       # (T,)  int16
        aux = it["aux_tgt"]      # (T,9) int16

        # sanity: residues must be in [0, m_i-1] and m_i <= m_max, so < m_max always holds
        if int(mats.max().item()) >= m_max:
            raise ValueError(f"Found residue >= m_max in mats_mod (m_max={m_max})")
        if int(rows.max().item()) >= m_max:
            raise ValueError(f"Found residue >= m_max in r_prev (m_max={m_max})")

        mats_mod[b, 1:L].copy_(mats)
        r_prev[b, 1:L].copy_(rows)
        y_tok[b, 1:L] = ys.to(torch.long)
        aux_tgt[b, 1:L, :] = aux.to(torch.long)

    return Batch(tok_type, mats_mod, r_prev, y_tok, lengths, m_i, qj, aux_tgt)


def make_loader(
    ds: PreloadedQueryStepDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    m_max: int,
    prefetch_factor: int,
    persistent_workers: bool,
    bucket_size: int,
    seed: int,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(
        lengths_T=ds.Ts,
        batch_size=batch_size,
        shuffle=shuffle,
        bucket_size=bucket_size,
        drop_last=False,
        seed=seed,
    )

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        collate_fn=lambda items: collate_batch(items, m_max=m_max),
    )


# =========================
# encoders
# =========================
class RowStateEncoder(nn.Module):
    def __init__(self, d_model: int, m_max: int):
        super().__init__()
        self.val_emb = nn.Embedding(m_max, d_model)
        self.pos_emb = nn.Embedding(3, d_model)
        self.proj = nn.Linear(3 * d_model, d_model)
        self.register_buffer("pos_idx", torch.arange(3, dtype=torch.long), persistent=False)

    def forward(self, r_prev_long: torch.Tensor) -> torch.Tensor:
        v = self.val_emb(r_prev_long)                         # (B,L,3,d)
        p = self.pos_emb(self.pos_idx).view(1, 1, 3, -1)      # (1,1,3,d)
        z = (v + p).reshape(r_prev_long.shape[0], r_prev_long.shape[1], -1)
        return self.proj(z)                                   # (B,L,d)


class MatrixResidueEncoder(nn.Module):
    def __init__(self, d_model: int, m_max: int):
        super().__init__()
        self.val_emb = nn.Embedding(m_max, d_model)
        self.entry_pos = nn.Embedding(9, d_model)
        self.proj = nn.Linear(9 * d_model, d_model)
        self.register_buffer("pos_idx", torch.arange(9, dtype=torch.long), persistent=False)

    def forward(self, mats_mod_long: torch.Tensor) -> torch.Tensor:
        v = self.val_emb(mats_mod_long)                       # (B,L,9,d)
        p = self.entry_pos(self.pos_idx).view(1, 1, 9, -1)
        z = (v + p).reshape(mats_mod_long.shape[0], mats_mod_long.shape[1], -1)
        return self.proj(z)                                   # (B,L,d)


class BaseTokenEncoder(nn.Module):
    # No sequence-level positional embedding: the RNN's own recurrence
    # already encodes token order, and a learned nn.Embedding(max_len, ...)
    # would leave rows beyond the training length untrained, silently
    # corrupting bin1/bin2 length-generalization eval (those rows never see
    # a gradient update). MatrixResidueEncoder/RowStateEncoder's own pos_emb
    # (size 9/3) is a different, legitimate thing -- it encodes which matrix
    # entry a value is, not a sequence position.
    def __init__(self, m_max: int, max_len: int, d_model: int, dropout: float, teacher_force_state: bool = False):
        super().__init__()
        self.max_len = max_len
        self.teacher_force_state = bool(teacher_force_state)
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)

        self.row_enc = RowStateEncoder(d_model=d_model, m_max=m_max)
        self.mat_enc = MatrixResidueEncoder(d_model=d_model, m_max=m_max)

        # query column j in {0,1,2}
        self.qj_emb = nn.Embedding(3, d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, mats_mod_long: torch.Tensor, r_prev_long: torch.Tensor, qj: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len}")

        x = self.type_emb(tok_type)

        mat_pos = (tok_type == TOK_MAT)
        mat_mask = mat_pos.unsqueeze(-1).to(x.dtype)

        x = x + self.mat_enc(mats_mod_long) * mat_mask
        if self.teacher_force_state:
            # leaks P_{t-1} -- see module docstring; off by default
            x = x + self.row_enc(r_prev_long) * mat_mask

        qj_vec = self.qj_emb(qj).unsqueeze(1)  # (B,1,d)
        x = x + qj_vec * mat_mask

        return self.drop(x)


# =========================
# RNN model (packed)
# =========================
class ModelRNNQuery(nn.Module):
    def __init__(self, rnn_kind: str, m_max: int, max_len: int, d_model: int, layers: int, dropout: float,
                 act_clip: float, teacher_force_state: bool = False):
        super().__init__()
        self.m_max = m_max
        self.act_clip = float(act_clip)

        self.enc = BaseTokenEncoder(m_max=m_max, max_len=max_len, d_model=d_model, dropout=dropout,
                                     teacher_force_state=teacher_force_state)

        if rnn_kind == "gru":
            self.rnn = nn.GRU(
                input_size=d_model, hidden_size=d_model, num_layers=layers,
                batch_first=True, dropout=(dropout if layers > 1 else 0.0)
            )
        elif rnn_kind == "rnn_tanh":
            self.rnn = nn.RNN(
                input_size=d_model, hidden_size=d_model, num_layers=layers, nonlinearity="tanh",
                batch_first=True, dropout=(dropout if layers > 1 else 0.0)
            )
        elif rnn_kind == "rnn_relu":
            self.rnn = nn.RNN(
                input_size=d_model, hidden_size=d_model, num_layers=layers, nonlinearity="relu",
                batch_first=True, dropout=(dropout if layers > 1 else 0.0)
            )
        else:
            raise ValueError(f"Unknown rnn_kind={rnn_kind}")

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, m_max)
        # output-only auxiliary head: predicts phi(P_t) (all 9 residues of the
        # full state), never fed back as input -- see compute_full_state_sequence.
        self.aux_head = nn.Linear(d_model, 9)

    def forward(self, tok_type, mats_mod_long, r_prev_long, qj, lengths):
        x = self.enc(tok_type, mats_mod_long, r_prev_long, qj)  # (B,L,d)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        if self.act_clip > 0:
            x = x.clamp(-self.act_clip, self.act_clip)

        x = self.ln(x)
        return self.head(x), self.aux_head(x)  # (B,L,m_max), (B,L,9)


# =========================
# loss / metrics
# =========================
def mask_logits_by_m(logits: torch.Tensor, m_i: torch.Tensor) -> torch.Tensor:
    B, L, m_max = logits.shape
    ar = torch.arange(m_max, device=logits.device).view(1, 1, m_max)
    mi = m_i.view(B, 1, 1)
    return logits.masked_fill(ar >= mi, -1e9)


def loss_steps_ce(logits: torch.Tensor, y_tok: torch.Tensor) -> torch.Tensor:
    B, L, m_max = logits.shape
    return F.cross_entropy(logits.reshape(B * L, m_max), y_tok.reshape(B * L), ignore_index=-100)


def aux_state_loss_smoothl1(aux_pred_phi: torch.Tensor, aux_tgt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum().item() == 0:
        return aux_pred_phi.new_tensor(0.0)
    tgt_phi = signed_log1p(aux_tgt)
    m = valid_mask.unsqueeze(-1)
    pred = aux_pred_phi[m.expand_as(aux_pred_phi)].view(-1, 9)
    tgt = tgt_phi[m.expand_as(tgt_phi)].view(-1, 9)
    return F.smooth_l1_loss(pred, tgt, reduction="mean")


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
# train / eval loops
# =========================
def pick_autocast_dtype(device: torch.device, amp_dtype: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


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
    val_max_batches: int,
    aux_w: float = 0.0,
) -> Tuple[float, float, float, int, int, int, bool]:
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

    for batch_idx, batch in enumerate(loader, start=1):
        if (not train) and (val_max_batches > 0) and (batch_idx > val_max_batches):
            break
        if train and max_steps > 0 and global_steps >= max_steps:
            hit_cap = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        lengths = batch.lengths.to(device, non_blocking=True)
        m_i = batch.m_i.to(device, non_blocking=True)
        qj = batch.qj.to(device, non_blocking=True)

        mats_mod_long = batch.mats_mod.to(device, non_blocking=True).long()
        r_prev_long = batch.r_prev.to(device, non_blocking=True).long()
        y_tok = batch.y_tok.to(device, non_blocking=True)
        aux_tgt = batch.aux_tgt.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits, aux_pred = model(tok_type, mats_mod_long, r_prev_long, qj, lengths)
            logits = mask_logits_by_m(logits, m_i)
            loss = loss_steps_ce(logits, y_tok)
            if aux_w > 0:
                loss = loss + float(aux_w) * aux_state_loss_smoothl1(aux_pred, aux_tgt, (y_tok != -100))

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

        if train and hit_cap:
            break

    denom = max(1, n_batches)
    return (
        total_loss / denom,
        total_step / denom,
        total_final / denom,
        n_batches,
        opt_steps,
        global_steps,
        hit_cap,
    )


# =========================
# main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alphabet", type=str, default="pm1", choices=["01", "pm1"])
    ap.add_argument("--m_max", type=int, required=True)

    ap.add_argument("--splits", type=str, default="train,val_bin0,test_bin0,test_bin1,test_bin2")
    ap.add_argument("--max_len", type=int, default=0)  # 0 => auto from data
    ap.add_argument("--teacher_force_state", action="store_true",
        help="Feed the ground-truth previous-row state r_prev[t]=row_i(P_{t-1}) at each MAT "
             "token (train AND eval). This leaks the answer -- v_t becomes a single dot "
             "product of the given row and the given matrix column -- so the model never "
             "needs to accumulate state across steps. OFF by default (input-only "
             "tokenization). Only enable for an explicit single-step-given-true-state ablation.")

    ap.add_argument("--rnn", type=str, default="gru", choices=["gru", "rnn_tanh", "rnn_relu"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--act_clip", type=float, default=0.0)
    ap.add_argument("--aux_w", type=float, default=0.0,
        help="Weight for an output-only auxiliary SmoothL1 loss predicting phi(P_t) "
             "(all 9 residues of the full state, not just the queried entry v_t) at "
             "every MAT token. Never fed back as input -- 0.0 disables it (default).")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--bucket_size", type=int, default=4096)

    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--log_every_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--val_max_batches", type=int, default=0)

    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--early_stop", type=str, default="loss", choices=["loss", "stepAcc", "finalAcc"])

    ap.add_argument("--save_path", type=str, default="ckpt_gru_mmquery_stepwise.pt")
    ap.add_argument("--quiet_preload", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.max_len <= 0:
        max_T = infer_max_T_from_dir(args.data_dir, split_list)
        args.max_len = max_T + 1
        print(f"[Auto] inferred max_T(all splits)={max_T} => max_len={args.max_len}")
    else:
        max_T = args.max_len - 1

    tf_desc = "teacher forcing r_prev row (leaks P_{t-1}, train+eval)" if args.teacher_force_state else "input-only (no teacher forcing)"
    print(f"[Device] {device}")
    print(f"[Task] MOD query stepwise: src=T|m|qk|mat.. tgt=v1..vT ; j=qk%3 | {tf_desc}")
    print(f"[Data] {args.data_dir} alphabet={args.alphabet} m_max={args.m_max} max_T~{max_T}")
    print(f"[Model] {args.rnn} packed + buckets | max_len={args.max_len} d_model={args.d_model} layers={args.layers} dropout={args.dropout} act_clip={args.act_clip}")
    print(f"[Perf] batch_size={args.batch_size} workers={args.num_workers} bucket_size={args.bucket_size} amp={args.amp} {args.amp_dtype}")
    print(f"[Aux ] aux_w={args.aux_w}")

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

    train_ds = PreloadedQueryStepDataset(train_src, train_tgt, args.alphabet, args.m_max, quiet=args.quiet_preload)
    val_ds   = PreloadedQueryStepDataset(val_src,   val_tgt,   args.alphabet, args.m_max, quiet=True)
    test0_ds = PreloadedQueryStepDataset(t0_src,    t0_tgt,    args.alphabet, args.m_max, quiet=True)
    test1_ds = PreloadedQueryStepDataset(t1_src,    t1_tgt,    args.alphabet, args.m_max, quiet=True)
    test2_ds = PreloadedQueryStepDataset(t2_src,    t2_tgt,    args.alphabet, args.m_max, quiet=True)

    train_loader = make_loader(train_ds, args.batch_size, True,  args.num_workers, args.m_max,
                               args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed)
    val_loader   = make_loader(val_ds,   args.batch_size, False, args.num_workers, args.m_max,
                               args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 999)
    test0_loader = make_loader(test0_ds, args.batch_size, False, args.num_workers, args.m_max,
                               args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 1999)
    test1_loader = make_loader(test1_ds, args.batch_size, False, args.num_workers, args.m_max,
                               args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 2999)
    test2_loader = make_loader(test2_ds, args.batch_size, False, args.num_workers, args.m_max,
                               args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 3999)

    model = ModelRNNQuery(
        rnn_kind=args.rnn,
        m_max=args.m_max,
        max_len=args.max_len,
        d_model=args.d_model,
        layers=args.layers,
        dropout=args.dropout,
        act_clip=args.act_clip,
        teacher_force_state=args.teacher_force_state,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Params] {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- fixed early-stop bookkeeping (no closure bugs) ----
    if args.early_stop == "loss":
        best_val = float("inf")
    else:
        best_val = -1.0
    bad = 0
    global_steps = 0

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        tr_loss, tr_step, tr_final, _, _, global_steps, hit_cap = run_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            global_steps=global_steps,
            log_every_steps=args.log_every_steps,
            val_max_batches=0,
            aux_w=args.aux_w,
        )

        do_eval = (epoch % max(1, args.eval_every) == 0) or hit_cap or (epoch == args.epochs)

        if do_eval:
            va_loss, va_step, va_final, _, _, _, _ = run_epoch(
                model, val_loader, device, None,
                amp=args.amp, amp_dtype=args.amp_dtype,
                grad_clip=0.0,
                max_steps=0, global_steps=0,
                log_every_steps=0,
                val_max_batches=args.val_max_batches,
                aux_w=args.aux_w,
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
        te_loss, te_step, te_final, _, _, _, _ = run_epoch(
            model, loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=0.0,
            max_steps=0, global_steps=0,
            log_every_steps=0,
            val_max_batches=0,
            aux_w=args.aux_w,
        )
        eval_str = f"{name:9s} | loss={te_loss:.4f} stepAcc={te_step*100:.2f}% finalAcc={te_final*100:.2f}%"
        print(eval_str)
        with open("final_rnn_eval.log", "a", encoding="utf-8") as logf:
            logf.write(args.data_dir + " " + eval_str + "\n")

    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
