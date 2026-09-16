#!/usr/bin/env python3
# train_transformer_mmZ_binary.py
"""
Transformer baseline for:
  3x3 matrix multiplication over Z (NO MOD),
  SINGLE BINARY LABEL.

Dataset:
  src: "n|a1,...,a9|a1,...,a9|...|a1,...,a9"  where n==T
  tgt: "0" or "1"

Model:
  tokens: [BOS], [MAT_1..MAT_T]
  encoder-only Transformer (no causal mask)
  take last valid token representation -> 2-class head

Notes:
- This is input-only (no teacher-forced states).
- If your generator used --cap for labels, that only affects labels; inputs stay pm1.
"""

from __future__ import annotations
import os, time, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def value_vocab_size(alphabet: str) -> int:
    if alphabet == "pm1":
        return 3
    if alphabet == "01":
        return 2
    raise ValueError(f"Unknown alphabet={alphabet}")


def mats_to_value_indices(mats_raw: np.ndarray, alphabet: str) -> np.ndarray:
    """
    mats_raw: (T,9) int64 entries
    returns:  (T,9) int64 indices
    """
    if alphabet == "pm1":
        ok = np.all((mats_raw == -1) | (mats_raw == 0) | (mats_raw == 1))
        if not ok:
            bad = mats_raw[(mats_raw != -1) & (mats_raw != 0) & (mats_raw != 1)]
            raise ValueError(f"Alphabet mismatch pm1; found values like {bad[:10]}")
        return (mats_raw + 1).astype(np.int64, copy=False)  # -1->0, 0->1, 1->2
    elif alphabet == "01":
        ok = np.all((mats_raw == 0) | (mats_raw == 1))
        if not ok:
            bad = mats_raw[(mats_raw != 0) & (mats_raw != 1)]
            raise ValueError(f"Alphabet mismatch 01; found values like {bad[:10]}")
        return mats_raw.astype(np.int64, copy=False)
    else:
        raise ValueError(f"Unknown alphabet={alphabet}")


def parse_src_n_mats(line: str, N: int = 3) -> Tuple[int, np.ndarray]:
    """
    src: "n|a1,...,a9|...|a1,...,a9"
    returns:
      T: int
      mats_raw: (T,9) int64
    """
    parts = line.strip().split("|")
    if len(parts) < 2:
        raise ValueError("Bad src line: expected n|mat1|...|matn")
    T = int(parts[0])
    mats_parts = parts[1:]
    if len(mats_parts) != T:
        raise ValueError(f"Bad src: header n={T} but got {len(mats_parts)} mat blocks")

    D = N * N
    mats = np.empty((T, D), dtype=np.int64)
    for t, blk in enumerate(mats_parts):
        xs = blk.split(",")
        if len(xs) != D:
            raise ValueError(f"Bad mat len at t={t}: got {len(xs)} expected {D}")
        mats[t] = np.fromiter((int(v) for v in xs), dtype=np.int64, count=D)
    return T, mats


def parse_tgt_binary(line: str) -> int:
    s = line.strip()
    if s not in ("0", "1"):
        raise ValueError(f"Bad tgt line (expected 0/1): {s[:50]}")
    return int(s)


def infer_max_T_from_src_path(src_path: str) -> int:
    mx = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("|")
            if len(parts) < 2:
                raise ValueError(f"Bad src line: {ln[:120]}")
            mx = max(mx, int(parts[0]))
    return mx


def infer_max_T_from_dir(data_dir: str, splits: List[str]) -> int:
    mx = 0
    for sp in splits:
        srcp = os.path.join(data_dir, f"{sp}_src.txt")
        if not os.path.exists(srcp):
            continue
        mx = max(mx, infer_max_T_from_src_path(srcp))
    if mx <= 0:
        raise ValueError(f"Could not infer max_T from {data_dir} over splits={splits}")
    return mx

# -------------------------
# lazy dataset via offsets (low RAM)
# -------------------------
def build_line_offsets(path: str) -> List[int]:
    offs: List[int] = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offs.append(pos)
    return offs


class LazyMMZBinaryDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, alphabet: str):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.alphabet = alphabet

        self.src_offs = build_line_offsets(src_path)
        self.tgt_offs = build_line_offsets(tgt_path)
        if len(self.src_offs) != len(self.tgt_offs):
            raise ValueError(f"src/tgt mismatch: {len(self.src_offs)} vs {len(self.tgt_offs)}")

        self._src_f = None
        self._tgt_f = None

    def _ensure_open(self):
        if self._src_f is None:
            self._src_f = open(self.src_path, "r", encoding="utf-8")
        if self._tgt_f is None:
            self._tgt_f = open(self.tgt_path, "r", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.src_offs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        self._ensure_open()
        self._src_f.seek(self.src_offs[i])
        self._tgt_f.seek(self.tgt_offs[i])
        src = self._src_f.readline().strip()
        tgt = self._tgt_f.readline().strip()

        T, mats_raw = parse_src_n_mats(src, N=3)
        y = parse_tgt_binary(tgt)
        mats_idx = mats_to_value_indices(mats_raw, alphabet=self.alphabet)
        return {"T": int(T), "mats_idx": mats_idx, "y": int(y)}


# -------------------------
# collate
# -------------------------
TOK_PAD = 0
TOK_BOS = 1
TOK_MAT = 2
NUM_TOKEN_TYPES = 3


@dataclass
class Batch:
    tok_type: torch.Tensor    # (B,L)
    mats_idx: torch.Tensor    # (B,L,9)
    attn01: torch.Tensor      # (B,L) bool
    lengths: torch.Tensor     # (B,)
    y: torch.Tensor           # (B,)


def collate_batch(items: List[Dict[str, Any]]) -> Batch:
    B = len(items)
    Ts = [int(it["T"]) for it in items]
    lengths = torch.tensor([t + 1 for t in Ts], dtype=torch.long)  # BOS + T
    Lmax = int(lengths.max().item())

    tok_type = torch.full((B, Lmax), TOK_PAD, dtype=torch.long)
    mats_idx = torch.zeros((B, Lmax, 9), dtype=torch.long)
    attn01 = torch.zeros((B, Lmax), dtype=torch.bool)
    y = torch.tensor([int(it["y"]) for it in items], dtype=torch.long)

    for b, it in enumerate(items):
        T = int(it["T"])
        L = T + 1
        tok_type[b, 0] = TOK_BOS
        tok_type[b, 1:L] = TOK_MAT
        attn01[b, :L] = True

        mats = torch.from_numpy(it["mats_idx"]).long()  # (T,9)
        mats_idx[b, 1:L] = mats

    return Batch(tok_type=tok_type, mats_idx=mats_idx, attn01=attn01, lengths=lengths, y=y)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


# -------------------------
# model
# -------------------------
class MatrixValueEncoder(nn.Module):
    """
    Encode 9 entries of a 3x3 matrix into d_model.
    """
    def __init__(self, d_model: int, val_vocab: int):
        super().__init__()
        self.val_emb = nn.Embedding(val_vocab, d_model)
        self.entry_pos = nn.Embedding(9, d_model)
        self.proj = nn.Linear(9 * d_model, d_model)
        self.register_buffer("pos_idx", torch.arange(9, dtype=torch.long), persistent=False)

    def forward(self, mats_idx: torch.Tensor) -> torch.Tensor:
        # mats_idx: (B,L,9)
        v = self.val_emb(mats_idx)                         # (B,L,9,d)
        p = self.entry_pos(self.pos_idx).view(1, 1, 9, -1) # (1,1,9,d)
        z = (v + p).reshape(mats_idx.shape[0], mats_idx.shape[1], -1)
        return self.proj(z)                                # (B,L,d)


class TokenEncoder(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float, val_vocab: int):
        super().__init__()
        self.max_len = int(max_len)
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.mat_enc = MatrixValueEncoder(d_model, val_vocab)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, mats_idx: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len} (increase --max_len)")
        pos = torch.arange(L, device=tok_type.device).unsqueeze(0)
        x = self.type_emb(tok_type) + self.pos_emb(pos)

        mat_mask = (tok_type == TOK_MAT).unsqueeze(-1).to(x.dtype)
        x = x + self.mat_enc(mats_idx) * mat_mask
        return self.drop(x)


class TransformerBinaryClassifier(nn.Module):
    def __init__(self, max_len: int, d_model: int, heads: int, layers: int,
                 dropout: float, ff_mult: int, val_vocab: int):
        super().__init__()
        self.enc = TokenEncoder(max_len, d_model, dropout, val_vocab)

        dim_ff = ff_mult * d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, tok_type: torch.Tensor, mats_idx: torch.Tensor, attn01: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # (B,L,d)
        x = self.enc(tok_type, mats_idx)

        # key_padding_mask: True where padding
        key_padding_mask = ~attn01
        x = self.tr(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)

        # last valid token per sample (index = lengths-1)
        idx = (lengths - 1).clamp_min(0)
        last = x[torch.arange(x.size(0), device=x.device), idx]  # (B,d)
        return self.head(last)  # (B,2)


# -------------------------
# train / eval
# -------------------------
def ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


@torch.no_grad()
def acc01(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    amp: bool,
    amp_dtype: str,
    grad_clip: float,
    max_steps: int = 0,
    global_step: int = 0,
) -> Tuple[float, float, int, bool]:
    train = optimizer is not None
    model.train(train)

    use_amp = amp and (device.type == "cuda")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_fp16 = use_amp and (amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_loss = total_acc = 0.0
    n_batches = 0
    hit_limit = False

    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for batch in pbar:
        if train and max_steps > 0 and global_step >= max_steps:
            hit_limit = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        mats_idx = batch.mats_idx.to(device, non_blocking=True)
        attn01   = batch.attn01.to(device, non_blocking=True)
        lengths  = batch.lengths.to(device, non_blocking=True)
        y        = batch.y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits = model(tok_type, mats_idx, attn01, lengths=lengths)
            loss = ce_loss(logits, y)

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

        a = acc01(logits.detach(), y)
        n_batches += 1
        total_loss += float(loss.item())
        total_acc += float(a)
        pbar.set_postfix(loss=total_loss / n_batches, acc=100 * total_acc / n_batches, gstep=global_step)

        if hit_limit:
            break

    denom = max(1, n_batches)
    return total_loss / denom, total_acc / denom, global_step, hit_limit


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alphabet", type=str, default="pm1", choices=["01", "pm1"])
    ap.add_argument("--max_len", type=int, default=0, help="0=infer (max_T+1 for BOS)")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ff_mult", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--early_stop", type=str, default="acc", choices=["acc", "loss"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--save_path", type=str, default="ckpt_transformer_mmZ_binary.pt")
    ap.add_argument("--max_steps", type=int, default=0, help="0=unlimited (full epochs)")
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    def sp_paths(split: str) -> Tuple[str, str]:
        return (os.path.join(args.data_dir, f"{split}_src.txt"),
                os.path.join(args.data_dir, f"{split}_tgt.txt"))

    train_src, _ = sp_paths("train")
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing train_src: {train_src}")

    all_splits = ["train", "val_bin0", "test_bin0", "test_bin1", "test_bin2"]
    max_T_all = infer_max_T_from_dir(args.data_dir, all_splits)

    if args.max_len <= 0:
        args.max_len = max_T_all + 1  # BOS + T
        print(f"[Auto] inferred max_T(all splits)={max_T_all} => max_len={args.max_len}")
    else:
        print(f"[User] max_len={args.max_len}")

    val_vocab = value_vocab_size(args.alphabet)

    print(f"[Device] {device}")
    print(f"[Stats] max_T={max_T_all} => max_len={args.max_len}")
    print(f"[Model] d_model={args.d_model} heads={args.heads} layers={args.layers} dropout={args.dropout} ff_mult={args.ff_mult}")
    if args.max_steps and args.max_steps > 0:
        print(f"[Train] max_steps={args.max_steps}")

    train_ds = LazyMMZBinaryDataset(*sp_paths("train"), alphabet=args.alphabet)
    val_ds   = LazyMMZBinaryDataset(*sp_paths("val_bin0"), alphabet=args.alphabet)
    test0_ds = LazyMMZBinaryDataset(*sp_paths("test_bin0"), alphabet=args.alphabet)
    test1_ds = LazyMMZBinaryDataset(*sp_paths("test_bin1"), alphabet=args.alphabet)
    test2_ds = LazyMMZBinaryDataset(*sp_paths("test_bin2"), alphabet=args.alphabet)

    train_loader = make_loader(train_ds, args.batch_size, True,  args.num_workers)
    val_loader   = make_loader(val_ds,   args.batch_size, False, args.num_workers)
    test0_loader = make_loader(test0_ds, args.batch_size, False, args.num_workers)
    test1_loader = make_loader(test1_ds, args.batch_size, False, args.num_workers)
    test2_loader = make_loader(test2_ds, args.batch_size, False, args.num_workers)

    model = TransformerBinaryClassifier(
        max_len=args.max_len,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        val_vocab=val_vocab,
    ).to(device)

    print(f"[Params] {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop == "acc":
        best = -1.0
        is_better = lambda cur: cur > best + 1e-6
    else:
        best = float("inf")
        is_better = lambda cur: cur < best - 1e-6

    bad = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, global_step, hit = run_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=args.grad_clip,
            max_steps=args.max_steps, global_step=global_step
        )
        va_loss, va_acc, _, _ = run_epoch(
            model, val_loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=0.0
        )

        cur = va_acc if args.early_stop == "acc" else va_loss
        improved = is_better(cur)
        if improved:
            best = cur
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "global_step": global_step}, args.save_path)
        else:
            bad += 1

        print(
            f"Epoch {epoch:03d} | step={global_step:06d} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val   loss={va_loss:.4f} acc={va_acc*100:.2f}% | "
            f"best({args.early_stop})={best:.6f} bad={bad}/{args.patience}"
            f"{' [saved]' if improved else ''}"
        )

        if hit:
            print("Reached max_steps. Stopping.")
            break
        if bad >= args.patience:
            print("Early stopping (patience).")
            break

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("\n[Eval best checkpoint]")
    for name, loader in [("test_bin0", test0_loader), ("test_bin1", test1_loader), ("test_bin2", test2_loader)]:
        te_loss, te_acc, _, _ = run_epoch(model, loader, device, None, amp=args.amp, amp_dtype=args.amp_dtype, grad_clip=0.0)
        print(f"{name:9s} | loss={te_loss:.4f} acc={te_acc*100:.2f}%")

    print(f"\nSaved: {args.save_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
