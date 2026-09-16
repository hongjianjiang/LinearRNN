#!/usr/bin/env python3
# train_mamba_nomod_final_binary.py
"""
Mamba baseline for NO-MOD 3x3 matrix multiplication with FINAL BINARY target.

FORMAT (from your gen.py):
  src: "n|a1,...,a9|a1,...,a9|...|a1,...,a9"
       where n == number of matrices (T)
  tgt: "0" or "1"

Label:
  y = 1 iff (P_T)[0,0] == 0
  P_T = M1 @ ... @ MT   (over Z)

Tokenization (no teacher forcing states):
  [BOS] + [MAT] * T
  length L = 1 + T

Model:
  TokenEncoder -> Mamba blocks -> LN -> (per-token) logit
  We supervise ONLY the final MAT token (position 1+T-1), using BCEWithLogits.

Speed:
  - preload + pre-tokenize once
  - optional length grouping for Mamba forward (pack=group)
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

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


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_src_nomod(line: str) -> Tuple[int, np.ndarray]:
    """
    Parse: "n|mat1|...|matT"
    mat: "a1,...,a9" (row-major)
    Returns: (T, mats_raw) where mats_raw: (T,9) int64
    """
    parts = line.strip().split("|")
    if len(parts) < 2:
        raise ValueError(f"Bad src (need >=2 fields): {line[:120]}")
    T = int(parts[0])
    mats_parts = parts[1:]
    if len(mats_parts) != T:
        raise ValueError(f"Bad src: header T={T} but got {len(mats_parts)} mats")
    mats = np.empty((T, 9), dtype=np.int64)
    for t, b in enumerate(mats_parts):
        xs = b.split(",")
        if len(xs) != 9:
            raise ValueError(f"Bad mat block at t={t}: got {len(xs)}")
        mats[t] = np.fromiter((int(v) for v in xs), dtype=np.int64, count=9)
    return T, mats


def parse_tgt_binary(line: str) -> int:
    s = line.strip()
    y = int(s)
    if y not in (0, 1):
        raise ValueError(f"Bad tgt (need 0/1): {line[:50]}")
    return y


def infer_max_T_from_dir(data_dir: str, splits: List[str]) -> int:
    mxT = 0
    found = False
    checked = []
    for sp in splits:
        p = os.path.join(data_dir, f"{sp}_src.txt")
        checked.append(p)
        if not os.path.exists(p):
            continue
        found = True
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    T = int(ln.split("|", 1)[0])
                except Exception:
                    continue
                mxT = max(mxT, T)
    if not found:
        raise ValueError("No '*_src.txt' found. Checked:\n  " + "\n  ".join(checked))
    if mxT <= 0:
        raise ValueError(f"Failed to infer max_T from {data_dir} (mxT={mxT})")
    return mxT


# -------------------------
# dataset (preloaded)
# -------------------------
class PreloadedNoModFinalBinaryDataset(Dataset):
    """
    Preload:
      - mats_raw: (T,9) int64
      - y: scalar 0/1
      - T, lengths (L=1+T)
    """
    def __init__(self, src_path: str, tgt_path: str, alphabet: str = "pm1"):
        src_lines = read_lines(src_path)
        tgt_lines = read_lines(tgt_path)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src/tgt mismatch: {len(src_lines)} vs {len(tgt_lines)}")

        self.Ts: List[int] = []
        self.lengths: List[int] = []
        self.mats: List[np.ndarray] = []
        self.y: List[int] = []

        it = list(zip(src_lines, tgt_lines))
        for i, (src, tgt) in enumerate(tqdm(it, desc=f"Preload {os.path.basename(src_path)}", dynamic_ncols=True)):
            T, mats = parse_src_nomod(src)  # (T,9)
            y = parse_tgt_binary(tgt)

            if alphabet == "pm1":
                ok = np.all((mats == -1) | (mats == 0) | (mats == 1))
                if not ok:
                    bad = mats[(mats != -1) & (mats != 0) & (mats != 1)]
                    raise ValueError(f"Alphabet mismatch at line {i}: e.g. {bad[:10].tolist()}")
            elif alphabet == "01":
                ok = np.all((mats == 0) | (mats == 1))
                if not ok:
                    raise ValueError(f"Alphabet mismatch at line {i}: expected 0/1")
            else:
                raise ValueError(f"Unknown alphabet: {alphabet}")

            self.Ts.append(int(T))
            self.lengths.append(int(1 + T))  # BOS + T mats
            self.mats.append(mats.astype(np.int64, copy=False))
            self.y.append(int(y))

    def __len__(self) -> int:
        return len(self.Ts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "T": self.Ts[idx],
            "L": self.lengths[idx],
            "mats": self.mats[idx],  # (T,9)
            "y": self.y[idx],        # int 0/1
        }


# -------------------------
# collate
# -------------------------
TOK_PAD = 0
TOK_BOS = 1
TOK_MAT = 2
NUM_TOKEN_TYPES = 3


@dataclass
class Batch:
    tok_type: torch.Tensor   # (B,Lmax)
    tok_val: torch.Tensor    # (B,Lmax,9) long (only MAT positions filled)
    lengths: torch.Tensor    # (B,)
    Ts: torch.Tensor         # (B,)
    y: torch.Tensor          # (B,) float32
    last_ix: torch.Tensor    # (B,) long (index in token sequence for FINAL MAT)


def collate_batch(items: List[dict]) -> Batch:
    B = len(items)
    Ts = torch.tensor([int(it["T"]) for it in items], dtype=torch.long)
    lengths = torch.tensor([int(it["L"]) for it in items], dtype=torch.long)
    Lmax = int(lengths.max().item())

    tok_type = torch.full((B, Lmax), TOK_PAD, dtype=torch.long)
    tok_val = torch.zeros((B, Lmax, 9), dtype=torch.long)

    y = torch.tensor([float(it["y"]) for it in items], dtype=torch.float32)
    last_ix = torch.zeros((B,), dtype=torch.long)

    for b, it in enumerate(items):
        T = int(it["T"])
        L = int(it["L"])
        mats = it["mats"]  # (T,9)

        tok_type[b, 0] = TOK_BOS
        if T > 0:
            tok_type[b, 1:L] = TOK_MAT
            tok_val[b, 1:L] = torch.from_numpy(mats).long()
            last_ix[b] = L - 1  # final MAT token
        else:
            # should not happen for your generators, but keep safe:
            last_ix[b] = 0

    return Batch(tok_type=tok_type, tok_val=tok_val, lengths=lengths, Ts=Ts, y=y, last_ix=last_ix)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


def _group_by_length_indices(lengths: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    lens = lengths.detach().to("cpu", non_blocking=True).tolist()
    buckets: Dict[int, List[int]] = {}
    for i, L in enumerate(lens):
        buckets.setdefault(int(L), []).append(i)
    out = []
    for L in sorted(buckets.keys()):
        out.append((L, torch.tensor(buckets[L], dtype=torch.long, device=lengths.device)))
    return out


# -------------------------
# token encoder
# -------------------------
class TokenValMLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, act: str = "gelu"):
        super().__init__()
        h = hidden_mult * d_model
        self.fc1 = nn.Linear(9, h)
        self.fc2 = nn.Linear(h, d_model)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, v9: torch.Tensor) -> torch.Tensor:
        x = v9.to(torch.float32)
        # stable magnitude compression for ints
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        x = self.fc2(self.act(self.fc1(x)))
        return self.ln(x)


class TokenEncoderNoMod(nn.Module):
    """
    x = type_emb + pos_emb + mat_mlp(M_t)
    """
    def __init__(self, max_len: int, d_model: int, dropout: float, mlp_hidden_mult: int, mlp_act: str):
        super().__init__()
        self.max_len = int(max_len)
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.mat_mlp = TokenValMLP(d_model=d_model, hidden_mult=mlp_hidden_mult, act=mlp_act)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, tok_val: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len}. Increase --max_len.")
        pos = torch.arange(L, device=tok_type.device).unsqueeze(0)
        x = self.type_emb(tok_type) + self.pos_emb(pos)

        mat_mask = (tok_type == TOK_MAT).unsqueeze(-1).to(x.dtype)
        x = x + self.mat_mlp(tok_val) * mat_mask
        return self.drop(x)


# -------------------------
# model (Mamba final binary)
# -------------------------
class MambaNoModFinalBinary(nn.Module):
    """
    Forward returns per-token logits: (B,Lmax)
    We supervise only at last_ix (final MAT token).
    """
    def __init__(
        self,
        max_len: int,
        d_model: int,
        layers: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        use_fast_path: bool,
        pack: str,
        mlp_hidden_mult: int,
        mlp_act: str,
    ):
        super().__init__()
        self.pack = pack

        self.enc = TokenEncoderNoMod(
            max_len=max_len,
            d_model=d_model,
            dropout=dropout,
            mlp_hidden_mult=mlp_hidden_mult,
            mlp_act=mlp_act,
        )

        from mamba_ssm import Mamba
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=use_fast_path)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)

    def forward(self, tok_type: torch.Tensor, tok_val: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.enc(tok_type, tok_val)  # (B,Lmax,d)
        B, Lmax, _ = x.shape

        if self.pack == "none":
            x = x * (tok_type != TOK_PAD).to(x.dtype).unsqueeze(-1)
            x = self._run_blocks(x)
        elif self.pack == "group":
            out = torch.zeros_like(x)
            buckets = _group_by_length_indices(lengths.to(torch.long))
            for L, idx in buckets:
                xs = x.index_select(0, idx)[:, :L, :]          # (nb, L, d)
                xs = self._run_blocks(xs)                      # (nb, L, d)

                # write back safely
                out[idx, :L, :] = xs
            x = out

        else:
            raise ValueError("pack must be 'none' or 'group'")

        logits = self.head(x).squeeze(-1)  # (B,Lmax)
        return logits


# -------------------------
# loss / metrics (final token)
# -------------------------
def final_bce_loss_and_acc(logits_bl: torch.Tensor, y_b: torch.Tensor, last_ix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = logits_bl.size(0)
    picked = logits_bl[torch.arange(B, device=logits_bl.device), last_ix]  # (B,)
    loss = F.binary_cross_entropy_with_logits(picked, y_b, reduction="mean")
    pred = (torch.sigmoid(picked) >= 0.5).to(torch.float32)
    acc = (pred == y_b).to(torch.float32).mean()
    return loss, acc


@torch.no_grad()
def eval_final(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, amp_dtype: str) -> dict:
    was_training = model.training
    model.eval()

    use_amp = amp and (device.type == "cuda")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        tok_type = batch.tok_type.to(device, non_blocking=True)
        tok_val = batch.tok_val.to(device, non_blocking=True)
        lengths = batch.lengths.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        last_ix = batch.last_ix.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits = model(tok_type, tok_val, lengths)
            loss, acc = final_bce_loss_and_acc(logits, y, last_ix)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        n += 1

    if was_training:
        model.train()

    return {"loss": total_loss / max(1, n), "acc": total_acc / max(1, n)}


def train_epoch(model, loader, device, opt, amp, amp_dtype, grad_clip, max_steps, global_step):
    model.train()
    use_amp = amp and (device.type == "cuda")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_fp16 = use_amp and (amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_loss = 0.0
    total_acc = 0.0
    n = 0
    hit = False

    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for batch in pbar:
        if max_steps > 0 and global_step >= max_steps:
            hit = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        tok_val = batch.tok_val.to(device, non_blocking=True)
        lengths = batch.lengths.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        last_ix = batch.last_ix.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits = model(tok_type, tok_val, lengths)
            loss, acc = final_bce_loss_and_acc(logits, y, last_ix)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

        if use_fp16:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        global_step += 1
        total_loss += float(loss.item())
        total_acc += float(acc.item())
        n += 1
        pbar.set_postfix(loss=total_loss / max(1, n), acc=f"{100*(total_acc/max(1,n)):.1f}%", gstep=global_step)

    return (total_loss / max(1, n), total_acc / max(1, n), global_step, hit)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alphabet", type=str, default="pm1", choices=["pm1", "01"])

    ap.add_argument("--splits", type=str, default="train,val_bin0,test_bin0,test_bin1,test_bin2")
    ap.add_argument("--max_len", type=int, default=0, help="0 => infer (1+max_T)")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--mlp_hidden_mult", type=int, default=4)
    ap.add_argument("--mlp_act", type=str, default="gelu", choices=["gelu", "relu"])

    ap.add_argument("--mamba_d_state", type=int, default=64)
    ap.add_argument("--mamba_d_conv", type=int, default=4)
    ap.add_argument("--mamba_expand", type=int, default=2)
    ap.add_argument("--mamba_fast_path", action="store_true")
    ap.add_argument("--pack", type=str, default="group", choices=["none", "group"])

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
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--save_path", type=str, default="ckpt_mamba_nomod_final_binary.pt")
    ap.add_argument("--early_stop", type=str, default="acc", choices=["loss", "acc"])
    ap.add_argument("--max_steps", type=int, default=30000)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    max_T = infer_max_T_from_dir(args.data_dir, split_list)

    if args.max_len <= 0:
        args.max_len = 1 + max_T
        print(f"[Auto] max_T={max_T} => max_len={args.max_len}")

    def sp(split: str) -> Tuple[str, str]:
        return (os.path.join(args.data_dir, f"{split}_src.txt"),
                os.path.join(args.data_dir, f"{split}_tgt.txt"))

    train_ds = PreloadedNoModFinalBinaryDataset(*sp("train"), alphabet=args.alphabet)
    val_ds   = PreloadedNoModFinalBinaryDataset(*sp("val_bin0"), alphabet=args.alphabet)
    t0_ds    = PreloadedNoModFinalBinaryDataset(*sp("test_bin0"), alphabet=args.alphabet)
    t1_ds    = PreloadedNoModFinalBinaryDataset(*sp("test_bin1"), alphabet=args.alphabet)
    t2_ds    = PreloadedNoModFinalBinaryDataset(*sp("test_bin2"), alphabet=args.alphabet)

    pin = (device.type == "cuda")
    train_loader = make_loader(train_ds, args.batch_size, True,  args.num_workers, pin)
    val_loader   = make_loader(val_ds,   args.batch_size, False, args.num_workers, pin)
    t0_loader    = make_loader(t0_ds,    args.batch_size, False, args.num_workers, pin)
    t1_loader    = make_loader(t1_ds,    args.batch_size, False, args.num_workers, pin)
    t2_loader    = make_loader(t2_ds,    args.batch_size, False, args.num_workers, pin)

    model = MambaNoModFinalBinary(
        max_len=args.max_len,
        d_model=args.d_model,
        layers=args.layers,
        dropout=args.dropout,
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand=args.mamba_expand,
        use_fast_path=args.mamba_fast_path,
        pack=args.pack,
        mlp_hidden_mult=args.mlp_hidden_mult,
        mlp_act=args.mlp_act,
    ).to(device)

    print(f"[Device] {device}")
    print(f"[Task] NO-MOD final binary: predict y at FINAL MAT token")
    print(f"[Data] {args.data_dir} max_T={max_T} max_len={args.max_len} alphabet={args.alphabet}")
    print(f"[Model] Mamba d_model={args.d_model} layers={args.layers} pack={args.pack}")
    print(f"[Params] {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop == "loss":
        best = float("inf")
        better = lambda cur: cur < best - 1e-6
    else:
        best = -1.0
        better = lambda cur: cur > best + 1e-12

    bad = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, global_step, hit = train_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps, global_step=global_step,
        )

        va = eval_final(model, val_loader, device, amp=args.amp, amp_dtype=args.amp_dtype)
        cur = va["loss"] if args.early_stop == "loss" else va["acc"]

        improved = better(cur)
        if improved:
            best = cur
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "global_step": global_step}, args.save_path)
        else:
            bad += 1

        print(
            f"Epoch {epoch:03d} | step={global_step:06d} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val loss={va['loss']:.4f} acc={va['acc']*100:.2f}% | "
            f"best({args.early_stop})={best:.6f} bad={bad}/{args.patience}"
            f"{' [saved]' if improved else ''}"
        )

        if hit:
            print(f"Reached max_steps={args.max_steps}. Stopping.")
            break
        if bad >= args.patience:
            print("Early stopping (patience).")
            break

    # eval best
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\n[Eval best checkpoint]")
    for name, loader in [("test_bin0", t0_loader), ("test_bin1", t1_loader), ("test_bin2", t2_loader)]:
        te = eval_final(model, loader, device, amp=args.amp, amp_dtype=args.amp_dtype)
        print(f"{name:9s} | loss={te['loss']:.4f} acc={te['acc']*100:.2f}%")

    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
