#!/usr/bin/env python3
# train_mamba.py
"""
Mamba training script (GPU-friendly) for your two-bucket reachability dataset.

Keeps the same dataset format + stability/engineering features as your RNN script:
- Fix 1 (train+eval): detects NaN/Inf logits, nan_to_num() before CE, counts nan batches in eval.
- Fix 2: proper checkpointing via best_v0 tracking.
- Prints global optimizer step periodically; hard-stops immediately when max_steps is reached.
- Adds --log_every_steps.

Data format (one per line):
  src;i1-j1;i2-j2;...;tgt

Expected files under --data_dir:
  train_src.txt / train_tgt.txt
  val_src_bin0.txt / val_tgt_bin0.txt
  val_src_bin1.txt / val_tgt_bin1.txt
  val_src_bin2.txt / val_tgt_bin2.txt

Run (example):
  python3 -u train_mamba.py --data_dir data/n100 --cuda \
    --epochs 999 --max_steps 30000 --batch_size 64 \
    --d_model 256 --depth 4 --dropout 0.1 --lr 3e-4 --grad_clip 1.0 \
    --log_every_steps 500 --amp \
    --num_workers 2 --prefetch_factor 4 --persistent_workers
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Mamba import (with a clear error)
# --------------------------
try:
    # Most common import path
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception as e:
    Mamba = None
    _MAMBA_IMPORT_ERR = e


# --------------------------
# Tokenizer (char-level)
# --------------------------

class CharTokenizer:
    """
    Vocabulary: 0,1,;,- plus PAD.
    """
    def __init__(self):
        self.pad = "<PAD>"
        self.vocab = [self.pad, "0", "1", ";", "-"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.pad_id = self.stoi[self.pad]

    def encode(self, s: str) -> List[int]:
        s = s.strip()
        ids: List[int] = []
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(
                    f"Unexpected char {repr(ch)} in input: {s[:120]}... "
                    f"Allowed: {self.vocab[1:]}"
                )
            ids.append(self.stoi[ch])
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# --------------------------
# Dataset
# --------------------------

class TxtPairDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, tok: CharTokenizer):
        self.samples: List[Tuple[List[int], int]] = []
        with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
            for s, y in zip(f_src, f_tgt):
                x = tok.encode(s)
                label = int(y.strip())
                self.samples.append((x, label))
        if not self.samples:
            raise RuntimeError(f"Empty dataset: {src_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_pad(batch: List[Tuple[List[int], int]], pad_id: int):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, T), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        x_pad[i, :len(x)] = torch.tensor(x, dtype=torch.long)

    y = torch.tensor(ys, dtype=torch.long)
    return x_pad, lengths, y


# --------------------------
# Mamba encoder blocks
# --------------------------

class MambaBlock(nn.Module):
    """
    Pre-norm residual Mamba block:
      x <- x + Dropout(Mamba(LN(x)))
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not installed or import failed. Install with: pip install mamba-ssm\n"
                f"Import error: {repr(_MAMBA_IMPORT_ERR)}"
            )

        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = self.mamba(self.ln(x))
        return x + self.drop(y)


class MambaClassifier(nn.Module):
    """
    Char embedding -> Mamba stack -> masked mean pool -> head (2 classes)

    Padding handling:
    - pad tokens embeddings are zeroed
    - after each block, we zero out padded positions (cheap, avoids any weird drift)
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.pad_id = int(pad_id)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.in_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(int(depth))
        ])

        self.pool_ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x_tok: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_tok: (B,T)
        B, T = x_tok.shape
        device = x_tok.device

        # mask: (B,T,1) float
        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float().unsqueeze(-1)

        x = self.emb(x_tok)          # (B,T,D)
        x = self.in_drop(x)

        # zero pads explicitly (embedding padding_idx already zeros them, but keep it explicit)
        x = x * mask

        for blk in self.blocks:
            x = blk(x)
            x = x * mask  # keep padded positions at 0

        # masked mean pool
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # (B,D)
        pooled = self.pool_ln(pooled)
        return self.head(pooled)  # (B,2)


# --------------------------
# Eval (Fix 1): NaN/Inf logits guard
# --------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    tot = 0
    cor = 0
    loss_sum = 0.0
    nan_batches = 0

    for x, lens, y in loader:
        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, lens)

        if not torch.isfinite(logits).all():
            nan_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=-1)
        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

    out = {"loss": loss_sum / max(1, tot), "acc": cor / max(1, tot)}
    if nan_batches > 0:
        out["nan_batches"] = float(nan_batches)
    return out


# --------------------------
# Early stop: consecutive perfect val0
# --------------------------

@dataclass
class ConsecPerfectStopper:
    need: int
    streak: int = 0

    def step(self, acc: float) -> bool:
        if acc >= 1.0 - 1e-12:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.need


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Train
# --------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp: bool,
    grad_clip: float,
    max_steps: int,
    global_step: int,
    log_every_steps: int,
) -> Tuple[float, float, int, bool]:
    """
    Returns: (train_loss, train_acc, new_global_step, reached_max_steps)
    """
    model.train()
    tot = 0
    cor = 0
    loss_sum = 0.0
    reached = False

    for step, (x, lens, y) in enumerate(loader, start=1):
        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x, lens)

            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            loss = F.cross_entropy(logits, y)

        if not torch.isfinite(loss):
            mx = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
            raise RuntimeError(f"Non-finite loss at global_step={global_step} step={step}, max|logit|={mx}")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        global_step += 1

        if log_every_steps > 0 and (global_step == 1 or global_step % log_every_steps == 0):
            print(f"[Step {global_step:06d}/{max_steps if max_steps > 0 else -1}] loss={loss.item():.4f}")

        pred = logits.argmax(dim=-1)
        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

        if step % 400 == 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and (not torch.isfinite(p).all()):
                        raise RuntimeError(f"Non-finite parameter: {n}")

    train_loss = loss_sum / max(1, tot)
    train_acc = cor / max(1, tot)
    return train_loss, train_acc, global_step, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data/n50")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=999)

    ap.add_argument("--max_steps", type=int, default=30000, help="Hard cap on optimizer updates (batches). 0=unlimited")
    ap.add_argument("--log_every_steps", type=int, default=500, help="Print step log every N optimizer steps (0=off)")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Mamba sizes
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)

    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--stop_consec_100", type=int, default=3, help="Stop after N consecutive epochs with val0=100%.")
    ap.add_argument("--save_path", type=str, default="ckpt_mamba_two_bucket.pt")

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--amp", action="store_true", help="Optional AMP. Still guards logits for stability.")

    args = ap.parse_args()

    if Mamba is None:
        raise ImportError(
            "mamba-ssm is required for this script.\n"
            "Install with: pip install mamba-ssm\n"
            f"Import error: {repr(_MAMBA_IMPORT_ERR)}"
        )

    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    tok = CharTokenizer()
    print("[Vocab]", tok.vocab, "size=", tok.vocab_size)

    def p(name: str) -> str:
        return os.path.join(args.data_dir, name)

    train_ds = TxtPairDataset(p("train_src.txt"), p("train_tgt.txt"), tok)
    val0_ds  = TxtPairDataset(p("val_src_bin0.txt"), p("val_tgt_bin0.txt"), tok)
    val1_ds  = TxtPairDataset(p("val_src_bin1.txt"), p("val_tgt_bin1.txt"), tok)
    val2_ds  = TxtPairDataset(p("val_src_bin2.txt"), p("val_tgt_bin2.txt"), tok)

    lens = [len(x) for x, _ in train_ds.samples[:2000]]
    print(f"[Len] sample avg={sum(lens)/len(lens):.1f} max={max(lens)} (first 2000)")

    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=lambda b: collate_pad(b, tok.pad_id),
    )

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
            collate_fn=lambda b: collate_pad(b, tok.pad_id),
        )

    val0_loader = make_loader(val0_ds)
    val1_loader = make_loader(val1_ds)
    val2_loader = make_loader(val2_ds)

    model = MambaClassifier(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        d_model=args.d_model,
        depth=args.depth,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[Params] {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    stopper = ConsecPerfectStopper(args.stop_consec_100)

    best_v0 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, global_step, hit = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            opt=opt,
            scaler=scaler,
            amp=args.amp,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            global_step=global_step,
            log_every_steps=args.log_every_steps,
        )

        v0 = evaluate(model, val0_loader, device)
        v1 = evaluate(model, val1_loader, device)
        v2 = evaluate(model, val2_loader, device)

        dt = time.time() - t0

        extra0 = f" nanB={int(v0['nan_batches'])}" if "nan_batches" in v0 else ""
        extra1 = f" nanB={int(v1['nan_batches'])}" if "nan_batches" in v1 else ""
        extra2 = f" nanB={int(v2['nan_batches'])}" if "nan_batches" in v2 else ""

        print(
            f"Epoch {epoch:03d} | step={global_step}/{args.max_steps} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val0 acc={v0['acc']*100:.2f}% loss={v0['loss']:.4f}{extra0} | "
            f"val1 acc={v1['acc']*100:.2f}% loss={v1['loss']:.4f}{extra1} | "
            f"val2 acc={v2['acc']*100:.2f}% loss={v2['loss']:.4f}{extra2} | "
            f"time={dt:.1f}s"
        )

        # Fix 2: checkpoint on best val0
        if v0["acc"] >= best_v0 - 1e-12:
            best_v0 = float(v0["acc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "args": vars(args), "best_v0": best_v0, "global_step": global_step},
                args.save_path,
            )
            print(f"  [saved] best_v0={best_v0*100:.2f}% -> {args.save_path}")

        if stopper.step(v0["acc"]):
            print(f"[EarlyStop] val0 hit 100% for {stopper.need} consecutive epochs.")
            break

        if hit:
            print(f"[MaxSteps] Reached global_step={global_step} (cap={args.max_steps}).")
            break

    print("[✓] Saved checkpoint:", args.save_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        f0 = evaluate(model, val0_loader, device)
        f1 = evaluate(model, val1_loader, device)
        f2 = evaluate(model, val2_loader, device)
        final_str = (
            f"[Final best@val0] data_dir={args.data_dir} | "
            f"val0={f0['acc']*100:.2f}% | "
            f"val1={f1['acc']*100:.2f}% | val2={f2['acc']*100:.2f}%"
        )
        print(final_str)
        with open("final_eval_mamba.log", "a", encoding="utf-8") as logf:
            logf.write(final_str + "\n")


if __name__ == "__main__":
    main()
