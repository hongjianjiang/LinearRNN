#!/usr/bin/env python3
"""
Train RNN/GRU/LSTM with STEPWISE binary supervision.

Dataset:
  src: "T|a1,...,a9|...|a1,...,a9"
  tgt: "y1|y2|...|yT" (0/1 per step)

Model:
  RNN over (T,9) -> per-step logits (T,1).
Loss:
  masked BCE over all valid steps.
Eval:
  - stepwise accuracy (all steps)
  - final accuracy (last step)
"""

from __future__ import annotations
import os, time, argparse
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_src_line(line: str) -> np.ndarray:
    parts = line.strip().split("|")
    T = int(parts[0])
    mats = parts[1:]
    if len(mats) != T:
        T = len(mats)
    x = np.zeros((T, 9), dtype=np.float32)
    for i, m in enumerate(mats):
        nums = m.split(",")
        x[i] = np.array([float(int(v)) for v in nums], dtype=np.float32)
    return x


def parse_tgt_stepwise(line: str, T: int) -> np.ndarray:
    parts = line.strip().split("|")
    if len(parts) != T:
        # tolerate mismatch by trunc/pad
        y = np.array([int(p) for p in parts], dtype=np.float32)
        if y.size < T:
            y = np.pad(y, (0, T - y.size), constant_values=0)
        else:
            y = y[:T]
        return y
    return np.array([int(p) for p in parts], dtype=np.float32)


class StepwiseDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str):
        with open(src_path) as f:
            self.src = [ln.rstrip("\n") for ln in f]
        with open(tgt_path) as f:
            self.tgt = [ln.rstrip("\n") for ln in f]
        if len(self.src) != len(self.tgt):
            raise ValueError("src/tgt length mismatch")

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = parse_src_line(self.src[idx])
        y = parse_tgt_stepwise(self.tgt[idx], T=x.shape[0])
        return x, y


def collate_pad(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    B = len(batch)
    Tm = int(lengths.max().item())
    x = torch.zeros((B, Tm, 9), dtype=torch.float32)
    y = torch.zeros((B, Tm), dtype=torch.float32)
    mask = torch.zeros((B, Tm), dtype=torch.float32)

    for i, (xi, yi) in enumerate(batch):
        t = xi.shape[0]
        x[i, :t] = torch.from_numpy(xi)
        y[i, :t] = torch.from_numpy(yi)
        mask[i, :t] = 1.0

    return x, y, mask


class RNNStepwise(nn.Module):
    def __init__(self, rnn_type: str, d_model: int, layers: int, dropout: float, bidir: bool):
        super().__init__()
        rnn_type = rnn_type.lower()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=9,
            hidden_size=d_model,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=bidir,
        )
        d_out = d_model * (2 if bidir else 1)
        self.head = nn.Linear(d_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B,T,9), mask: (B,T) 1 for valid
        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,dh)
        logits = self.head(out).squeeze(-1)  # (B,T)
        return logits


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits,y,mask: (B,T)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def eval_stepwise(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, amp_dtype: str) -> Dict[str, float]:
    model.eval()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    tot_loss = 0.0
    tot_mask = 0.0

    step_correct = 0.0
    step_total = 0.0

    final_correct = 0.0
    final_total = 0.0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp and device.type == "cuda"):
            logits = model(x, mask)
            loss = masked_bce_with_logits(logits, y, mask)

        tot_loss += loss.item() * mask.sum().item()
        tot_mask += mask.sum().item()

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()

        # stepwise acc
        step_correct += ((pred == y) * mask).sum().item()
        step_total += mask.sum().item()

        # final acc (last valid step)
        lengths = mask.sum(dim=1).long()
        idx = (lengths - 1).clamp_min(0)  # (B,)
        b = torch.arange(x.size(0), device=device)
        final_pred = pred[b, idx]
        final_y = y[b, idx]
        final_correct += (final_pred == final_y).float().sum().item()
        final_total += x.size(0)

    return {
        "loss": tot_loss / max(1.0, tot_mask),
        "step_acc": step_correct / max(1.0, step_total),
        "final_acc": final_correct / max(1.0, final_total),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, opt, device, amp: bool, amp_dtype: str, grad_clip: float) -> float:
    model.train()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda" and amp_dtype == "fp16"))

    tot = 0.0
    denom = 0.0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp and device.type == "cuda"):
            logits = model(x, mask)
            loss = masked_bce_with_logits(logits, y, mask)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        tot += loss.item() * mask.sum().item()
        denom += mask.sum().item()

    return tot / max(1.0, denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_path", default="ckpt_rnn_stepwise.pt")

    ap.add_argument("--rnn_type", choices=["rnn", "gru", "lstm"], default="lstm")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bidir", action="store_true")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--early_stop", choices=["final_acc", "step_acc", "loss"], default="final_acc")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--eval_tests", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[{now()}] device={device}")

    def paths(split: str) -> Tuple[str, str]:
        return (os.path.join(args.data_dir, f"{split}_src.txt"),
                os.path.join(args.data_dir, f"{split}_tgt.txt"))

    train_ds = StepwiseDataset(*paths("train"))
    val_ds = StepwiseDataset(*paths("val_bin0"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            collate_fn=collate_pad)

    test_loaders: Dict[str, DataLoader] = {}
    if args.eval_tests:
        for s in ["test_bin0", "test_bin1", "test_bin2"]:
            ds = StepwiseDataset(*paths(s))
            test_loaders[s] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                                         collate_fn=collate_pad)

    model = RNNStepwise(args.rnn_type, args.d_model, args.layers, args.dropout, args.bidir).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.rnn_type.upper()} d_model={args.d_model} layers={args.layers} bidir={args.bidir} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    key = args.early_stop
    best = -1e9 if key in ("final_acc", "step_acc") else 1e9
    best_epoch = -1
    bad = 0

    def better(v: float, b: float) -> bool:
        return (v > b) if key in ("final_acc", "step_acc") else (v < b)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, args.amp, args.amp_dtype, args.grad_clip)
        val = eval_stepwise(model, val_loader, device, args.amp, args.amp_dtype)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} "
              f"val_loss={val['loss']:.4f} val_step_acc={100*val['step_acc']:.2f}% "
              f"val_final_acc={100*val['final_acc']:.2f}% time={dt:.1f}s")

        if args.eval_tests:
            for name, loader in test_loaders.items():
                m = eval_stepwise(model, loader, device, args.amp, args.amp_dtype)
                print(f"  [{name}] loss={m['loss']:.4f} step_acc={100*m['step_acc']:.2f}% final_acc={100*m['final_acc']:.2f}%")

        v = float(val[key])
        if better(v, best):
            best = v
            best_epoch = epoch
            bad = 0
            torch.save({
                "args": vars(args),
                "epoch": epoch,
                "best_key": key,
                "best_val": best,
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
            }, args.save_path)
            print(f"  [save] best -> {args.save_path} ({key}={best:.6f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  [early-stop] best_epoch={best_epoch} best_{key}={best:.6f}")
                break

    print(f"[done] best_epoch={best_epoch} best_{key}={best:.6f} ckpt={args.save_path}")


if __name__ == "__main__":
    main()
