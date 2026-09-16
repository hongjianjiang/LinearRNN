#!/usr/bin/env python3
# train_transformer_mm_query.py

from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Parsing + dataset
# ----------------------------

def parse_src_line(line: str) -> Tuple[int, int, int, List[List[int]]]:
    parts = line.strip().split("|")
    if len(parts) < 4:
        raise ValueError(f"bad src line: {line[:120]}")
    T = int(parts[0])
    m = int(parts[1])
    qk = int(parts[2])
    mats_str = parts[3:]
    if len(mats_str) != T:
        raise ValueError(f"declared T={T} but got {len(mats_str)} matrices")
    mats: List[List[int]] = []
    for s in mats_str:
        xs = [int(x) for x in s.split(",")]
        if len(xs) != 9:
            raise ValueError("matrix must have 9 ints")
        mats.append(xs)
    return T, m, qk, mats


def parse_tgt_line_stepwise(line: str, T: int) -> List[int]:
    parts = line.strip().split("|") if line.strip() else []
    if len(parts) != T:
        raise ValueError(f"bad tgt line: expected T={T} labels but got {len(parts)}")
    return [int(x) for x in parts]


class MMQueryStepwiseDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str):
        self.src_path = src_path
        self.tgt_path = tgt_path
        with open(src_path, "r") as f:
            self.src_lines = f.read().splitlines()
        with open(tgt_path, "r") as f:
            self.tgt_lines = f.read().splitlines()
        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError("src/tgt size mismatch")

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Dict:
        T, m, qk, mats = parse_src_line(self.src_lines[idx])
        y_list = parse_tgt_line_stepwise(self.tgt_lines[idx], T)
        return {"T": T, "m": m, "qk": qk, "mats": mats, "y_list": y_list}


def infer_stats_from_split(src_path: str) -> Tuple[int, int]:
    """
    Returns (max_T, max_m) for a src file.
    Format: T|m|qk|...
    """
    max_T = 0
    max_m = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) < 3:
                continue
            T = int(parts[0])
            m = int(parts[1])
            if T > max_T:
                max_T = T
            if m > max_m:
                max_m = m
    return max_T, max_m


def infer_stats_all_splits(data_dir: str, splits: List[str]) -> Tuple[int, int]:
    max_T_all = 0
    max_m_all = 0
    for sp in splits:
        src_path = os.path.join(data_dir, f"{sp}_src.txt")
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        max_T, max_m = infer_stats_from_split(src_path)
        max_T_all = max(max_T_all, max_T)
        max_m_all = max(max_m_all, max_m)
    return max_T_all, max_m_all


# ----------------------------
# Batch + collate
# ----------------------------

@dataclass
class Batch:
    x: torch.Tensor
    attn_mask: torch.Tensor
    y: torch.Tensor
    m: torch.Tensor
    T: torch.Tensor
    mat_mask: torch.Tensor


def collate_mmquery_stepwise(
    batch: List[Dict],
    feature_dim: int,
    ignore_index: int,
    m_global_max: int,
) -> Batch:
    """
    Uses GLOBAL modulus scale (m_global_max) for META/m_norm to be stable across batches.
    """
    B = len(batch)
    Ts = torch.tensor([b["T"] for b in batch], dtype=torch.long)
    maxT = int(Ts.max().item())
    L = 2 + maxT
    Fdim = feature_dim

    x = torch.zeros((B, L, Fdim), dtype=torch.float32)
    attn_mask = torch.zeros((B, L), dtype=torch.bool)
    m = torch.tensor([b["m"] for b in batch], dtype=torch.long)

    y = torch.full((B, maxT), fill_value=ignore_index, dtype=torch.long)
    mat_mask = torch.zeros((B, maxT), dtype=torch.bool)

    m_scale = max(1.0, float(m_global_max - 1))

    for i, b in enumerate(batch):
        T = b["T"]
        mi = int(b["m"])
        qk = int(b["qk"])

        attn_mask[i, : 2 + T] = True
        x[i, 0, -1] = 1.0  # BOS

        # META
        m_norm = (mi - 1.0) / m_scale
        qk_norm = qk / 8.0  # qk in [0..8]
        T_norm = (T - 1.0) / max(1.0, maxT - 1)

        meta = torch.tensor([m_norm, qk_norm, T_norm], dtype=torch.float32)
        x[i, 1, : meta.numel()] = meta

        # MAT tokens
        for t in range(T):
            vals = torch.tensor(b["mats"][t], dtype=torch.float32)
            denom = max(1.0, float(mi - 1))
            vals = (vals / denom) * 2.0 - 1.0  # [-1,1]
            pos = 2 + t
            x[i, pos, :9] = vals
            x[i, pos, 9] = m_norm
            x[i, pos, 10] = (t / max(1.0, T - 1.0))

        y_list = b["y_list"]
        y[i, :T] = torch.tensor(y_list, dtype=torch.long)
        mat_mask[i, :T] = True

    return Batch(x=x, attn_mask=attn_mask, y=y, m=m, T=Ts, mat_mask=mat_mask)


# ----------------------------
# Model
# ----------------------------

class TransformerEncoderStepwise(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, n_layers: int, n_heads: int,
                 dropout: float, max_len: int, max_mod: int):
        super().__init__()
        self.max_mod = max_mod
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, max_mod)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.in_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos)[None, :, :]
        h = self.dropout(h)
        pad_mask = ~attn_mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.dropout(h)
        return self.head(h)  # (B,L,C)


# ----------------------------
# Loss / Eval
# ----------------------------

def stepwise_masked_ce_loss(logits_mat: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                            ignore_index: int) -> torch.Tensor:
    B, Tm, C = logits_mat.shape

    # HARD SAFETY: if any modulus exceeds head size, training is invalid
    m_max_batch = int(m.max().item())
    if m_max_batch > C:
        raise RuntimeError(f"[FATAL] batch has m_max={m_max_batch} > max_mod(head)={C}. "
                           f"Set --max_mod >= {m_max_batch} (e.g. 300).")

    cls = torch.arange(C, device=logits_mat.device).view(1, 1, C)
    invalid = cls >= m.view(B, 1, 1)
    logits_masked = logits_mat.masked_fill(invalid, float("-inf"))

    return F.cross_entropy(
        logits_masked.reshape(B * Tm, C),
        y.reshape(B * Tm),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             ignore_index: int) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        x = batch.x.to(device)
        mask = batch.attn_mask.to(device)
        y = batch.y.to(device)
        m = batch.m.to(device)
        mat_mask = batch.mat_mask.to(device)

        logits = model(x, mask)
        logits_mat = logits[:, 2:, :]

        loss = stepwise_masked_ce_loss(logits_mat, y, m, ignore_index)

        B, Tm, C = logits_mat.shape
        cls = torch.arange(C, device=device).view(1, 1, C)
        invalid = cls >= m.view(B, 1, 1)
        pred = logits_mat.masked_fill(invalid, float("-inf")).argmax(dim=-1)

        valid = (y != ignore_index) & mat_mask
        correct = ((pred == y) & valid)

        total_loss += loss.item() * int(valid.sum().item())
        total_correct += int(correct.sum().item())
        total_count += int(valid.sum().item())

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, 100.0 * total_correct / total_count


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--auto_infer", action="store_true",
                    help="infer max_T and max_m from all splits and auto-set max_len/max_mod if not provided")
    ap.add_argument("--splits", type=str,
                    default="train,val_bin0,test_bin0,test_bin1,test_bin2")

    ap.add_argument("--max_mod", type=int, default=0,
                    help="classifier head size; set 0 to auto-use inferred max m")
    ap.add_argument("--max_len", type=int, default=0,
                    help="max tokens; set 0 to auto-use 2+inferred max_T")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--feature_dim", type=int, default=16)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")

    ap.add_argument("--save_path", type=str, default="ckpt_transformer_mmquery_stepwise.pt")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Auto-infer stats (recommended for random m)
    if args.auto_infer or args.max_mod == 0 or args.max_len == 0:
        max_T_all, max_m_all = infer_stats_all_splits(args.data_dir, split_list)
        print(f"[Auto] inferred m_max(all splits)={max_m_all}")
        print(f"[Auto] inferred max_T(all splits)={max_T_all} => max_len={2 + max_T_all}")

        if args.max_mod == 0:
            args.max_mod = max_m_all
        if args.max_len == 0:
            args.max_len = 2 + max_T_all

    print(f"[Config] max_mod(head)={args.max_mod} max_len={args.max_len}")

    ignore_index = -100

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = MMQueryStepwiseDataset(
            src_path=os.path.join(args.data_dir, f"{split}_src.txt"),
            tgt_path=os.path.join(args.data_dir, f"{split}_tgt.txt"),
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_mmquery_stepwise(
                b,
                feature_dim=args.feature_dim,
                ignore_index=ignore_index,
                m_global_max=args.max_mod,
            ),
            drop_last=False,
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val_bin0", shuffle=False)
    test0_loader = make_loader("test_bin0", shuffle=False)
    test1_loader = make_loader("test_bin1", shuffle=False)
    test2_loader = make_loader("test_bin2", shuffle=False)

    model = TransformerEncoderStepwise(
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
        max_len=args.max_len,
        max_mod=args.max_mod,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    best_val_acc = -1.0
    best_state = None

    step = 0
    model.train()
    it = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x = batch.x.to(device)
        mask = batch.attn_mask.to(device)
        y = batch.y.to(device)
        m = batch.m.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x, mask)
            logits_mat = logits[:, 2:, :]
            loss = stepwise_masked_ce_loss(logits_mat, y, m, ignore_index)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        step += 1

        if step % 200 == 0:
            print(f"step={step} loss={loss.item():.4f}")

        if step % args.eval_every == 0 or step == args.max_steps:
            vloss, vacc = evaluate(model, val_loader, device, ignore_index)
            print(f"[Val] step={step} loss/step={vloss:.6f} acc={vacc:.2f}% (stepwise)")

            if vacc > best_val_acc:
                best_val_acc = vacc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model": best_state, "args": vars(args)}, args.save_path)
                print(f"[Saved] {args.save_path} (best val acc={best_val_acc:.2f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    for name, loader in [("test_bin0", test0_loader), ("test_bin1", test1_loader), ("test_bin2", test2_loader)]:
        tloss, tacc = evaluate(model, loader, device, ignore_index)
        print(f"{name} | loss/step={tloss:.6f} acc={tacc:.2f}% (stepwise)")


if __name__ == "__main__":
    main()
