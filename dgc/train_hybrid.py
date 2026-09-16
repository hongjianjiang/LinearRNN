#!/usr/bin/env python3
# train_hybrid.py
"""
DeltaNet(+Transformer) training script for your two-bucket reachability TXT dataset.

Data format (one per line):
  src;i1-j1;i2-j2;...;tgt
(all are binary strings, edges sorted)

Files under --data_dir:
  train_src.txt / train_tgt.txt
  val_src_bin0.txt / val_tgt_bin0.txt
  val_src_bin1.txt / val_tgt_bin1.txt
  val_src_bin2.txt / val_tgt_bin2.txt

Model (hybrid):
  Char-embedding -> DeltaNet stack -> TransformerEncoder stack -> masked mean pool -> MLP head

Run (example):
  python3 train_hybrid.py --data_dir data/n20 --cuda --epochs 50 --batch_size 128 \
    --emb_dim 128 --hidden_size 256 --layers 2 --num_heads 8 --mode chunk --dropout 0.1 \
    --tf_layers 2 --tf_heads 8 --tf_ff_mult 4 --tf_dropout 0.1 --amp --max_steps 30000
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fla.layers.delta_net import DeltaNet  # type: ignore


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
                    f"Unexpected char {repr(ch)} in input: {s[:80]}... "
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
# DeltaNet stack
# --------------------------

class DeltaNetStack(nn.Module):
    """
    A simple stack of DeltaNet layers with dropout + residual + layernorm.

    We pass layer_idx=i to match your constructor snippet.
    """
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_heads: int,
        layers: int,
        dropout: float,
        use_short_conv: bool,
        conv_size: int,
        allow_neg_eigval: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(layers):
            self.layers.append(
                DeltaNet(
                    mode=mode,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    use_beta=True,
                    use_gate=True,
                    use_short_conv=use_short_conv,
                    conv_size=conv_size,
                    conv_bias=False,
                    allow_neg_eigval=allow_neg_eigval,
                    layer_idx=i,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H]
        for layer, ln in zip(self.layers, self.norms):
            y = layer(x)
            if isinstance(y, tuple):
                y = y[0]
            x = ln(x + self.drop(y))
        return x


# --------------------------
# Hybrid backbone: DeltaNet -> Transformer
# --------------------------

class HybridBackbone(nn.Module):
    """
    DeltaNet stack followed by a TransformerEncoder stack.
    """
    def __init__(
        self,
        deltanet: DeltaNetStack,
        hidden_size: int,
        tf_layers: int,
        tf_heads: int,
        tf_ff_mult: int,
        tf_dropout: float,
    ):
        super().__init__()
        self.deltanet = deltanet
        self.tf_layers = int(tf_layers)

        if self.tf_layers <= 0:
            self.tf: Optional[nn.Module] = None
        else:
            ff_dim = int(hidden_size * tf_ff_mult)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=tf_heads,
                dim_feedforward=ff_dim,
                dropout=tf_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.tf = nn.TransformerEncoder(enc_layer, num_layers=self.tf_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H]
        h = self.deltanet(x)

        if self.tf is None:
            return h

        B, T, _ = h.shape
        device = h.device
        # True = pad positions for TransformerEncoder
        pad_mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]  # [B,T]
        h = self.tf(h, src_key_padding_mask=pad_mask)
        return h


# --------------------------
# Classifier
# --------------------------

class DeltaNetClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int,
        hidden_size: int,
        layers: int,
        num_heads: int,
        dropout: float,
        mode: str,
        use_short_conv: bool,
        conv_size: int,
        allow_neg_eigval: bool,
        # Transformer-on-top
        tf_layers: int = 0,
        tf_heads: int = 8,
        tf_ff_mult: int = 4,
        tf_dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.in_proj = nn.Linear(emb_dim, hidden_size) if emb_dim != hidden_size else nn.Identity()

        deltanet = DeltaNetStack(
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            allow_neg_eigval=allow_neg_eigval,
        )

        self.backbone = HybridBackbone(
            deltanet=deltanet,
            hidden_size=hidden_size,
            tf_layers=tf_layers,
            tf_heads=tf_heads,
            tf_ff_mult=tf_ff_mult,
            tf_dropout=tf_dropout,
        )

        self.pool_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x_tok: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_tok: [B,T], lengths: [B]
        x = self.emb(x_tok)            # [B,T,emb]
        x = self.in_proj(x)            # [B,T,H]
        h = self.backbone(x, lengths)  # [B,T,H]

        # masked mean pool
        B, T, H = h.shape
        device = h.device
        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float().unsqueeze(-1)  # [B,T,1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B,H]
        pooled = self.pool_norm(pooled)
        return self.head(pooled)       # [B,2]


# --------------------------
# Train/Eval
# --------------------------

def pick_amp_dtype(device: torch.device) -> torch.dtype:
    # DeltaNet tends to want bf16 on CUDA; fall back to fp16 if unsupported.
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    tot = 0
    cor = 0
    loss_sum = 0.0

    amp_dtype = pick_amp_dtype(device)

    for x, lens, y in loader:
        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda"), dtype=amp_dtype):
            logits = model(x, lens)
            loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=-1)
        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

    return {"loss": loss_sum / max(1, tot), "acc": cor / max(1, tot)}


@dataclass
class EarlyStopper:
    patience: int
    best: float = -1.0
    bad: int = 0

    def step(self, metric: float) -> bool:
        if metric > self.best:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--mode", type=str, default="chunk")

    ap.add_argument("--use_short_conv", action="store_true")
    ap.add_argument("--conv_size", type=int, default=4)
    ap.add_argument("--allow_neg_eigval", action="store_true")

    # transformer-on-top args
    ap.add_argument("--tf_layers", type=int, default=2)
    ap.add_argument("--tf_heads", type=int, default=8)
    ap.add_argument("--tf_ff_mult", type=int, default=4)
    ap.add_argument("--tf_dropout", type=float, default=0.1)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--save_path", type=str, default="ckpt_deltanet_two_bucket.pt")

    ap.add_argument("--num_workers", type=int, default=2)

    # NEW: max steps
    ap.add_argument(
        "--max_steps",
        type=int,
        default=30000,
        help="Hard cap on optimizer update steps (global).",
    )

    # NEW: step log interval
    ap.add_argument(
        "--log_every_steps",
        type=int,
        default=500,
        help="Print a short training log every N optimizer steps (0 disables).",
    )

    args = ap.parse_args()

    # seed
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_pad(b, tok.pad_id),
    )

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_pad(b, tok.pad_id),
        )

    val0_loader = make_loader(val0_ds)
    val1_loader = make_loader(val1_ds)
    val2_loader = make_loader(val2_ds)

    model = DeltaNetClassifier(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mode=args.mode,
        use_short_conv=args.use_short_conv,
        conv_size=args.conv_size,
        allow_neg_eigval=args.allow_neg_eigval,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        tf_ff_mult=args.tf_ff_mult,
        tf_dropout=args.tf_dropout,
    ).to(device)

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[Params] {n_params/1e6:.2f}M")
    print(model.__class__.__name__)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    stopper = EarlyStopper(args.patience)
    best_state = None

    # NEW: global step
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        tot = 0
        cor = 0
        loss_sum = 0.0

        amp_dtype = pick_amp_dtype(device)

        for x, lens, y in train_loader:
            # NEW: hard cap on optimizer updates
            if global_step >= args.max_steps:
                print(f"[MaxSteps] Reached {global_step}/{args.max_steps} steps. Stopping training.")
                break

            x = x.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(args.amp and device.type == "cuda"),
                dtype=amp_dtype,
            ):
                logits = model(x, lens)
                loss = F.cross_entropy(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # NEW: increment + print step
            global_step += 1
            if args.log_every_steps > 0 and (global_step == 1 or global_step % args.log_every_steps == 0):
                print(f"[Step {global_step:06d}/{args.max_steps}] loss={loss.item():.4f}")

            pred = logits.argmax(dim=-1)
            cor += int((pred == y).sum().item())
            tot += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())

        train_loss = loss_sum / max(1, tot)
        train_acc = cor / max(1, tot)

        v0 = evaluate(model, val0_loader, device, amp=args.amp)
        v1 = evaluate(model, val1_loader, device, amp=args.amp)
        v2 = evaluate(model, val2_loader, device, amp=args.amp)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | steps={global_step}/{args.max_steps} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val0 acc={v0['acc']*100:.2f}% loss={v0['loss']:.4f} | "
            f"val1 acc={v1['acc']*100:.2f}% loss={v1['loss']:.4f} | "
            f"val2 acc={v2['acc']*100:.2f}% loss={v2['loss']:.4f} | "
            f"time={dt:.1f}s"
        )

        # checkpoint on val0 (in-distribution)
        if v0["acc"] >= stopper.best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args)}, args.save_path)

        if stopper.step(v0["acc"]):
            print(f"[EarlyStop] best val0={stopper.best*100:.2f}%")
            break

        # NEW: also stop after epoch if max_steps reached mid-epoch
        if global_step >= args.max_steps:
            break

    print("[✓] Saved checkpoint:", args.save_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        f0 = evaluate(model, val0_loader, device, amp=args.amp)
        f1 = evaluate(model, val1_loader, device, amp=args.amp)
        f2 = evaluate(model, val2_loader, device, amp=args.amp)
        final_str = (
            f"[Final best@val0] data_dir={args.data_dir} | "
            f"val0={f0['acc']*100:.2f}% | "
            f"val1={f1['acc']*100:.2f}% | val2={f2['acc']*100:.2f}%"
        )
        print(final_str)
        with open("final_eval_hybrid.log", "a", encoding="utf-8") as logf:
            logf.write(final_str + "\n")


if __name__ == "__main__":
    main()
