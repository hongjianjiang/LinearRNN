#!/usr/bin/env python3
# train_transformer_honest.py
"""
"HONEST" Transformer baseline for deterministic graph connectivity (two-bucket reachability).

Key changes vs your original encoder+meanpool:
  1) CAUSAL attention mask (no peeking to the right).
  2) NO masked mean pooling. Classify from LAST REAL TOKEN (or EOS if enabled).

Fix included (important):
  - Robustly checks max sequence length across TRAIN + VAL0/1/2 (not just first 2000),
    preventing CUDA Indexing.cu out-of-bounds crashes in positional embedding.

Data format (one per line):
  src;i1-j1;i2-j2;...;tgt

    (characters are in {0,1,;,-} plus PAD; optional EOS)

Expected files under --data_dir:
  train_src.txt / train_tgt.txt
  val_src_bin0.txt / val_tgt_bin0.txt
  val_src_bin1.txt / val_tgt_bin1.txt
  val_src_bin2.txt / val_tgt_bin2.txt

Run (example):
  python3 train_transformer_honest.py --data_dir data/n100 --cuda --epochs 50 --batch_size 256 \
    --emb_dim 128 --nhead 4 --layers 1 --ff_dim 128 --dropout 0.1 --lr 3e-4 --weight_decay 0.01 \
    --max_steps 30000 --log_every_steps 500 --amp --max_len 16384

Notes:
  - For "honesty", prefer last-token or --use_eos.
  - --use_cls is supported, but CLS can reintroduce some global shortcut behavior.
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
# Tokenizer (char-level)
# --------------------------

class CharTokenizer:
    """
    Base vocab: PAD, "0", "1", ";", "-".
    Optional EOS token appended at end of each sequence.
    """
    def __init__(self, use_eos: bool = False):
        self.pad = "<PAD>"
        self.eos = "<EOS>"
        self.base_vocab = [self.pad, "0", "1", ";", "-"]
        self.use_eos = use_eos

        self.vocab = list(self.base_vocab)
        if self.use_eos:
            self.vocab.append(self.eos)

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.pad_id = self.stoi[self.pad]
        self.eos_id = self.stoi[self.eos] if self.use_eos else None

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
        if self.use_eos:
            ids.append(self.eos_id)  # type: ignore[arg-type]
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
# Transformer model (HONEST)
# --------------------------

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        use_cls: bool,
        causal: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.use_cls = use_cls
        self.causal = causal

        # Optional CLS token: prepend one token id not in base vocab
        self.cls_id = vocab_size
        self.vocab_size = vocab_size + (1 if use_cls else 0)

        self.emb = nn.Embedding(self.vocab_size, emb_dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len + (1 if use_cls else 0), emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # float mask with -inf above diagonal; shape [T,T]
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x_tok: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x_tok: [B,T] with PADs
        lengths: [B] original lengths (including EOS if tokenizer appends it)
        Returns: [B,2] logits
        """
        B, T = x_tok.shape
        device = x_tok.device

        if self.use_cls:
            cls = torch.full((B, 1), self.cls_id, dtype=x_tok.dtype, device=device)
            x_tok = torch.cat([cls, x_tok], dim=1)  # [B,T+1]
            lengths = lengths + 1
            T = T + 1

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.emb(x_tok) + self.pos(pos_ids)  # [B,T,D]

        key_padding_mask = (x_tok == self.pad_id)  # [B,T] True where PAD

        attn_mask = self._causal_mask(T, device) if self.causal else None

        # PyTorch TransformerEncoder: "mask" is the (T,T) attention mask
        h = self.enc(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # [B,T,D]

        # HONEST pooling: last real token (or CLS if use_cls)
        if self.use_cls:
            pooled = h[:, 0, :]
        else:
            idx = (lengths - 1).clamp_min(0)  # [B]
            pooled = h[torch.arange(B, device=device), idx, :]  # [B,D]

        pooled = self.norm(pooled)
        return self.head(pooled)


# --------------------------
# Train/Eval
# --------------------------

def pick_amp_dtype(device: torch.device) -> torch.dtype:
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

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(amp and device.type == "cuda"),
            dtype=amp_dtype,
        ):
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


@dataclass
class ConsecPerfectStopper:
    """Stop after N consecutive epochs with val0 == 100%."""
    need: int = 3
    streak: int = 0

    def step(self, acc: float) -> bool:
        if acc >= 1.0 - 1e-12:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data/n50")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=512)

    ap.add_argument("--max_len", type=int, default=4096, help="max supported sequence length (positional table)")
    ap.add_argument("--use_cls", action="store_true", help="prepend CLS and classify at CLS (less 'honest')")
    ap.add_argument("--use_eos", action="store_true", help="append EOS token and classify at last token (=EOS)")
    ap.add_argument("--non_causal", action="store_true", help="disable causal mask (back to bidirectional)")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--stop_consec_100", type=int, default=3,
                    help="Stop after N consecutive epochs with val0=100%. 0 disables.")
    ap.add_argument("--save_path", type=str, default="ckpt_transformer_two_bucket_honest.pt")

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--max_steps", type=int, default=30000, help="Hard cap on optimizer updates (global). 0=unlimited")
    ap.add_argument("--log_every_steps", type=int, default=500, help="Print step log every N updates (0=off)")

    args = ap.parse_args()

    # seeds
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    tok = CharTokenizer(use_eos=args.use_eos)
    print("[Vocab]", tok.vocab, "size=", tok.vocab_size, "| use_eos=", args.use_eos)

    def p(name: str) -> str:
        return os.path.join(args.data_dir, name)

    train_ds = TxtPairDataset(p("train_src.txt"), p("train_tgt.txt"), tok)
    val0_ds  = TxtPairDataset(p("val_src_bin0.txt"), p("val_tgt_bin0.txt"), tok)
    val1_ds  = TxtPairDataset(p("val_src_bin1.txt"), p("val_tgt_bin1.txt"), tok)
    val2_ds  = TxtPairDataset(p("val_src_bin2.txt"), p("val_tgt_bin2.txt"), tok)

    # --- ROBUST max_len check over ALL splits ---
    def ds_max_len(ds: TxtPairDataset) -> int:
        return max(len(x) for x, _ in ds.samples)

    max_train = ds_max_len(train_ds)
    max_val0  = ds_max_len(val0_ds)
    max_val1  = ds_max_len(val1_ds)
    max_val2  = ds_max_len(val2_ds)
    max_seen = max(max_train, max_val0, max_val1, max_val2)

    print(f"[Len] max_train={max_train} max_val0={max_val0} max_val1={max_val1} max_val2={max_val2} => max={max_seen}")

    need_len = max_seen + (1 if args.use_cls else 0)
    if need_len > args.max_len:
        sugg = max(need_len, args.max_len * 2)
        raise ValueError(
            f"max_len={args.max_len} is too small; need at least {need_len}. "
            f"Increase --max_len (suggest >= {sugg})."
        )

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

    model = TransformerClassifier(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        emb_dim=args.emb_dim,
        nhead=args.nhead,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        use_cls=args.use_cls,
        causal=(not args.non_causal),
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    pool = "CLS" if args.use_cls else ("EOS(last)" if args.use_eos else "last")
    print(f"[Params] {n_params/1e6:.2f}M | causal={not args.non_causal} | pool={pool}")
    if args.use_cls and args.use_eos:
        print("[Warn] Both --use_cls and --use_eos enabled. Classification uses CLS (first token).")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    stopper = EarlyStopper(args.patience)
    perfect = ConsecPerfectStopper(args.stop_consec_100) if args.stop_consec_100 and args.stop_consec_100 > 0 else None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        tot = 0
        cor = 0
        loss_sum = 0.0

        amp_dtype = pick_amp_dtype(device)

        for x, lens, y in train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
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

            global_step += 1
            if args.log_every_steps > 0 and (global_step == 1 or global_step % args.log_every_steps == 0):
                cap = args.max_steps if args.max_steps > 0 else -1
                print(f"[Step {global_step:06d}/{cap}] loss={loss.item():.4f}")

            pred = logits.argmax(dim=-1)
            cor += int((pred == y).sum().item())
            tot += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_loss = loss_sum / max(1, tot)
        train_acc = cor / max(1, tot)

        v0 = evaluate(model, val0_loader, device, amp=args.amp)
        v1 = evaluate(model, val1_loader, device, amp=args.amp)
        v2 = evaluate(model, val2_loader, device, amp=args.amp)

        dt = time.time() - t0
        cap = args.max_steps if args.max_steps > 0 else -1
        print(
            f"Epoch {epoch:03d} | steps={global_step}/{cap} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val0 acc={v0['acc']*100:.2f}% loss={v0['loss']:.4f} | "
            f"val1 acc={v1['acc']*100:.2f}% loss={v1['loss']:.4f} | "
            f"val2 acc={v2['acc']*100:.2f}% loss={v2['loss']:.4f} | "
            f"time={dt:.1f}s"
        )

        # checkpoint on best val0
        if v0["acc"] >= stopper.best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args), "global_step": global_step}, args.save_path)

        # stop after N consecutive perfect val0 epochs
        if perfect is not None and perfect.step(v0["acc"]):
            print(f"[EarlyStop] val0 hit 100% for {perfect.need} consecutive epochs.")
            break

        # patience-based early stop
        if stopper.step(v0["acc"]):
            print(f"[EarlyStop] best val0={stopper.best*100:.2f}%")
            break

        if args.max_steps > 0 and global_step >= args.max_steps:
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
        with open("final_eval_transformer_honest.log", "a", encoding="utf-8") as logf:
            logf.write(final_str + "\n")
    else:
        print("[!] No best_state saved (unexpected).")


if __name__ == "__main__":
    main()
