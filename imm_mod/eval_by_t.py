#!/usr/bin/env python3
"""
Per-step-position accuracy breakdown for imm_mod (dense stepwise supervision).

Unlike imm_unbound's eval_by_T.py -- which buckets by the *whole example's*
length T, because that task only supervises the single final token -- this
task supervises v_t at *every* MAT token t=1..T. The question we actually
care about is: does accuracy at step index t degrade as t grows, regardless
of which test split (bin0/1/2) the example came from? A bin1 example with
T=150 still has a real, honestly-supervised target at t=1..150, so we pool
test_bin0+test_bin1+test_bin2 together and bucket every (example, t) pair by
its absolute step index t. This is the same "bin-level accuracy blends easy
short-prefix positions with hard long-prefix positions" trap flagged for
imm_unbound -- this script exists to check whether it's happening here too.

Usage:
  python3 eval_by_t.py rnn      ckpt_mm_tf_rnn.pt          data/mm_T100_bins
  python3 eval_by_t.py deltanet ckpt_deltanet_stepwise.pt  data/mm_T100_bins
  python3 eval_by_t.py rwkv     ckpt_rwkv7_perm_s3.pt      data/perm_s3_T100_bins
"""
import os
# must be set before importing train_rwkv7, which applies its
# torch.compile-disable monkeypatch at import time based on this env var.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import importlib
from collections import defaultdict

import torch

kind, ckpt_path, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
assert kind in ("rnn", "deltanet", "rwkv"), "first arg must be 'rnn', 'deltanet', or 'rwkv'"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location="cpu")
a = ckpt["args"]

correct_by_t = defaultdict(int)
total_by_t = defaultdict(int)

if kind == "rnn":
    mod = importlib.import_module("train_rnn_relu")
    model = mod.ModelRNNQuery(
        rnn_kind=a["rnn"], m_max=a["m_max"], max_len=a["max_len"], d_model=a["d_model"],
        layers=a["layers"], dropout=a["dropout"], act_clip=a["act_clip"],
        teacher_force_state=a.get("teacher_force_state", False),
    )
    # strict=False: older checkpoints (trained before the aux_head existed)
    # won't have aux_head.* keys -- fine, this script never reads aux output.
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    for split in ["test_bin0", "test_bin1", "test_bin2"]:
        ds = mod.PreloadedQueryStepDataset(
            f"{data_dir}/{split}_src.txt", f"{data_dir}/{split}_tgt.txt",
            alphabet=a.get("alphabet", "pm1"), m_max=a["m_max"], quiet=True,
        )
        items = [ds[i] for i in range(len(ds))]
        with torch.no_grad():
            for i in range(0, len(items), 256):
                chunk = items[i:i + 256]
                batch = mod.collate_batch(chunk, m_max=a["m_max"])
                logits, _ = model(batch.tok_type.to(device), batch.mats_mod.long().to(device),
                                   batch.r_prev.long().to(device), batch.qj.to(device),
                                   batch.lengths)
                pred = logits.argmax(dim=-1).cpu()
                valid = (batch.y_tok != -100)
                b_ix, l_ix = valid.nonzero(as_tuple=True)
                t = l_ix  # index 0=BOS, index t holds MAT_t's prediction/target
                correct = (pred[b_ix, l_ix] == batch.y_tok[b_ix, l_ix])
                for tt, c in zip(t.tolist(), correct.tolist()):
                    total_by_t[tt] += 1
                    correct_by_t[tt] += int(c)

elif kind == "rwkv":
    mod = importlib.import_module("train_rwkv7")
    model = mod.ModelRWKVStepwise(
        m_max=a["m_max"], max_len=a["max_len"], d_model=a["d_model"],
        head_dim=a["rwkv7_head_dim"], layers=a["rwkv7_depth"], dropout=a["dropout"],
        rwkv_mode=a["rwkv7_mode"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    for split in ["test_bin0", "test_bin1", "test_bin2"]:
        ds = mod.PreloadedModQueryStepwiseDataset(
            f"{data_dir}/{split}_src.txt", f"{data_dir}/{split}_tgt.txt",
            alphabet=a.get("alphabet", "pm1"), m_max=a["m_max"], quiet=True,
        )
        items = [ds[i] for i in range(len(ds))]
        with torch.no_grad():
            for i in range(0, len(items), 256):
                chunk = items[i:i + 256]
                batch = mod.collate_batch(chunk, m_max=a["m_max"])
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    logits = model(batch.tok_type.to(device), batch.tok_val.to(device),
                                   batch.attn01.to(device))
                masked = mod.mask_logits_by_m(logits.float(), batch.m_i.to(device))
                pred = masked.argmax(dim=-1).cpu()
                valid = (batch.y_tok != -100)
                b_ix, l_ix = valid.nonzero(as_tuple=True)
                t = l_ix - 1  # index 0=BOS, 1=META, index (t+1) holds MAT_t
                correct = (pred[b_ix, l_ix] == batch.y_tok[b_ix, l_ix])
                for tt, c in zip(t.tolist(), correct.tolist()):
                    total_by_t[tt] += 1
                    correct_by_t[tt] += int(c)

else:
    mod = importlib.import_module("train_deltanet")
    model = mod.ModelDeltaNet(
        m_max=a["m_max"], max_len=a["max_len"], d_model=a["d_model"], heads=a["heads"],
        layers=a["layers"], dropout=a["dropout"], deltanet_mode=a["deltanet_mode"],
        allow_neg_eigval=a["allow_neg_eigval"], target_mode=a["target_mode"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    for split in ["test_bin0", "test_bin1", "test_bin2"]:
        ds = mod.PrecomputedIMMModStepwiseDataset(
            f"{data_dir}/{split}_src.txt", f"{data_dir}/{split}_tgt.txt",
            alphabet=a.get("alphabet", "pm1"), target_mode=a["target_mode"], m_max=a["m_max"],
            cache_path=None, teacher_force_state=a.get("teacher_force_state", False),
        )
        items = [ds[i] for i in range(len(ds))]
        with torch.no_grad():
            for i in range(0, len(items), 256):
                chunk = items[i:i + 256]
                batch = mod.collate_batch(chunk)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    cls_logits, _ = model(batch.tok_type.to(device), batch.tok_val.to(device),
                                           batch.attn01.to(device))
                masked = mod.mask_logits_by_m(cls_logits.float(), batch.m_i.to(device))
                pred = masked.argmax(dim=-1).cpu()
                b_ix, l_ix = batch.y_mask.nonzero(as_tuple=True)
                t = l_ix - 1  # index 0=BOS, 1=META, index (t+1) holds MAT_t
                correct = (pred[b_ix, l_ix] == batch.y_tgt[b_ix, l_ix])
                for tt, c in zip(t.tolist(), correct.tolist()):
                    total_by_t[tt] += 1
                    correct_by_t[tt] += int(c)

Ts = sorted(total_by_t)
print(f"{kind}: per-t step accuracy (pooled over test_bin0+1+2, n at t=1 -> {total_by_t[Ts[0]]})")
for t in Ts:
    acc = correct_by_t[t] / total_by_t[t]
    marker = " <-- train range ends" if t == 100 else ""
    if t <= 15 or t % 20 == 0 or t in (99, 100, 101, 199, 200, 201, 299, 300):
        print(f"  t={t:3d}  n={total_by_t[t]:6d}  acc={acc*100:5.1f}%{marker}")
