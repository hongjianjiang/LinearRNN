#!/usr/bin/env python3
import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import trange

from components.rnns import RNNModel
from components.transformers import TransformerModel, TransformerXLModel, SimpleTransformerModel
from components.mogrifierLSTM import MogrifierLSTMModel
from components.hybrid_transformer_deltanet import HybridModel
from data_sortedgraph import (
    load_sortedgraph_data,
    build_char_vocab_from_files,
    Sampler,
)


# -----------------------------
# Helpers: decide loss/threshold
# -----------------------------
def model_outputs_probs(model_type: str) -> bool:
    # Treat EVERYTHING as logits.
    return False


def get_loss_and_pred_fn(outputs_are_probs: bool):
    if outputs_are_probs:
        loss_fn = nn.BCELoss()
        pred_to_label = lambda p: (p >= 0.5).float()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        pred_to_label = lambda p: (p >= 0.0).float()
    return loss_fn, pred_to_label


# -----------------------------
# Wrapper to unify model API
# -----------------------------
class _ModelWrapper(nn.Module):
    def __init__(self, inner: nn.Module, model_type: str):
        super().__init__()
        self.inner = inner
        self.model_type = model_type

    def init_hidden(self, bsz):
        return None

    def forward(self, src_TB, hidden, lengths):
        if self.model_type == "SAN":
            out = self.inner(src_TB, has_mask=False)
        else:
            out = self.inner(src_TB)

        if isinstance(out, tuple):
            out = out[0]
        return out, None


# -----------------------------
# DDP helpers
# -----------------------------
def ddp_is_on() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_setup(backend: str = "nccl"):
    if not ddp_is_on():
        return
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not ddp_is_on()) or ddp_rank() == 0


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def ddp_reduce_sum(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m


def ddp_broadcast_int(x: int, device: torch.device, src: int = 0) -> int:
    if not dist.is_initialized():
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.broadcast(t, src=src)
    return int(t.item())


# -----------------------------
# Deterministic equal-batch sharding for custom Sampler
# -----------------------------
def ddp_batch_plan(sampler: Sampler):
    """
    Returns (rank, world, bs, local_steps_per_epoch, global_batches_used)

    We compute:
      total_batches = floor(len(sampler) / bs)
      total_batches_used = (total_batches // world) * world   (drop remainder so ranks match)
      local_batches = total_batches_used // world
    """
    use_ddp = dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world = dist.get_world_size() if use_ddp else 1
    bs = sampler.batch_size

    total_batches = len(sampler) // bs
    total_batches_used = (total_batches // world) * world
    local_batches = total_batches_used // world

    return rank, world, bs, local_batches, total_batches_used


# -----------------------------
# Eval / Train
# -----------------------------
@torch.no_grad()
def evaluate(model, sampler, device, loss_fn, pred_to_label):
    model.eval()

    total_loss = 0.0
    total_seq = 0
    correct_seq = 0

    # For eval, uneven is OK, but we still keep it aligned (nice + fast)
    rank, world, bs, local_batches, _ = ddp_batch_plan(sampler)

    for k in range(local_batches):
        batch_id_global = rank + k * world
        i = batch_id_global * bs

        source, labels, lengths = sampler.get_batch(i)
        source = source.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        _, B = source.size()
        hidden = unwrap_model(model).init_hidden(B)
        output, _ = model(source, hidden, lengths)

        last_outputs = []
        for b in range(B):
            L = int(lengths[b].item())
            last_outputs.append(output[L - 1, b, 0])
        last_outputs = torch.stack(last_outputs).unsqueeze(1)

        loss = loss_fn(last_outputs, labels)
        total_loss += loss.item() * B

        preds = pred_to_label(last_outputs)
        correct_seq += (preds == labels).float().sum().item()
        total_seq += B

    total_loss = ddp_reduce_sum(total_loss, device)
    total_seq = ddp_reduce_sum(float(total_seq), device)
    correct_seq = ddp_reduce_sum(float(correct_seq), device)

    avg_loss = total_loss / max(total_seq, 1.0)
    acc = correct_seq / max(total_seq, 1.0)
    return avg_loss, acc


def train_epoch(model, sampler, optimizer, device, loss_fn, pred_to_label, local_step_budget: int = 0):
    """
    local_step_budget: maximum number of optimizer steps each rank performs this epoch (0 = no cap)
    IMPORTANT: must be identical across ranks in DDP.
    """
    model.train()

    total_loss = 0.0
    total_seq = 0
    correct_seq = 0
    steps_done = 0

    rank, world, bs, local_batches, _ = ddp_batch_plan(sampler)

    # If budget is set, cap the local batches
    if local_step_budget and local_step_budget < local_batches:
        local_batches = local_step_budget

    for k in range(local_batches):
        batch_id_global = rank + k * world
        i = batch_id_global * bs

        source, labels, lengths = sampler.get_batch(i)
        source = source.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        _, B = source.size()
        hidden = unwrap_model(model).init_hidden(B)

        optimizer.zero_grad(set_to_none=True)
        output, _ = model(source, hidden, lengths)

        last_outputs = []
        for b in range(B):
            L = int(lengths[b].item())
            last_outputs.append(output[L - 1, b, 0])
        last_outputs = torch.stack(last_outputs).unsqueeze(1)

        loss = loss_fn(last_outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        total_seq += B
        preds = pred_to_label(last_outputs)
        correct_seq += (preds == labels).float().sum().item()

        steps_done += 1

    total_loss = ddp_reduce_sum(total_loss, device)
    total_seq = ddp_reduce_sum(float(total_seq), device)
    correct_seq = ddp_reduce_sum(float(correct_seq), device)

    avg_loss = total_loss / max(total_seq, 1.0)
    train_acc = correct_seq / max(total_seq, 1.0)
    return avg_loss, train_acc, steps_done


# -----------------------------
# builders
# -----------------------------
def build_rnn(ntoken, args, noutputs=1):
    return RNNModel(
        rnn_type=args.rnn_type,
        ntoken=ntoken,
        noutputs=noutputs,
        ninp=ntoken,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        tie_weights=args.tied,
        is_embedding=args.use_emb
    )


def build_san(ntoken, args, noutputs=1):
    return TransformerModel(
        ntoken, noutputs,
        args.d_model, args.heads, args.d_ffn, args.depth, args.dropout,
        pos_encode=args.pos_encode,
        bias=False,
        pos_encode_type=args.pos_encode_type,
        max_period=args.max_period
    )


def build_san_simple(ntoken, args, noutputs=1):
    return SimpleTransformerModel(
        ntoken, noutputs,
        args.d_model, args.heads, args.d_ffn, args.depth, args.dropout,
        pos_encode=args.pos_encode,
        bias=False,
        pos_encode_type=args.pos_encode_type,
        max_period=args.max_period
    )


def build_san_rel(ntoken, args, noutputs=1):
    return TransformerXLModel(
        ntoken, noutputs,
        args.d_model, args.heads, args.d_ffn, args.depth, args.dropout
    )


def build_mogrify(ntoken, args, noutputs=1):
    base = MogrifierLSTMModel(args.rnn_type, ntoken, args.emb_size, args.nhid, args.nlayers, args.dropout, args.tied)
    base.decoder = nn.Linear(base.nhid, noutputs)

    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input, hidden, lengths):
            decoded, hidden = self.inner(input, hidden, lengths)
            return decoded, hidden

        def init_hidden(self, bsz):
            return self.inner.init_hidden(bsz)

    return _Wrap(base)


def build_hybrid(ntoken, args, noutputs=1):
    return HybridModel(
        ntoken, noutputs,
        d_model=args.d_model,
        nhead=args.heads,
        d_ffn=args.d_ffn,
        nlayers=args.depth,
        dropout=args.dropout,
        use_pos=args.pos_encode
    )


builders = {
    'RNN': build_rnn,
    'SAN': build_san,
    'SAN-Simple': build_san_simple,
    'SAN-Rel': build_san_rel,
    'Mogrify': build_mogrify,
    'HybridModel': build_hybrid,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument("--n", type=int, default=50)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30000, help='GLOBAL optimizer step cap (0=no cap)')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--rnn_type', type=str, default='RNN_RELU',
                        choices=['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'])
    parser.add_argument('--model_type', type=str, default='RNN',
                        choices=['RNN', 'SAN', 'SAN-Simple', 'SAN-Rel', 'Mogrify', 'HybridModel'])

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--pos_encode', action='store_true')
    parser.add_argument('--pos_encode_type', type=str, default='sin')
    parser.add_argument('--max_period', type=int, default=10000)

    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--use_emb', action='store_true')
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    use_ddp = ddp_is_on()

    # init DDP + device
    if use_ddp:
        ddp_setup(backend="nccl")
        lrk = ddp_local_rank()
        torch.cuda.set_device(lrk)
        device = torch.device("cuda", lrk)
    else:
        device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if is_main_process():
        print(f"Using device: {device} (ddp={use_ddp}, world_size={ddp_world_size()})")

    dataset = f'n{args.n}'

    train_corpus, val_corpora, data_dir = load_sortedgraph_data(data_root=args.data_root, dataset=dataset)
    if is_main_process():
        print(f"Using data directory: {data_dir}")
        print(f"Loaded {len(train_corpus.source)} train examples")
        print(f"Loaded {len(val_corpora)} validation bins")

    all_text = list(train_corpus.source)
    for c in val_corpora:
        all_text.extend(c.source)
    voc = build_char_vocab_from_files(all_text)
    ntoken = voc.nwords
    if is_main_process():
        print(f"Vocab size (including '<PAD>' + 'T'): {ntoken}")
        print(f"PAD id={voc.get_id('<PAD>')}  EOS(T) id={voc.get_id('T')}")

    train_sampler = Sampler(train_corpus, voc, batch_size=args.batch_size)
    val_samplers = [Sampler(c, voc, batch_size=args.batch_size) for c in val_corpora]

    model = builders[args.model_type](ntoken, args)
    if args.model_type in ("SAN", "SAN-Simple", "SAN-Rel"):
        model = _ModelWrapper(model, args.model_type)

    model = model.to(device)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
            broadcast_buffers=False,   # IMPORTANT: fixes your earlier alias-buffer crash
        )

    outputs_are_probs = model_outputs_probs(args.model_type)
    loss_fn, pred_to_label = get_loss_and_pred_fn(outputs_are_probs)
    if is_main_process():
        print(f"[LossMode] model_type={args.model_type} outputs_are_probs={outputs_are_probs} "
              f"=> loss={loss_fn.__class__.__name__}, threshold={'0.5' if outputs_are_probs else '0.0'}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_dir = os.path.join(data_dir, 'checkpoints')
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    best_accs = [0.0 for _ in val_samplers]
    best_model_path = None

    # GLOBAL step count (rank0 authoritative; broadcast step budget each epoch)
    global_steps_done = 0

    # early-stop: consecutive full (100%) acc on val bin0
    val0_consec_full = 0
    early_stop_epoch = None

    epoch_iter = trange(1, args.epochs + 1, desc='Epochs', disable=not is_main_process())

    try:
        for epoch in epoch_iter:
            # compute local step budget for this epoch (must match on all ranks)
            if args.max_steps and args.max_steps > 0:
                if is_main_process():
                    remaining_global = max(args.max_steps - global_steps_done, 0)
                    # each DDP step consumes 1 global step (all ranks together)
                    local_budget = remaining_global
                else:
                    local_budget = 0
                local_budget = ddp_broadcast_int(local_budget, device, src=0)
                if local_budget <= 0:
                    break
            else:
                local_budget = 0  # no cap

            train_loss, train_acc, local_steps = train_epoch(
                model, train_sampler, optimizer, device,
                loss_fn=loss_fn, pred_to_label=pred_to_label,
                local_step_budget=local_budget
            )

            # Update global step count consistently
            if is_main_process():
                global_steps_done += local_steps

            if use_ddp:
                # sync global_steps_done to all ranks (not strictly needed, but keeps logs consistent)
                global_steps_done = ddp_broadcast_int(global_steps_done if is_main_process() else 0, device, src=0)

            if is_main_process():
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | steps={global_steps_done}")

            # validation
            val0_acc_this_epoch = None
            for b_idx, vs in enumerate(val_samplers):
                val_loss, val_acc = evaluate(model, vs, device, loss_fn=loss_fn, pred_to_label=pred_to_label)

                if val_acc > best_accs[b_idx]:
                    best_accs[b_idx] = val_acc

                if b_idx == 0:
                    val0_acc_this_epoch = val_acc

                if is_main_process():
                    if b_idx == 0 and val_acc > best_val_acc:
                        print(f"  Val bin {b_idx}: loss={val_loss:.4f}, acc={val_acc:.4f}")
                        best_val_acc = val_acc
                        save_path = os.path.join(checkpoint_dir, f'best_model_{args.model_type}_{args.rnn_type}.pt')
                        torch.save(unwrap_model(model).state_dict(), save_path)
                        best_model_path = save_path

            # early-stop decision on rank0, broadcast
            early_stop_flag = 0
            if is_main_process():
                if val0_acc_this_epoch is not None and val0_acc_this_epoch >= 0.999999:
                    val0_consec_full += 1
                    print(f"  val0 consecutive full-accuracy count: {val0_consec_full}")
                else:
                    val0_consec_full = 0

                if val0_consec_full >= 3:
                    early_stop_epoch = epoch
                    es_path = os.path.join(checkpoint_dir, f'best_model_{args.model_type}_{args.rnn_type}_earlystop_epoch{epoch}.pt')
                    try:
                        torch.save(unwrap_model(model).state_dict(), es_path)
                        best_model_path = es_path
                    except Exception:
                        pass
                    print(f"Early stopping: val0 reached 100% for {val0_consec_full} consecutive evaluations (epoch {epoch}).")
                    early_stop_flag = 1

            if use_ddp:
                early_stop_flag = ddp_broadcast_int(early_stop_flag if is_main_process() else 0, device, src=0)
                if early_stop_flag == 1:
                    break
            else:
                if early_stop_flag == 1:
                    break

        if is_main_process():
            print(f"Training done. Best bin0 val acc = {best_val_acc:.4f}")

            # write log
            try:
                now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                log_fname = os.path.join(data_dir, f'train_log_{now}.json')
                base_model = unwrap_model(model)
                model_summary = {
                    'class': base_model.__class__.__name__,
                    'num_parameters': sum(p.numel() for p in base_model.parameters())
                }
                log_data = {
                    'timestamp_utc': now,
                    'args': vars(args),
                    'model': model_summary,
                    'ddp': {
                        'enabled': bool(use_ddp),
                        'world_size': ddp_world_size(),
                    },
                    'loss_mode': {
                        'outputs_are_probs': bool(outputs_are_probs),
                        'loss': loss_fn.__class__.__name__,
                        'threshold': 0.5 if outputs_are_probs else 0.0,
                    },
                    'best_accs_per_bin': best_accs,
                    'best_bin0_val_acc': best_val_acc,
                    'best_model_path': best_model_path,
                    'early_stop': {
                        'triggered': bool(early_stop_epoch is not None),
                        'reason': 'val0_3x_100pct' if early_stop_epoch is not None else None,
                        'epoch': early_stop_epoch,
                        'val0_consec_full': val0_consec_full,
                    },
                    'total_steps': global_steps_done,
                }
                with open(log_fname, 'w', encoding='utf-8') as lf:
                    json.dump(log_data, lf, indent=2)
                print(f'Wrote training log to {log_fname}')
            except Exception as e:
                print(f'Warning: failed to write training log: {e}')

    finally:
        # clean shutdown
        try:
            ddp_barrier()
        except Exception:
            pass
        ddp_cleanup()


if __name__ == '__main__':
    main()
