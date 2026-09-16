import sys, torch, importlib
from collections import defaultdict

module_name, ckpt_path, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
mod = importlib.import_module(module_name)

ckpt = torch.load(ckpt_path, map_location="cpu")
a = ckpt["args"]

if module_name == "train_rnn":
    model = mod.ModelRNNFinalBinary(max_len=a["max_len"], d_model=a["d_model"], hidden=a["hidden"],
                                     layers=a["layers"], dropout=a["dropout"], rnn_type=a["rnn_type"])
else:
    model = mod.ModelDeltaNetFinalBinary(max_len=a["max_len"], d_model=a["d_model"], layers=a["layers"],
                                          heads=a["heads"], dropout=a["dropout"], chunk_size=a.get("chunk_size", 64))
model.load_state_dict(ckpt["model"])
model.eval()

correct_by_T = defaultdict(int)
total_by_T = defaultdict(int)

for split in ["test_bin0", "test_bin1", "test_bin2"]:
    ds = mod.PreloadedNoModBinaryDataset(f"{data_dir}/{split}_src.txt", f"{data_dir}/{split}_tgt.txt", "pm1", quiet=True)
    items = [ds[i] for i in range(len(ds))]
    # batch by exact T for simplicity/speed
    by_T = defaultdict(list)
    for it in items:
        by_T[it["T"]].append(it)
    for T, group in by_T.items():
        with torch.no_grad():
            for i in range(0, len(group), 256):
                chunk = group[i:i+256]
                batch = mod.collate_batch(chunk, teacher_force_state=False)
                if module_name == "train_rnn":
                    logits = model(batch.tok_type, batch.tok_val_phi, batch.lengths)
                else:
                    logits = model(batch.tok_type, batch.tok_val_phi, batch.attn01)
                acc = mod.acc_final(logits, batch.y, batch.y_mask)
                correct_by_T[T] += acc * len(chunk)
                total_by_T[T] += len(chunk)

Ts = sorted(total_by_T)
print(f"{module_name}: per-T accuracy (n per T = {total_by_T[Ts[0]]})")
for T in Ts:
    acc = correct_by_T[T] / total_by_T[T]
    marker = " <-- train range ends" if T == 100 else ""
    if T <= 15 or T % 20 == 0 or T in (99,100,101,199,200,201,299,300):
        print(f"  T={T:3d}  acc={acc*100:5.1f}%{marker}")
