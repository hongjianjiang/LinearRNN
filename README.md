# LinearRNN

Length-generalization experiments that compare nonlinear RNNs, Transformers, and linear RNNs (DeltaNet, Mamba, RWKV-7) on synthetic state-tracking tasks. For each task, models train on short sequences and are tested on longer ones.

## Tasks

| Directory | Task | Label |
|---|---|---|
| `dgc/` | Two-bucket graph reachability. Input is `src;i1-j1;i2-j2;...;tgt` with binary node ids and sorted edges. | Final: is `tgt` reachable from `src`? (0/1) |
| `imm_mod/` | Iterated 3x3 matrix multiplication over Z_m (prime m). Matrices come from {-1,0,1} and must be invertible mod m. | Stepwise: `(M1 @ ... @ Mt)[qk] mod m` for every t |
| `imm_unbound/` | Iterated 3x3 matrix multiplication over the integers, with no modulus. Matrices come from {-1,0,1}. | Final: is `(P_T)[0,0] == 0`? (0/1) |

`gen_perm.py` in `imm_mod/` and `imm_unbound/` builds a variant of the same task that uses 3x3 permutation matrices (the group S_3). The state then stays finite and never diverges.

## Layout

Each task directory holds the same kinds of files:

- `gen.py` / `gen_perm.py` / `generate_data.py`: dataset generators
- `train_rnn*.py`, `train_transformer.py`, `train_deltanet.py`, `train_mamba.py`, `train_rwkv7.py`: one training script per architecture
- `rnn.sh`, `transformer.sh`, `delta.sh`, `mamba.sh`, `rwkv.sh`: SLURM job scripts that wrap the training scripts
- `eval_by_t.py` (`imm_mod`) / `eval_by_T.py` (`imm_unbound`): accuracy broken down by step or sequence length

`dgc/` also contains `main.py`, a general training entry point that uses the models in `components/` and the helpers in `utils/`.

## Requirements

- Python 3.12
- PyTorch 2.5.1 (CUDA)
- `numpy`, `tqdm`
- [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) 0.4.1 (DeltaNet, RWKV-7)
- `mamba-ssm` 2.3.0 (Mamba)

The job scripts expect an existing virtualenv at `$VENV_DIR`; set that variable to point at your own environment. The RWKV-7 kernels need an Ampere GPU such as an A100 when used with triton 3.1.

## Data

Datasets are **not** committed: `data/` is listed in `.gitignore`, along with caches and checkpoints. Generate the data locally inside each task directory before training:

```bash
# dgc: writes to data/n<n>/ (train, val_bin0/1/2 with N in [2,n], [n+1,2n], [2n+1,3n])
cd dgc && python3 generate_data.py --n 100

# imm_mod: train on T <= T_train; test_bin1/2 extend by T_gap each
cd imm_mod && python3 gen.py --out_dir data/mm_modm --T_train 50 --T_gap 100 --m 29

# imm_unbound
cd imm_unbound && python3 gen.py --out_dir data/mm_nomod_binary_T100 --T_train 100 --T_gap 100 --balanced
```

Each split is written as a `<split>_src.txt` / `<split>_tgt.txt` pair. The splits are `train`, `val_bin0`, `test_bin0`, `test_bin1`, and `test_bin2`; `dgc` instead names them `val_src_bin{0,1,2}`. Run a generator with `--help` to see all of its options.

## Training

Submit a job from inside the task directory, and use environment variables to override its defaults:

```bash
cd imm_unbound
DATA_DIR=data/mm_nomod_binary_T100 RNN_TYPE=gru sbatch rnn.sh
```

You can also call a training script directly. See `imm_unbound/READEME.md` for example commands, or run any `train_*.py --help`.

## License

MIT. See [LICENSE](LICENSE).
