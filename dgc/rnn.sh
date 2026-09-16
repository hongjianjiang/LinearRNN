#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J rnn
#SBATCH -o rnn.log
 
set -euo pipefail

# ============================================================
# User knobs (env overrides)
# ============================================================
# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, fla 0.4.1, mamba-ssm 2.3.0)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

# Project paths (defaults assume you submit from repo root)
WORKDIR="${WORKDIR:-$PWD}"
DATA_DIR="${DATA_DIR:-data/n100}"
TRAIN_PY="${TRAIN_PY:-$WORKDIR/train_rnn.py}"        # <-- the TF-RNN script you saved
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# 0 (default) => input-only tokenization (matches Mamba/Transformer baselines)
# 1 => leaky STATE-token teacher forcing ablation (see train_rnn.py --teacher_force_state)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# Training hyperparams (sane defaults)
RNN_TYPE="${RNN_TYPE:-gru}"          # gru / rnn_tanh / rnn_relu
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-2}"
# 0.1/1e-3 let the GRU hit 99.8% train acc vs ~55% val/test (pure
# memorization). Stronger dropout + weight decay to close that gap.
DROPOUT="${DROPOUT:-0.3}"

MLP_HMULT="${MLP_HMULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"           # gelu / relu

BATCH_SIZE="${BATCH_SIZE:-256}"
# epochs=200 was silently the binding stop condition (~15800/30000 steps,
# bad=85/100 patience) -- raise it so max_steps/patience actually govern.
EPOCHS="${EPOCHS:-500}"
MAX_STEPS="${MAX_STEPS:-30000}"

LR="${LR:-3e-4}"
WD="${WD:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"

# In-distribution (test_bin0) accuracy was only ~54-58% even after the
# dropout/WD/epochs fixes -- the bottleneck is the sparse single-bit,
# up-to-T=100-step credit assignment, not regularization. Dense per-step
# auxiliary supervision targets this directly.
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.3}"
# ramp train sequence length T=5->100 over the first 150 epochs instead of
# exposing the full 100-step credit-assignment horizon from epoch 1.
CURRICULUM_MIN_T="${CURRICULUM_MIN_T:-5}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-150}"

# TF tokenization knobs
# max_len=0 => auto-infer from train_src (recommended)
MAX_LEN="${MAX_LEN:-0}"
# state_cap=0 => no clipping (recommended if your gen.py used --value_cap)
STATE_CAP="${STATE_CAP:-0}"

# DataLoader
NUM_WORKERS="${NUM_WORKERS:-2}"

# Output checkpoint
SAVE_PATH="${SAVE_PATH:-$WORKDIR/ckpt_mm_tf_rnn.pt}"

# ============================================================
# 1) Activate shared venv
# ============================================================
echo "[1/3] Activate venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ============================================================
# 2) Environment summary + sanity checks
# ============================================================
echo
echo "===================="
echo "[2/3] Environment"
echo "===================="
python - <<'PY'
import sys
import torch
import numpy as np
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy :", np.__version__)
PY



# ============================================================
# 3) Run training
# ============================================================
echo
echo "===================="
echo "[3/3] Run training"
echo "===================="

CMD=(
python3 -u train_rnn.py --data_dir "$DATA_DIR" --cuda \
    --epochs 999 --max_steps 30000 --batch_size 256 --emb_dim 128 --hidden 256 --layers 1 \
    --dropout 0.1 --weight_decay 1e-4 --lr 3e-4 --grad_clip 1.0 \
    --hh_shrink 0.3 --act_clip 6.0 \
    --log_every_steps 500 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers \
    --amp
)
if [[ "$MAX_LEN" != "0" ]]; then
  CMD+=(--max_len "$MAX_LEN")
fi
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done."
echo "Venv: $VENV_DIR"
echo "Checkpoint: $SAVE_PATH"

