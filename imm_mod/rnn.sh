#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J immmod_rnn
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
DATA_DIR="${DATA_DIR:-$PWD/data/mm_T100_bins}"
TRAIN_PY="${TRAIN_PY:-$WORKDIR/train_rnn_relu.py}"
M_MAX="${M_MAX:-29}"
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# 0 (default) => input-only tokenization (no leaked r_prev row)
# 1 => leaky --teacher_force_state ablation (see train_rnn_relu.py)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# Training hyperparams (sane defaults)
RNN_TYPE="${RNN_TYPE:-gru}"          # gru / rnn_tanh / rnn_relu
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"

# 0.0 (default) => no auxiliary loss. >0 => also predict phi(P_t) (all 9
# residues of the full state, output-only, no leak) via SmoothL1, weighted
# by AUX_W, alongside the primary cross-entropy loss on v_t.
AUX_W="${AUX_W:-0.0}"

MLP_HMULT="${MLP_HMULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"           # gelu / relu

BATCH_SIZE="${BATCH_SIZE:-256}"
# with n_train~70000 and batch_size=256, ~275 steps/epoch; 200 epochs was the
# silent binding stop condition in imm_unbound (see project memory) -- raise
# it so max_steps/patience actually govern.
EPOCHS="${EPOCHS:-400}"
MAX_STEPS="${MAX_STEPS:-60000}"

LR="${LR:-3e-4}"
WD="${WD:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PATIENCE="${PATIENCE:-30}"
EARLY_STOP="${EARLY_STOP:-finalAcc}"

# TF tokenization knobs
# max_len=0 => auto-infer from train_src (recommended)
MAX_LEN="${MAX_LEN:-0}"
# state_cap=0 => no clipping (recommended if your gen.py used --value_cap)
STATE_CAP="${STATE_CAP:-0}"

# AMP
AMP_BF16="${AMP_BF16:-1}"            # 1 => pass --amp_bf16

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

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "ERROR: training script not found: $TRAIN_PY"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi

# quick presence check for expected files
for sp in train val_bin0 test_bin0 test_bin1 test_bin2; do
  [[ -f "$DATA_DIR/${sp}_src.txt" ]] || { echo "ERROR: missing $DATA_DIR/${sp}_src.txt"; exit 1; }
  [[ -f "$DATA_DIR/${sp}_tgt.txt" ]] || { echo "ERROR: missing $DATA_DIR/${sp}_tgt.txt"; exit 1; }
done

# ============================================================
# 3) Run training
# ============================================================
echo
echo "===================="
echo "[3/3] Run training"
echo "===================="

CMD=(
python3 -u "$TRAIN_PY" \
  --data_dir "$DATA_DIR" \
  --alphabet "$ALPHABET" --m_max "$M_MAX" \
  --cuda --amp --amp_dtype bf16 \
  --rnn "$RNN_TYPE" --d_model "$D_MODEL" --layers "$LAYERS" --dropout "$DROPOUT" --act_clip 0.0 \
  --aux_w "$AUX_W" \
  --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WD" --grad_clip "$GRAD_CLIP" \
  --max_steps "$MAX_STEPS" --epochs "$EPOCHS" --eval_every 2 --patience "$PATIENCE" --early_stop "$EARLY_STOP" \
  --num_workers "$NUM_WORKERS" --prefetch_factor 4 --persistent_workers \
  --save_path "$SAVE_PATH"
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
