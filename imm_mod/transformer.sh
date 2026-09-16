#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J transformer
#SBATCH -o transformer.log

set -euo pipefail

# ============================================================
# User knobs (env overrides)
# ============================================================
# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, fla 0.4.1, mamba-ssm 2.3.0)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

# Project paths (defaults assume you submit from repo root)
WORKDIR="${WORKDIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/mm_T100_bins}"                 # <-- your NO-MOD binary dataset dir
TRAIN_PY="${TRAIN_PY:-$WORKDIR/train_transformer.py}"       # <-- UPDATED binary transformer script

# Training hyperparams
ALPHABET="${ALPHABET:-pm1}"      # pm1 / 01

D_MODEL="${D_MODEL:-256}"
HEADS="${HEADS:-8}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
FF_MULT="${FF_MULT:-4}"

USE_M_EMB="${USE_M_EMB:-0}"      # 1 => pass --use_m_embedding

BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-200}"
MAX_STEPS="${MAX_STEPS:-30000}"

LR="${LR:-3e-4}"
WD="${WD:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PATIENCE="${PATIENCE:-20}"
EARLY_STOP="${EARLY_STOP:-loss}" # loss / acc

# Auto-infer if 0 (recommended)
MAX_LEN="${MAX_LEN:-0}"

# AMP
AMP="${AMP:-1}"                 # 1 => pass --amp
AMP_DTYPE="${AMP_DTYPE:-bf16}"  # bf16 / fp16

# DataLoader
NUM_WORKERS="${NUM_WORKERS:-2}"

# Output checkpoint
SAVE_PATH="${SAVE_PATH:-$WORKDIR/ckpt_mm_transformer_norowtf_binary.pt}"

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
python3 -u train_transformer.py \
  --data_dir "$DATA_DIR" \
  --auto_infer \
  --cuda --amp --amp_dtype bf16 \
  --d_model 256 --heads 4 --layers 2 --dropout 0.1 \
  --batch_size 256 --num_workers 4 \
  --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 \
  --max_steps 60000 --eval_every 1000 \
  --save_path ckpt_transformer_stepwise_mrange.pt

)

if [[ "$USE_M_EMB" == "1" ]]; then
  CMD+=(--use_m_embedding)
fi

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp --amp_dtype "$AMP_DTYPE")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done."
echo "Venv: $VENV_DIR"
echo "Checkpoint: $SAVE_PATH"
