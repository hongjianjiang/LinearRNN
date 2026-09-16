#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J transformer
#SBATCH -o transformer.log

set -euo pipefail

# ============================================================
# User knobs (env overrides)
# ============================================================
CU_INDEX="${CU_INDEX:-cu126}"

# Put venv + pip cache in /tmp (avoid home quota)
TMP_BASE="${TMP_BASE:-/tmp/$USER/mm_tf_tr_bin}"
VENV_DIR="${VENV_DIR:-$TMP_BASE/venv}"

# Project paths (defaults assume you submit from repo root)
WORKDIR="${WORKDIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/mm_T100}"                 # <-- your NO-MOD binary dataset dir
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
# 1) Prepare /tmp dirs
# ============================================================
echo "[1/8] Prepare /tmp directories: $TMP_BASE"
mkdir -p "$TMP_BASE" "$TMP_BASE/pip-cache" "$TMP_BASE/pip-tmp"

# ============================================================
# 2) Create/reuse venv
# ============================================================
echo "[2/8] Create or reuse venv: $VENV_DIR"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  echo "  [reuse] venv exists"
else
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ============================================================
# 3) Force pip cache/temp to /tmp
# ============================================================
echo "[3/8] Configure pip cache/temp to /tmp"
export PIP_CACHE_DIR="$TMP_BASE/pip-cache"
export TMPDIR="$TMP_BASE/pip-tmp"

# ============================================================
# 4) Upgrade pip tooling
# ============================================================
echo "[4/8] Upgrade pip tooling"
python -m pip install --no-cache-dir -U pip setuptools wheel

# ============================================================
# 5) Install PyTorch (CUDA wheels) if missing
# ============================================================
echo "[5/8] Ensure torch installed (CUDA wheels: $CU_INDEX)"
if python - <<'PY'
import importlib
try:
    importlib.import_module("torch")
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  echo "  [skip] torch already installed"
else
  python -m pip install --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/${CU_INDEX}" \
    torch torchvision torchaudio
fi

# ============================================================
# 6) Install deps
# ============================================================
echo "[6/8] Install deps"
python -m pip install --no-cache-dir -U numpy tqdm

# ============================================================
# 7) Environment summary + sanity checks
# ============================================================
echo
echo "===================="
echo "[7/8] Environment"
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
# 8) Run training
# ============================================================
echo
echo "===================="
echo "[8/8] Run training"
echo "===================="

CMD=(
  python3 -u train_transformer.py \
  --data_dir "$DATA_DIR" \
  --alphabet pm1 --cuda --amp --amp_dtype bf16 \
  --d_model 256 --heads 8 --layers 2 --dropout 0.1 --ff_mult 4 \
  --batch_size 256 --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 \
  --max_steps 60000 --patience 200 \
  --save_path ckpt_tf_imm_T100.pt

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
