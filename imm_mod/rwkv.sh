#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J rwkv7_train
#SBATCH -o rwkv.log

set -euo pipefail

# MUST be defined before any use (because of -u)
BASE="/tmp/${USER}/rwkv7_train_${SLURM_JOB_ID:-local}"

# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, triton 3.1.0, fla-core/flash-linear-attention 0.4.1)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

# -----------------------
# stability / cluster knobs
# -----------------------
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export DISABLE_RWKV7_FUSED_ADDCMUL=1

# keep caches off $HOME (important on NFS)
export TRITON_CACHE_DIR="${BASE}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/torchinductor-cache"

# extra “quiet + stable” knobs (optional but recommended)
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATCH_TORCH_LERP_DTYPE=1

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# avoid CPU oversubscription during preload / tokenization
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[Host] $(hostname)"
echo "[BASE] $BASE"
echo "[VENV] $VENV_DIR"
echo "[TRITON_CACHE_DIR] $TRITON_CACHE_DIR"
echo "[TORCHINDUCTOR_CACHE_DIR] $TORCHINDUCTOR_CACHE_DIR"

# -----------------------
# venv
# -----------------------
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
python - <<'PY'
import torch
import triton
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("triton:", triton.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# -----------------------
# run training
# -----------------------
DATA_DIR="${DATA_DIR:-data/mm_T100_bins}"
M_MAX="${M_MAX:-29}"
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})
SAVE_PATH="${SAVE_PATH:-ckpt_rwkv7_mmquery_stepwise.pt}"

# 0 (default) => input-only tokenization (no leaked STATE token)
# 1 => leaky --teacher_force_state ablation (see train_rwkv7.py)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# 1 (default) => enable --amp. 0 => full fp32 (test whether bf16's ~8-bit
# mantissa is the source of the mid-sequence precision drift found on the
# permutation-matrix task).
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
D_MODEL="${D_MODEL:-256}"
HEAD_DIM="${HEAD_DIM:-64}"
DEPTH="${DEPTH:-2}"

CMD=(
python3 -u train_rwkv7.py \
  --data_dir "$DATA_DIR" \
  --alphabet "$ALPHABET" \
  --m_max "$M_MAX" \
  --cuda \
  --d_model "$D_MODEL" --rwkv7_head_dim "$HEAD_DIM" --rwkv7_depth "$DEPTH" --rwkv7_mode chunk \
  --dropout 0.1 \
  --batch_size 256 --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 \
  --max_steps 60000 --eval_every 2 --patience 20 --early_stop loss \
  --num_workers 2 \
  --save_path "$SAVE_PATH"
)
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi
if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp --amp_dtype "$AMP_DTYPE")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
