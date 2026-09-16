#!/bin/bash
#SBATCH -p gpu-a100
# NOT gpu-h100: triton 3.1 (pinned by torch 2.5.1) aborts compiling fla's RWKV7 kernels on
# non-Ampere GPUs ("mma -> mma layout conversion is only supported on Ampere").
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J step1_rwkv7
#SBATCH -o step1_rwkv7_%j.out
#SBATCH -e step1_rwkv7_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/n100}"

# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, triton 3.1.0, fla-core/flash-linear-attention 0.4.1)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

echo "[1/3] Activate venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# fla JIT-compiles triton kernels; keep that cache node-local instead of ~/.triton
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
mkdir -p "$TRITON_CACHE_DIR"

# fla 0.4.1's fused_addcmul triton kernel breaks under triton 3.1 -> torch.addcmul fallback.
# The v_first=None guard that this script used to patch into site-packages is now applied
# at runtime by train_rwkv7.py (torch.lerp guard), so the shared venv stays unmodified.
export DISABLE_RWKV7_FUSED_ADDCMUL=1

echo
echo "===================="
echo "[2/3] Run RWKV-7 training"
echo "===================="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

CMD=(python3 -u train_rwkv7.py --data_dir "${DATA_DIR}" --cuda \
  --epochs 999 --max_steps 30000 --batch_size 64 \
  --emb_dim 128 --hidden 256 \
  --rwkv7_depth 2 --rwkv7_head_dim 64 --rwkv7_mode chunk \
  --dropout 0.1 --weight_decay 1e-4 --lr 3e-5 --grad_clip 0.5 \
  --amp --log_every_steps 500 --print_impl)

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo
echo "[3/3] Done."
echo "Venv: $VENV_DIR"
