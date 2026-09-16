#!/bin/bash
#SBATCH -p gpu-a100
# NOT gpu-h100: triton 3.1 (pinned by torch 2.5.1) aborts compiling fla's RWKV7 kernels on
# non-Ampere GPUs ("mma -> mma layout conversion is only supported on Ampere").
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=rwkv7
#SBATCH -o rwkv.log
#SBATCH --time=24:00:00
set -euo pipefail

# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, fla 0.4.1, mamba-ssm 2.3.0)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# fla JIT-compiles triton kernels; keep that cache node-local instead of ~/.triton
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
mkdir -p "$TRITON_CACHE_DIR"

# fla 0.4.1's fused_addcmul triton kernel breaks under triton 3.1 -> torch.addcmul fallback
# (read by train_rwkv7.py, same as imm_mod/rwkv.sh)
export DISABLE_RWKV7_FUSED_ADDCMUL=1

python3 -u train_rwkv7.py \
  --data_dir data/n100 --cuda \
  --rwkv7_depth 2 --rwkv7_head_dim 64 --rwkv7_mode chunk \
  --epochs 999 --max_steps 30000 --batch_size 64 \
  --emb_dim 128 --hidden 256 --dropout 0.1 \
  --lr 3e-5 --weight_decay 1e-4 --grad_clip 0.5 \
  --amp --log_every_steps 500
