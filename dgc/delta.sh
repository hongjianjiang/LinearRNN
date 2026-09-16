#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o delta.log

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

DATA_DIR="${DATA_DIR:-data/n100}"
python3 train_deltanet.py --data_dir "$DATA_DIR" --cuda --epochs 100 --batch_size 128 \
    --emb_dim 128 --hidden_size 256 --layers 1 --num_heads 4 --mode chunk --dropout 0.1 \
    --max_steps 30000 --log_every_steps 500

