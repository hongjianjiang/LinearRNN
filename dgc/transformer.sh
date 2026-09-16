#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH -o 22out.log 
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

python3 train_transformer.py --data_dir data/n100 --cuda \
  --epochs 50 --batch_size 64 \
  --emb_dim 64 --nhead 2 --layers 1 --ff_dim 32 \
  --dropout 0.1 --lr 3e-4 --weight_decay 0.01 \
  --amp --max_steps 30000 --log_every_steps 500 \
  --use_eos --max_len 5000

