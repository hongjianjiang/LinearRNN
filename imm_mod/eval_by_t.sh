#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J eval_by_t
#SBATCH -o eval_by_t.log
set -euo pipefail
cd ~/LinearRNN/imm_mod

# disabling makes fla's module-level @torch.compile decorators a no-op
# (same as rwkv.sh); only strictly needed on Python 3.13+, harmless on 3.12.
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"

# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, triton 3.1.0, fla-core/flash-linear-attention 0.4.1)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env ready ==="
python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"

echo
echo "########## RNN per-t eval ##########"
python3 eval_by_t.py rnn ckpt_mm_tf_rnn.pt data/mm_T100_bins

echo
echo "########## DeltaNet per-t eval ##########"
python3 eval_by_t.py deltanet ckpt_deltanet_stepwise.pt data/mm_T100_bins
