#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o mamba.log

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

DATA_DIR="${DATA_DIR:-data/n100}"

python - <<'PY'
import sys
try:
    import mamba_ssm  # noqa: F401
except Exception as e:
    sys.exit(
        "mamba_ssm not importable in this environment (" + repr(e) + "). "
        "The shared venv should provide it (mamba-ssm 2.3.0 + causal-conv1d 1.6.0 prebuilt wheels); "
        "rebuild with LinearRNN/.setup_venv.sh."
    )
PY

python3 -u train_mamba.py --data_dir "$DATA_DIR" --cuda \
    --epochs 999 --max_steps 30000 --batch_size 256 \
    --d_model 256 --depth 2 --d_state 128 --d_conv 4 --expand 2 \
    --dropout 0.1 --weight_decay 1e-4 --lr 3e-4 --grad_clip 1.0 \
    --log_every_steps 500 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers \
    --amp
