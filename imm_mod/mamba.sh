#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J mamba
#SBATCH -o mamba.log

set -euo pipefail

# -----------------------
# Tunables (env overrides)
# -----------------------
# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, fla 0.4.1, mamba-ssm 2.3.0 + causal-conv1d 1.6.0 prebuilt wheels)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

WORKDIR="${WORKDIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/mm_T100_bins}"
ALPHABET="${ALPHABET:-pm1}"

PYFILE="${PYFILE:-$WORKDIR/train_mamba.py}"

# CUDA robustness
export CUDA_MODULE_LOADING=LAZY

# -----------------------
# setup
# -----------------------
echo "[1/5] Activate venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Verify mamba imports"
python - <<'PY'
import torch, os
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "torch.version.cuda:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import causal_conv1d
print("causal_conv1d ok:", getattr(causal_conv1d, "__version__", "unknown"))
import mamba_ssm
print("mamba_ssm ok:", getattr(mamba_ssm, "__version__", "unknown"))
from mamba_ssm.modules.mamba_simple import Mamba
print("Mamba class:", Mamba)
PY

# -----------------------
# NEW: CUDA sanity tests
# -----------------------
echo "[3/5] CUDA sanity tests (nvidia-smi + cuBLAS matmul)"
echo "==== nvidia-smi ===="
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("[FAIL] No CUDA device visible. This job is not on a GPU or GPU allocation failed.")

torch.cuda.init()
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print("device0:", name)
print(f"mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB")

# cuBLAS handle / GEMM test
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b
torch.cuda.synchronize()
print("[OK] cuBLAS matmul works:", c.shape, c.dtype)
PY

echo "[4/5] Run training"
CMD=(
  python3 train_mamba.py --data_dir "$DATA_DIR" --auto_infer --cuda --amp --amp_dtype bf16 --d_model 256 --layers 2 --dropout 0.1 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2 --batch_size 128 --num_workers 0 --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 --max_steps 60000 --save_path ckpt_mamba_query.pt
)

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "[5/5] Done. venv=$VENV_DIR"
