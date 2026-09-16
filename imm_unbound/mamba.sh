#!/bin/bash
#SBATCH -p gpu22
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
# Use cu118 wheels to avoid CUDA mismatch builds on many clusters.
CU_INDEX="${CU_INDEX:-cu118}"

TMP_BASE="${TMP_BASE:-/tmp/$USER/mamba}"
VENV_DIR="${VENV_DIR:-$TMP_BASE/venv}"
PIP_CACHE_DIR_LOCAL="${PIP_CACHE_DIR:-$TMP_BASE/pip-cache}"
TMPDIR_LOCAL="${TMPDIR:-$TMP_BASE/pip-tmp}"

WORKDIR="${WORKDIR:-$PWD}"

DATA_DIR="${DATA_DIR:-$WORKDIR/data/mm_T100}"
ALPHABET="${ALPHABET:-pm1}"

# Your training script
PYFILE="${PYFILE:-$WORKDIR/train_mamba.py}"

# Args (must match argparse in train_mamba.py)
SPLITS="${SPLITS:-train,val_bin0,test_bin0,test_bin1,test_bin2}"
MAX_LEN="${MAX_LEN:-0}"              # 0 => auto infer (max_T+1) in python

D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
MLP_HIDDEN_MULT="${MLP_HIDDEN_MULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"

MAMBA_D_STATE="${MAMBA_D_STATE:-64}"
MAMBA_D_CONV="${MAMBA_D_CONV:-4}"
MAMBA_EXPAND="${MAMBA_EXPAND:-2}"
MAMBA_FAST_PATH="${MAMBA_FAST_PATH:-0}"   # 1 => add --mamba_fast_path
PACK="${PACK:-none}"                      # none/group

EPOCHS="${EPOCHS:-999}"
MAX_STEPS="${MAX_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
WD="${WD:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PATIENCE="${PATIENCE:-30}"
SEED="${SEED:-0}"

NUM_WORKERS="${NUM_WORKERS:-2}"

AMP="${AMP:-1}"                # 1 => --amp
AMP_DTYPE="${AMP_DTYPE:-bf16}" # bf16/fp16

EARLY_STOP="${EARLY_STOP:-f1}"       # loss/f1
POS_WEIGHT="${POS_WEIGHT:-auto}"     # auto or float
SAVE_PATH="${SAVE_PATH:-ckpt_mamba_nomod_stepwise.pt}"

# Build knobs (limit parallel compiles)
MAX_JOBS="${MAX_JOBS:-4}"
export MAX_JOBS

# -----------------------
# setup
# -----------------------
echo "[1/9] Prepare /tmp dirs: $TMP_BASE"
mkdir -p "$TMP_BASE" "$PIP_CACHE_DIR_LOCAL" "$TMPDIR_LOCAL"

echo "[2/9] Create/reuse venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/9] Force pip cache/tmp to /tmp"
export PIP_CACHE_DIR="$PIP_CACHE_DIR_LOCAL"
export TMPDIR="$TMPDIR_LOCAL"

echo "[4/9] Upgrade pip tooling + build tools"
python -m pip install --no-cache-dir -U pip setuptools wheel packaging ninja cmake

echo "[5/9] Install torch (cu118 pinned) if missing"
if python - <<'PY'
import importlib, sys
try:
    importlib.import_module("torch")
    import torch
    print("torch already installed:", torch.__version__)
    sys.exit(0)
except Exception as e:
    print("torch not installed:", e)
    sys.exit(1)
PY
then
  echo "[5/9] torch present"
else
  echo "[5/9] Installing torch from cu118 wheels..."
  # torchvision==0.20.1 / torchaudio==2.5.1 have been pruned from the cu118
  # wheel index and neither is used by train_mamba.py -- install torch only.
  python -m pip install --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/${CU_INDEX}" \
    torch==2.5.1
fi

echo "[5.5/9] Verify torch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

echo "[6/9] Install deps (numpy/tqdm)"
python -m pip install --no-cache-dir -U numpy tqdm

echo "[7/9] Install Mamba deps (no build isolation; low-memory friendly)"
# causal-conv1d first (may fail on some nodes; we'll try and continue)
python -m pip install --no-cache-dir --no-build-isolation "causal-conv1d==1.6.0" || true

# then mamba-ssm
python -m pip install --no-cache-dir --no-build-isolation "mamba-ssm==2.3.0" || true

# fallback extra (pulls causal-conv1d dependency)
if ! python -c "import mamba_ssm" >/dev/null 2>&1; then
  echo "[fallback] try mamba-ssm[causal-conv1d]==2.3.0 (no-build-isolation)"
  python -m pip install --no-cache-dir --no-build-isolation "mamba-ssm[causal-conv1d]==2.3.0"
fi

echo "[7.5/9] Verify mamba imports"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "torch.version.cuda:", torch.version.cuda)
import causal_conv1d
print("causal_conv1d ok:", getattr(causal_conv1d, "__version__", "unknown"))
import mamba_ssm
print("mamba_ssm ok:", getattr(mamba_ssm, "__version__", "unknown"))
from mamba_ssm.modules.mamba_simple import Mamba
print("Mamba class:", Mamba)
PY

echo "[8/9] Run training"
FAST_FLAG=()
if [[ "${MAMBA_FAST_PATH}" == "1" ]]; then
  FAST_FLAG+=(--mamba_fast_path)
fi

AMP_FLAG=()
if [[ "${AMP}" == "1" ]]; then
  AMP_FLAG+=(--amp --amp_dtype "${AMP_DTYPE}")
fi

CMD=(python3 -u train_mamba.py \
  --data_dir "$DATA_DIR" \
  --cuda --amp --amp_dtype bf16 \
  --d_model 256 --layers 2 --dropout 0.1 \
  --mamba_d_state 64 --mamba_d_conv 4 --mamba_expand 2 --pack group \
  --batch_size 256 --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 \
  --max_steps 60000 --patience 30 --early_stop acc \
  --save_path ckpt_mamba_imm_nmod_T100.pt

)

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "[9/9] Done. venv=$VENV_DIR"
