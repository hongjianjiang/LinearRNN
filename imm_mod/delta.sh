#!/bin/bash
#SBATCH -p gpu-a40
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J immmod_deltanet
#SBATCH -o delta.log
set -euo pipefail

# -----------------------
# User knobs (env overrides)
# -----------------------
# Shared venv, built once by LinearRNN/.setup_venv.sh
# (py3.12, torch 2.5.1+cu118, fla 0.4.1, mamba-ssm 2.3.0)
VENV_DIR="${VENV_DIR:-/scratch/inf0/user/hongjian/LinearRNN/.venv}"

# fla JIT-compiles triton kernels; keep that cache node-local instead of ~/.triton
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"

DATA_DIR="${DATA_DIR:-$PWD/data/mm_T100_bins}"
TRAIN_PY="${TRAIN_PY:-$PWD/train_deltanet.py}"
SAVE_PATH="${SAVE_PATH:-$PWD/ckpt_deltanet_stepwise.pt}"
M_MAX="${M_MAX:-29}"
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# 0 (default) => input-only tokenization (no leaked STATE token)
# 1 => leaky --teacher_force_state ablation (see train_deltanet.py)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# model / train hyperparams
D_MODEL="${D_MODEL:-256}"
HEADS="${HEADS:-4}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
AUX_W="${AUX_W:-0.0}"

# 0 (default) => beta_t in (0,1): cannot represent negative-eigenvalue
# transitions (e.g. parity, odd permutations/transpositions).
# 1 => beta_t in (0,2), see Grazzi et al. 2025 (cited in the paper this
# benchmark is from): required for DeltaNet to recognize parity.
ALLOW_NEG_EIGVAL="${ALLOW_NEG_EIGVAL:-0}"

MLP_HIDDEN_MULT="${MLP_HIDDEN_MULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"

BATCH_SIZE="${BATCH_SIZE:-256}"
# with n_train~70000 and batch_size=256, ~275 steps/epoch; 200 epochs was the
# silent binding stop condition in imm_unbound (see project memory) -- raise
# it so max_steps/patience actually govern.
EPOCHS="${EPOCHS:-400}"
MAX_STEPS="${MAX_STEPS:-60000}"
LR="${LR:-3e-4}"
WD="${WD:-0.005}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

PATIENCE="${PATIENCE:-30}"
EARLY_STOP="${EARLY_STOP:-acc}"    # loss / acc (this script's choices)

# tokenization / safety
MAX_LEN="${MAX_LEN:-0}"            # 0 => auto infer (recommended)

# perf
NUM_WORKERS="${NUM_WORKERS:-2}"

# AMP
AMP="${AMP:-1}"                    # 1 => enable --amp
AMP_DTYPE="${AMP_DTYPE:-bf16}"     # bf16 or fp16 (must match your script)

echo "[1/3] Activate venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR (build it with LinearRNN/.setup_venv.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
mkdir -p "$TRITON_CACHE_DIR"

echo "[2/3] Environment"
python - <<'PY'
import sys
import torch, numpy as np
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy :", np.__version__)
try:
    import fla
    print("fla   :", getattr(fla, "__version__", "unknown"))
except Exception as e:
    print("fla   : not importable:", e)
PY

echo "[3/3] Run training"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "ERROR: TRAIN_PY not found: $TRAIN_PY" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

CMD=(
python3 -u "$TRAIN_PY" \
  --data_dir "$DATA_DIR" \
  --alphabet "$ALPHABET" --target_mode multiclass --m_max "$M_MAX" \
  --cuda \
  --d_model "$D_MODEL" --heads "$HEADS" --layers "$LAYERS" --dropout "$DROPOUT" \
  --aux_w "$AUX_W" \
  --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WD" --grad_clip "$GRAD_CLIP" \
  --epochs "$EPOCHS" --max_steps "$MAX_STEPS" --patience "$PATIENCE" --early_stop "$EARLY_STOP" --seed 0 \
  --save_path "$SAVE_PATH" \
  --eval_log delta_stepwise_eval.log
)

# optional args
if [[ "$MAX_LEN" != "0" ]]; then
  CMD+=(--max_len "$MAX_LEN")
fi
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi
if [[ "$ALLOW_NEG_EIGVAL" == "1" ]]; then
  CMD+=(--allow_neg_eigval)
fi

# AMP flags MUST be inside CMD (this fixes your error)
if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp --amp_dtype "$AMP_DTYPE")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Saved: $SAVE_PATH"
