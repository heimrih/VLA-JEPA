#!/bin/bash
#PBS -N real_widowxai_ft
#PBS -P gcg51472
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -k oe

set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

# conda-pack activation is not compatible with nounset (`set -u`)
set +u
source "$HOME/conda_envs/VLA_JEPA/bin/activate"
set -u

REPO_ROOT="$HOME/VLA-JEPA"
cd "$REPO_ROOT"

echo "=================================================="
echo "Job ID:       ${PBS_JOBID:-unknown}"
echo "Hostname:     $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "Python:       $(which python)"
echo "=================================================="

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CONFIG_YAML="${CONFIG_YAML:-./scripts/config/vlajepa_widowx.yaml}"

# ABCI paths corresponding to the /po2 paths in the YAML
QWEN_MODEL="${QWEN_MODEL:-$HOME/VLA-JEPA/model/Qwen3-VL-2B-Instruct}"

VJEPA_MODEL="${VJEPA_MODEL:-$HOME/VLA-JEPA/model/vjepa2-vitl-fpc64-256}"

WIDOWX_DATA_ROOT="${WIDOWX_DATA_ROOT:-$HOME/VLA-JEPA/checkpoints/vla-jepa-pretrain/vla-jepa/real_widowxai_success_only}"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$HOME/VLA-JEPA/checkpoints/vla-jepa-pretrain/vla-jepa/molmoact2-ssv2-absolute-pretrain-evalfix-v2/checkpoints/steps_50000_pytorch_model.pt}"

# These match the YAML defaults but can still be overridden when calling qsub
RUN_ID="${RUN_ID:-real_widowxai_ft}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$HOME/VLA-JEPA/checkpoints}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# ------------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------------

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# ABCI per-job local temporary storage
export TMPDIR="${PBS_LOCALDIR:-$HOME/tmp/${PBS_JOBID:-manual}}"
mkdir -p "$TMPDIR"

# Keep Triton cache on local job storage as well
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TRITON_CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=8

# W&B authentication comes from ~/.netrc
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_RobotFT}"
export WANDB_ENTITY="${WANDB_ENTITY:-pjt-sbr}"

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

[[ -f "$CONFIG_YAML" ]] || {
    echo "ERROR: Missing config:"
    echo "  $CONFIG_YAML" >&2
    exit 1
}

[[ -d "$QWEN_MODEL" ]] || {
    echo "ERROR: Missing Qwen model:"
    echo "  $QWEN_MODEL" >&2
    exit 1
}

[[ -d "$VJEPA_MODEL" ]] || {
    echo "ERROR: Missing V-JEPA model:"
    echo "  $VJEPA_MODEL" >&2
    exit 1
}

[[ -d "$WIDOWX_DATA_ROOT" ]] || {
    echo "ERROR: Missing WidowX dataset:"
    echo "  $WIDOWX_DATA_ROOT" >&2
    exit 1
}

[[ -f "$PRETRAINED_CHECKPOINT" ]] || {
    echo "ERROR: Missing pretrained checkpoint:"
    echo "  $PRETRAINED_CHECKPOINT" >&2
    exit 1
}

mkdir -p "$RUN_ROOT_DIR"

# ------------------------------------------------------------------
# Print resolved configuration
# ------------------------------------------------------------------

echo
echo "================ Training configuration ================"
echo "Config YAML:           $CONFIG_YAML"
echo "Run ID:                $RUN_ID"
echo "Run root:              $RUN_ROOT_DIR"
echo "Dataset:               $WIDOWX_DATA_ROOT"
echo "Qwen model:            $QWEN_MODEL"
echo "V-JEPA model:          $VJEPA_MODEL"
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Resume checkpoint:     ${RESUME_FROM_CHECKPOINT:-none}"
echo "========================================================"
echo

# ------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------

TRAIN_OVERRIDES=(
    --run_id "$RUN_ID"
    --run_root_dir "$RUN_ROOT_DIR"

    --framework.qwenvl.base_vlm "$QWEN_MODEL"
    --framework.vj2_model.base_encoder "$VJEPA_MODEL"

    --datasets.vla_data.data_root_dir "$WIDOWX_DATA_ROOT"
    --datasets.vla_data.data_mix "real_widowxai_success_only"

    --trainer.pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
)

# Resume from a training-state checkpoint if supplied.
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    [[ -d "$RESUME_FROM_CHECKPOINT" ]] || {
        echo "ERROR: Missing resume checkpoint directory:"
        echo "  $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    }

    TRAIN_OVERRIDES+=(
        --trainer.is_resume true
        --trainer.resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
    )
fi

# ------------------------------------------------------------------
# GPU information
# ------------------------------------------------------------------

echo "Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

accelerate launch \
    --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 \
    ./starVLA/training/train_starvla.py \
    --config_yaml "$CONFIG_YAML" \
    "${TRAIN_OVERRIDES[@]}"

echo
echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="