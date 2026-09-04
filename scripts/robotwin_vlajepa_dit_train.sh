#!/bin/bash
#PBS -N vla_jepa_robotwin_dit_30k_bs64_coef0
#PBS -P gcg51472
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -k oe
#PBS -o /home/aci18750xd/output_log/vla_jepa_robotwin_dit_o_2.log
#PBS -e /home/aci18750xd/output_log/vla_jepa_robotwin_dit_e_2.log


set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
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

CONFIG_YAML="${CONFIG_YAML:-./scripts/config/robotwin_vlajepa_dit_train.yaml}"

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-$HOME/VLA-JEPA/dataset/RoboTwin-Randomized}"

RUN_ID="${RUN_ID:-robotwin_vlajepa_dit_train_30k_bs64_coef0}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-}"
DATA_MIX="${DATA_MIX:-robotwin_all_50}"
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

# ABCI supplies per-job local temporary storage through PBS_LOCALDIR.
export TMPDIR="${PBS_LOCALDIR:-$HOME/tmp/${PBS_JOBID:-manual}}"
mkdir -p "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=8

# Do not hard-code WANDB_API_KEY here.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-vla-jepa-robotwin}"
export WANDB_ENTITY="${WANDB_ENTITY:-pjt-sbr}"

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

[[ -f "$CONFIG_YAML" ]] || {
    echo "Missing config: $CONFIG_YAML" >&2
    exit 1
}

[[ -d "$ROBOTWIN_DATA_ROOT/Clean" ]] || {
    echo "Missing RoboTwin Clean data under: $ROBOTWIN_DATA_ROOT" >&2
    exit 1
}

[[ -d "$ROBOTWIN_DATA_ROOT/Randomized" ]] || {
    echo "Missing RoboTwin Randomized data under: $ROBOTWIN_DATA_ROOT" >&2
    exit 1
}

# ------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------

TRAIN_OVERRIDES=(
    --datasets.vla_data.data_root_dir "$ROBOTWIN_DATA_ROOT"
    --datasets.vla_data.data_mix "$DATA_MIX"
)

if [[ -n "$RUN_ID" ]]; then
    TRAIN_OVERRIDES+=(--run_id "$RUN_ID")
fi

if [[ -n "$RUN_ROOT_DIR" ]]; then
    TRAIN_OVERRIDES+=(--run_root_dir "$RUN_ROOT_DIR")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    [[ -d "$RESUME_FROM_CHECKPOINT" ]] || {
        echo "Missing checkpoint directory: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    }

    TRAIN_OVERRIDES+=(
        --trainer.is_resume true
        --trainer.resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
    )
fi

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

accelerate launch \
    --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 \
    ./starVLA/training/train_starvla.py \
    --config_yaml "$CONFIG_YAML" \
    "${TRAIN_OVERRIDES[@]}"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="