#!/bin/bash
#SBATCH --job-name=vla_jepa_robotwin
#SBATCH --partition=48-8
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --nodelist="yagi40"
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output=/po2/heimrih/thesis/output_log/vla_jepa_robotwin-%j.out
#SBATCH --error=/po2/heimrih/thesis/output_log/vla_jepa_robotwin-%j.err

set -eo pipefail

source /home/heimrih/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
set -u

REPO_ROOT=/po2/heimrih/VLA-JEPA
cd "${REPO_ROOT}"

ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-/po2/heimrih/VLA-JEPA/dataset/RoboTwin-Randomized}
RUN_ID=${RUN_ID:-robotwin_qwenoft_all50}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-checkpoints}
DATA_MIX=${DATA_MIX:-robotwin_all_50}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/po2/heimrih/VLA-JEPA/dataset-local/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "${TMPDIR}"

export WANDB_MODE=${WANDB_MODE:-online}

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/robotwin_qwenoft_train.yaml \
  --datasets.vla_data.data_root_dir "${ROBOTWIN_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
