#!/bin/bash
#SBATCH --job-name=vla_jepa_cotrain
#SBATCH --partition=48-6
#SBATCH --gres=gpu:6
#SBATCH --mem=200G
#SBATCH --nodelist="yagi41"
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output=/po2/heimrih/thesis/output_log/vla_jepa_cotrain-%j.out
#SBATCH --error=/po2/heimrih/thesis/output_log/vla_jepa_cotrain-%j.err

set -eo pipefail

source /home/heimrih/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
set -u

REPO_ROOT=/po2/heimrih/VLA-JEPA
cd "${REPO_ROOT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TMPDIR=/po2/heimrih/VLA-JEPA/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "${TMPDIR}"

export WANDB_MODE=${WANDB_MODE:-online}

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 6 \
  ./starVLA/training/train_vlajepa_cotrain.py \
  --config_yaml ./scripts/config/vlajepa_cotrain.yaml
