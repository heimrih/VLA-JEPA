#!/bin/bash
#SBATCH --job-name=qwengroot_dit_jepa
#SBATCH --partition=48-8
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --nodelist="yagi40"
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --output=/po2/heimrih/matsuo/output_log/qwengroot_dit_jepa-%j.out
#SBATCH --error=/po2/heimrih/matsuo/output_log/qwengroot_dit_jepa-%j.err

set -eo pipefail

source /home/heimrih/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
set -u

REPO_ROOT=/po2/heimrih/VLA-JEPA
cd "${REPO_ROOT}"

CONFIG_YAML=${CONFIG_YAML:-./scripts/config/robotwin_qwengroot_dit_train.yaml}
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/dataset/RoboTwin-Randomized}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-checkpoints}
DATA_MIX=${DATA_MIX:-robotwin_all_50}
run_stamp=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ID=${RUN_ID:-robotwin_qwengroot_dit_jepa_init_${run_stamp}}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=${REPO_ROOT}/dataset-local/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p "${TMPDIR}" /po2/heimrih/matsuo/output_log

echo "[INFO] config=${CONFIG_YAML}"
echo "[INFO] output=${RUN_ROOT_DIR}/${RUN_ID}"
echo "[INFO] data_mix=${DATA_MIX}"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.data_root_dir "${ROBOTWIN_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
