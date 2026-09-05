#!/usr/bin/env bash
#SBATCH --job-name=widowx-open-loop
#SBATCH --partition=48-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --nodelist="yagi40"
#SBATCH --output=/po2/heimrih/matsuo/output_log/widowx-open-loop-%j.out
#SBATCH --error=/po2/heimrih/matsuo/output_log/widowx-open-loop-%j.err

set -eo pipefail

REPO_ROOT=${REPO_ROOT:-/po2/heimrih/VLA-JEPA}
CONDA_ROOT=${CONDA_ROOT:-}
CONDA_ENV=${CONDA_ENV:-VLA_JEPA}

if [[ -z "${CONDA_ROOT}" ]]; then
    for conda_candidate in \
        /home/heimrih/miniconda3 \
        /home/heimrih/miniforge3
    do
        if [[ -f "${conda_candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_ROOT=${conda_candidate}
            break
        fi
    done
fi

CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/checkpoints/real_widowxai_ft/final_model/pytorch_model.pt}
DATASET_ROOT=${DATASET_ROOT:-${REPO_ROOT}/checkpoints/vla-jepa-pretrain/vla-jepa/real_widowxai_test}
DATASET_NAME=${DATASET_NAME:-real_widowxai_test}
NORMALIZATION_DATASET_ROOT=${NORMALIZATION_DATASET_ROOT:-${REPO_ROOT}/checkpoints/vla-jepa-pretrain/vla-jepa/real_widowxai_success_only}
NORMALIZATION_DATASET_NAME=${NORMALIZATION_DATASET_NAME:-real_widowxai_success_only}
NUM_EXAMPLES=${NUM_EXAMPLES:-0}
SAMPLES_PER_EXAMPLE=${SAMPLES_PER_EXAMPLE:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-${REPO_ROOT}/results/real_widowxai_open_loop_${SLURM_JOB_ID}.json}
SAVE_VISUALIZATIONS=${SAVE_VISUALIZATIONS:-1}
MAX_VISUALIZATIONS=${MAX_VISUALIZATIONS:-32}
VISUALIZATION_DIR=${VISUALIZATION_DIR:-${REPO_ROOT}/results/real_widowxai_open_loop_${SLURM_JOB_ID}_visualizations}

[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] || {
    echo "Missing Conda initialization script: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    exit 1
}
[[ -f "${CHECKPOINT}" ]] || {
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
}
[[ -d "${DATASET_ROOT}/${DATASET_NAME}" ]] || {
    echo "Missing evaluation dataset: ${DATASET_ROOT}/${DATASET_NAME}" >&2
    exit 1
}

mkdir -p "${REPO_ROOT}/results" "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/vlajepa-matplotlib-${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Evaluation dataset: ${DATASET_ROOT}/${DATASET_NAME}"
echo "Normalization dataset: ${NORMALIZATION_DATASET_ROOT}/${NORMALIZATION_DATASET_NAME}"
echo "Examples: ${NUM_EXAMPLES}; samples per example: ${SAMPLES_PER_EXAMPLE}"
echo "Metrics output: ${OUTPUT}"
echo "Visualization output: ${VISUALIZATION_DIR} (enabled=${SAVE_VISUALIZATIONS})"
nvidia-smi

VISUALIZATION_ARGS=()
if [[ "${SAVE_VISUALIZATIONS}" != "0" ]]; then
    VISUALIZATION_ARGS+=(
        --save-visualizations
        --visualization-dir "${VISUALIZATION_DIR}"
        --max-visualizations "${MAX_VISUALIZATIONS}"
    )
fi

python examples/RealWidowXAI/open_loop_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --normalization-dataset-root "${NORMALIZATION_DATASET_ROOT}" \
    --normalization-dataset-name "${NORMALIZATION_DATASET_NAME}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-examples "${NUM_EXAMPLES}" \
    --samples-per-example "${SAMPLES_PER_EXAMPLE}" \
    --seed "${SEED}" \
    --output "${OUTPUT}" \
    "${VISUALIZATION_ARGS[@]}"
