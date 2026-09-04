#!/bin/bash
#SBATCH --job-name=vla_jepa_robotwin_dit_eval
#SBATCH --partition=48-4
#SBATCH --gres=gpu:3
#SBATCH --mem=100G
#SBATCH --nodelist="yagi34"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=/po2/heimrih/matsuo/output_log/vla_jepa_robotwin_dit_eval-%j.out
#SBATCH --error=/po2/heimrih/matsuo/output_log/vla_jepa_robotwin_dit_eval-%j.err

set -eo pipefail

source /home/heimrih/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
set -u

REPO_ROOT=/po2/heimrih/VLA-JEPA
cd "${REPO_ROOT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SAPIEN_RENDERER=egl
export TMPDIR=${REPO_ROOT}/dataset-local/tmp
mkdir -p "${TMPDIR}"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-${REPO_ROOT}/RoboTwin}
export STARVLA_PYTHON=${STARVLA_PYTHON:-/home/heimrih/miniconda3/envs/VLA_JEPA/bin/python}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/home/heimrih/miniconda3/envs/robotwin/bin/python}
export PATH="$(dirname "${ROBOTWIN_PYTHON}"):${PATH}"
export ROBOTWIN_POLICY_NAME=model2robotwin_dit_interface
export ROBOTWIN_USE_BF16=${ROBOTWIN_USE_BF16:-1}
export ROBOTWIN_JOBS_PER_GPU=1
export ROBOTWIN_SERVER_TIMEOUT=${ROBOTWIN_SERVER_TIMEOUT:-900}

ckpt=${ROBOTWIN_CKPT:-${REPO_ROOT}/checkpoints/robotwin_vlajepa_dit_v2/checkpoints/steps_90000_pytorch_model.pt}
mode=${ROBOTWIN_MODE:-demo_randomized}
run_name=${ROBOTWIN_RUN_NAME:-robotwin_vlajepa_dit_v2_steps90000}
tasks=${ROBOTWIN_TASKS:-all}
run_stamp=${ROBOTWIN_RUN_STAMP:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}

if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Checkpoint does not exist: ${ckpt}" >&2
    exit 1
fi

# Fail before launching tasks if any allocated GPU cannot initialize CUDA.
if [[ "${ROBOTWIN_GPU_PREFLIGHT:-1}" == "1" ]]; then
    "${STARVLA_PYTHON}" - <<'PY'
import sys
import torch

failures = []
for index in range(torch.cuda.device_count()):
    try:
        with torch.cuda.device(index):
            tensor = torch.empty(1, device=f"cuda:{index}")
            del tensor
            torch.cuda.synchronize(index)
        print(f"[INFO] CUDA preflight passed: gpu={index} name={torch.cuda.get_device_name(index)}")
    except Exception as exc:
        failures.append(f"gpu={index}: {exc}")

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    failures.append("no CUDA device is visible")
if failures:
    print("[ERROR] CUDA preflight failed; no evaluation tasks were launched:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
PY
fi

if [[ -z "${ROBOTWIN_BASE_PORT:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        export ROBOTWIN_BASE_PORT=$((12000 + SLURM_JOB_ID % 40000))
    else
        export ROBOTWIN_BASE_PORT=15694
    fi
fi

log_parent=${ROBOTWIN_LOG_PARENT:-${REPO_ROOT}/results/robotwin_eval}
export ROBOTWIN_LOG_ROOT=${ROBOTWIN_LOG_ROOT:-${log_parent}/${run_name}_${mode}_${run_stamp}}

if [[ "${ROBOTWIN_UPDATE_EMBODIMENT_CONFIG:-1}" == "1" ]]; then
    (
        cd "${ROBOTWIN_PATH}"
        "${ROBOTWIN_PYTHON}" script/update_embodiment_config_path.py
    )
fi

bash "${REPO_ROOT}/examples/Robotwin/eval_files/start_eval.sh" \
    --mode "${mode}" \
    --name "${run_name}" \
    --ckpt "${ckpt}" \
    ${tasks}
