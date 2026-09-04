#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-${REPO_ROOT}/RoboTwin}"
robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
if [[ ! -f "${robotwin_eval_script}" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_eval_script}" >&2
    exit 1
fi

policy_name="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-vlajepa_robotwin}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

cd "${ROBOTWIN_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    --overrides \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
