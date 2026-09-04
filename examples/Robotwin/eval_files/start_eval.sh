#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_ALL_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock
    click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot
    move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop
    open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
    place_bread_basket place_bread_skillet place_burger_fries place_can_basket
    place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan
    place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand
    place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two stack_bowls_three
    stack_bowls_two stamp_seal turn_switch
)

usage() {
    cat >&2 <<'EOF'
Usage:
  bash start_eval.sh -m <demo_clean|demo_randomized> -n <run_name> -c <ckpt_path> [options] <tasks...>

Options:
  -s, --seed              Eval seed (default: 0)
  -j, --jobs-per-gpu      Concurrent jobs per GPU (default: 1)
  -p, --base-port         First port to allocate (default: 5694)
      --server-timeout    Seconds to wait for server (default: 600)

Tasks can be names, comma-separated names, a task-list file, or `all`.
EOF
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "${value}"
}

port_in_use() {
    local port="$1"
    "${STARVLA_PYTHON:-python}" -c "import websockets.sync.client as c; ws=c.connect('ws://127.0.0.1:${port}', open_timeout=2, compression=None); ws.close()" 2>/dev/null
}

wait_for_server() {
    local port="$1"
    local timeout_s="${2:-600}"
    local server_pid="${3:-}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
            return 1
        fi
        if port_in_use "${port}"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

resolve_tasks() {
    local -a raw_inputs=("$@")
    local -a resolved=()
    local input task line
    if (( ${#raw_inputs[@]} == 1 )) && [[ -f "${raw_inputs[0]}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            [[ -n "${line}" ]] && resolved+=("${line}")
        done < "${raw_inputs[0]}"
    else
        for input in "${raw_inputs[@]}"; do
            if [[ "${input}" == "all" ]]; then
                resolved+=("${ROBOTWIN_ALL_TASKS[@]}")
                continue
            fi
            IFS=',' read -ra split_inputs <<< "${input}"
            for task in "${split_inputs[@]}"; do
                task="$(trim "${task}")"
                [[ -n "${task}" ]] && resolved+=("${task}")
            done
        done
    fi
    (( ${#resolved[@]} > 0 )) || return 1
    printf '%s\n' "${resolved[@]}"
}

detect_cuda_devices() {
    local -a devices=()
    local -a cleaned=()
    local gpu_count idx device
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra devices <<< "${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        for (( idx = 0; idx < gpu_count; ++idx )); do devices+=("${idx}"); done
    fi
    for device in "${devices[@]+"${devices[@]}"}"; do
        device="$(trim "${device}")"
        [[ -n "${device}" ]] && cleaned+=("${device}")
    done
    (( ${#cleaned[@]} > 0 )) || cleaned=(0)
    printf '%s\n' "${cleaned[@]}"
}

kill_descendants() {
    local target_pid="$1"
    local sig="${2:-TERM}"
    local child_pids cpid
    child_pids="$(ps -o pid= --ppid "${target_pid}" 2>/dev/null)" || true
    for cpid in ${child_pids}; do kill_descendants "${cpid}" "${sig}"; done
    kill -"${sig}" "${target_pid}" 2>/dev/null || true
}

cleanup_active_jobs() {
    trap '' INT TERM EXIT
    local pid
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        [[ -n "${pid}" ]] && kill_descendants "${pid}" TERM
    done
}
trap cleanup_active_jobs EXIT INT TERM

TASK_CONFIG=""
POLICY_NAME=""
CKPT_PATH=""
ROBOTWIN_SEED="${ROBOTWIN_SEED:-0}"
ROBOTWIN_JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-1}"
ROBOTWIN_BASE_PORT="${ROBOTWIN_BASE_PORT:-5694}"
ROBOTWIN_SERVER_TIMEOUT="${ROBOTWIN_SERVER_TIMEOUT:-600}"

while (( $# > 0 )); do
    case "$1" in
        -m|--mode) TASK_CONFIG="$2"; shift 2 ;;
        -n|--name) POLICY_NAME="$2"; shift 2 ;;
        -c|--ckpt) CKPT_PATH="$2"; shift 2 ;;
        -s|--seed) ROBOTWIN_SEED="$2"; shift 2 ;;
        -j|--jobs-per-gpu) ROBOTWIN_JOBS_PER_GPU="$2"; shift 2 ;;
        -p|--base-port) ROBOTWIN_BASE_PORT="$2"; shift 2 ;;
        --server-timeout) ROBOTWIN_SERVER_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) break ;;
    esac
done

[[ -n "${TASK_CONFIG}" && -n "${POLICY_NAME}" && -n "${CKPT_PATH}" ]] || { usage; exit 1; }
[[ "${TASK_CONFIG}" == "demo_clean" || "${TASK_CONFIG}" == "demo_randomized" ]] || { echo "Bad mode: ${TASK_CONFIG}" >&2; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint path does not exist: ${CKPT_PATH}" >&2; exit 1; }
(( $# > 0 )) || { echo "No tasks specified." >&2; usage; exit 1; }

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
export STARVLA_PYTHON ROBOTWIN_PYTHON

mapfile -t TASKS < <(resolve_tasks "$@")
mapfile -t CUDA_DEVICES < <(detect_cuda_devices)

TOTAL_SLOTS=$((${#CUDA_DEVICES[@]} * ROBOTWIN_JOBS_PER_GPU))
LOG_DIR="${ROBOTWIN_LOG_ROOT:-$(dirname "${CKPT_PATH}")/robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "[INFO] starvla python: ${STARVLA_PYTHON}"
echo "[INFO] robotwin python: ${ROBOTWIN_PYTHON}"
echo "[INFO] mode=${TASK_CONFIG} name=${POLICY_NAME} seed=${ROBOTWIN_SEED}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] gpus=${CUDA_DEVICES[*]} slots=${TOTAL_SLOTS}"
echo "[INFO] tasks=${TASKS[*]}"

ACTIVE_PIDS=()
FAILED_TASKS=()
next_task_idx=0
completed_tasks=0

launch_task() {
    local slot_idx="$1"
    local task_name="$2"
    local gpu_id="${CUDA_DEVICES[$((slot_idx % ${#CUDA_DEVICES[@]}))]}"
    local port=$((ROBOTWIN_BASE_PORT + slot_idx))
    local task_safe="${task_name//\//_}"
    local server_log="${LOG_DIR}/${task_safe}_${TASK_CONFIG}_gpu${gpu_id}_port${port}_server.log"
    local eval_log="${LOG_DIR}/${task_safe}_${TASK_CONFIG}_gpu${gpu_id}_port${port}_eval.log"

    echo "[INFO] Launching task=${task_name} gpu=${gpu_id} port=${port}"
    (
        set -euo pipefail
        server_pid=""
        cleanup_server() {
            [[ -n "${server_pid}" ]] && kill "${server_pid}" 2>/dev/null || true
        }
        trap cleanup_server EXIT INT TERM
        bash "${SCRIPT_DIR}/run_policy_server.sh" "${CKPT_PATH}" "${gpu_id}" "${port}" > "${server_log}" 2>&1 &
        server_pid=$!
        if ! wait_for_server "${port}" "${ROBOTWIN_SERVER_TIMEOUT}" "${server_pid}"; then
            echo "[ERROR] Policy server failed for task=${task_name}. See ${server_log}" >&2
            exit 1
        fi
        bash "${SCRIPT_DIR}/eval.sh" \
            "${task_name}" "${TASK_CONFIG}" "${POLICY_NAME}" "${ROBOTWIN_SEED}" "${gpu_id}" "${CKPT_PATH}" "${port}" \
            > >(tee "${eval_log}" | grep --line-buffered "Success rate" | sed -u "s/^/[RESULT] ${task_name}: /") 2>&1
    ) &
    ACTIVE_PIDS[$slot_idx]=$!
    ACTIVE_TASKS[$slot_idx]="${task_name}"
}

while (( completed_tasks < ${#TASKS[@]} )); do
    for (( slot_idx = 0; slot_idx < TOTAL_SLOTS; ++slot_idx )); do
        current_pid="${ACTIVE_PIDS[$slot_idx]:-}"
        if [[ -n "${current_pid}" ]] && ! kill -0 "${current_pid}" 2>/dev/null; then
            if ! wait "${current_pid}"; then FAILED_TASKS+=("${ACTIVE_TASKS[$slot_idx]}"); fi
            ACTIVE_PIDS[$slot_idx]=""
            completed_tasks=$((completed_tasks + 1))
        fi
        if [[ -z "${ACTIVE_PIDS[$slot_idx]:-}" && ${next_task_idx} -lt ${#TASKS[@]} ]]; then
            launch_task "${slot_idx}" "${TASKS[$next_task_idx]}"
            next_task_idx=$((next_task_idx + 1))
        fi
    done
    sleep 2
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] Failed tasks: ${FAILED_TASKS[*]}" >&2
    exit 1
fi

echo "[INFO] All RoboTwin eval tasks completed. Logs: ${LOG_DIR}"
