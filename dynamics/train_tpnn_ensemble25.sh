#!/usr/bin/env bash
#
# Train a 25-member bootstrapped TPNN ensemble on GPUs 0-4.
#
# Each GPU processes five seeds sequentially.  Within each seed, Stage 1 (MSE)
# must finish before Stage 2 (frozen mean/shared model, NLL log-variance head)
# starts.  Existing valid checkpoints are reused, while incomplete or ambiguous
# checkpoint layouts are left untouched for manual inspection.
#
# Usage:
#   bash dynamics/train_tpnn_ensemble25.sh
#
# Optional overrides:
#   GPU_IDS="0 1" NUM_WORKERS=4 bash dynamics/train_tpnn_ensemble25.sh
#   PYTHON_BIN=/path/to/python bash dynamics/train_tpnn_ensemble25.sh

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-/data/datasets/fusion_new/bin/python}
STEP1_CONFIG=${STEP1_CONFIG:-tpnn_noshape_gas_benchmark_step_one_mse}
STEP2_CONFIG=${STEP2_CONFIG:-tpnn_noshape_gas_benchmark_step_two_logvar}

STEP1_ROOT=${STEP1_ROOT:-/data/models/tpnn_noshape_gas_benchmark_ensemble25_step_one_mse}
STEP2_ROOT=${STEP2_ROOT:-/data/models/tpnn_noshape_gas_benchmark_ensemble25_step_two_logvar}
LOG_ROOT=${LOG_ROOT:-/data/models/tpnn_noshape_gas_benchmark_ensemble25_logs}

NUM_MODELS=${NUM_MODELS:-25}
NUM_WORKERS=${NUM_WORKERS:-4}
MIN_CHECKPOINT_EPOCH=${MIN_CHECKPOINT_EPOCH:-5}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4"}

read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS=${#GPUS[@]}

if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "No GPUs were provided in GPU_IDS."
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable not found: ${PYTHON_BIN}"
    exit 1
fi

mkdir -p "${STEP1_ROOT}" "${STEP2_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}" || exit 1

has_any_checkpoint() {
    local seed_dir=$1
    find "${seed_dir}" -type f -name '*.ckpt' -print -quit 2>/dev/null | grep -q .
}

has_valid_checkpoint() {
    local seed_dir=$1
    local checkpoint
    local epoch

    while IFS= read -r checkpoint; do
        epoch=$(basename "${checkpoint}" | sed -n 's/^epoch=\([0-9][0-9]*\)-.*/\1/p')
        if [ -n "${epoch}" ] && [ "${epoch}" -ge "${MIN_CHECKPOINT_EPOCH}" ]; then
            return 0
        fi
    done < <(find "${seed_dir}" -type f -name '*.ckpt' 2>/dev/null)

    return 1
}

has_ambiguous_checkpoint_layout() {
    local seed_dir=$1
    local checkpoint_dir
    local checkpoint_dir_count=0

    [ -d "${seed_dir}" ] || return 1

    while IFS= read -r checkpoint_dir; do
        checkpoint_dir_count=$((checkpoint_dir_count + 1))
        if ! find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' \
            -print -quit 2>/dev/null | grep -q .; then
            return 0
        fi
    done < <(find "${seed_dir}" -type d -name checkpoints 2>/dev/null)

    [ "${checkpoint_dir_count}" -gt 1 ]
}

validate_existing_seed_dir() {
    local label=$1
    local seed_dir=$2

    if has_ambiguous_checkpoint_layout "${seed_dir}"; then
        echo "${label} has an empty or ambiguous checkpoints layout: ${seed_dir}"
        echo "Leaving it untouched; inspect it before rerunning this seed."
        return 1
    fi

    if has_any_checkpoint "${seed_dir}" && ! has_valid_checkpoint "${seed_dir}"; then
        echo "${label} contains only an incomplete checkpoint: ${seed_dir}"
        echo "Leaving it untouched; inspect it before rerunning this seed."
        return 1
    fi

    return 0
}

train_stage1() {
    local seed=$1
    local gpu=$2

    "${PYTHON_BIN}" dynamics/train_dynamics_rohit.py \
        --config-name "${STEP1_CONFIG}" \
        seed="${seed}" \
        cuda_device="${gpu}" \
        save_path="${STEP1_ROOT}" \
        data_module.bootstrap=true \
        data_module.seed="${seed}" \
        data_module.num_workers="${NUM_WORKERS}"
}

train_stage2() {
    local seed=$1
    local gpu=$2

    "${PYTHON_BIN}" dynamics/train_dynamics_rohit.py \
        --config-name "${STEP2_CONFIG}" \
        seed="${seed}" \
        cuda_device="${gpu}" \
        save_path="${STEP2_ROOT}" \
        model.load_dir="${STEP1_ROOT}" \
        data_module.bootstrap=true \
        data_module.seed="${seed}" \
        data_module.num_workers="${NUM_WORKERS}"
}

run_member() {
    local seed=$1
    local gpu=$2
    local step1_dir="${STEP1_ROOT}/${seed}"
    local step2_dir="${STEP2_ROOT}/${seed}"
    local log="${LOG_ROOT}/seed_${seed}.log"

    {
        echo "[$(date)] TPNN ensemble member ${seed} assigned to GPU ${gpu}"
        echo "Stage-1 root: ${STEP1_ROOT}"
        echo "Stage-2 root: ${STEP2_ROOT}"
        echo "bootstrap=true data_module.seed=${seed}"

        if ! validate_existing_seed_dir "Stage 1 seed ${seed}" "${step1_dir}"; then
            return 1
        fi
        if ! validate_existing_seed_dir "Stage 2 seed ${seed}" "${step2_dir}"; then
            return 1
        fi

        if has_valid_checkpoint "${step2_dir}"; then
            echo "[$(date)] Valid Stage-2 checkpoint already exists; skipping seed ${seed}."
            return 0
        fi

        if has_valid_checkpoint "${step1_dir}"; then
            echo "[$(date)] Reusing the valid Stage-1 checkpoint for seed ${seed}."
        else
            echo "[$(date)] Starting Stage 1 (MSE) for seed ${seed}."
            if ! train_stage1 "${seed}" "${gpu}"; then
                echo "[$(date)] Stage 1 failed for seed ${seed}."
                return 1
            fi
            if ! has_valid_checkpoint "${step1_dir}"; then
                echo "[$(date)] Stage 1 produced no valid checkpoint for seed ${seed}."
                return 1
            fi
        fi

        echo "[$(date)] Starting Stage 2 (NLL log-variance) for seed ${seed}."
        if ! train_stage2 "${seed}" "${gpu}"; then
            echo "[$(date)] Stage 2 failed for seed ${seed}."
            return 1
        fi
        if ! has_valid_checkpoint "${step2_dir}"; then
            echo "[$(date)] Stage 2 produced no valid checkpoint for seed ${seed}."
            return 1
        fi

        echo "[$(date)] Seed ${seed} completed successfully."
        return 0
    } >"${log}" 2>&1
}

run_gpu_worker() {
    local worker_index=$1
    local gpu=${GPUS[$worker_index]}
    local seed
    local worker_status=0

    for ((seed=worker_index; seed<NUM_MODELS; seed+=NUM_GPUS)); do
        echo "[$(date)] GPU ${gpu}: starting seed ${seed}; log=${LOG_ROOT}/seed_${seed}.log"
        if run_member "${seed}" "${gpu}"; then
            echo "[$(date)] GPU ${gpu}: seed ${seed} completed or was already complete."
        else
            echo "[$(date)] GPU ${gpu}: seed ${seed} failed; check ${LOG_ROOT}/seed_${seed}.log"
            worker_status=1
        fi
    done

    return "${worker_status}"
}

echo "Training ${NUM_MODELS} bootstrapped TPNN members on GPUs: ${GPUS[*]}"
echo "At most ${NUM_GPUS} training processes will run concurrently."
echo "Each seed runs Stage 1 followed by Stage 2 on the same GPU."

pids=()

terminate_process_tree() {
    local parent_pid=$1
    local child_pid

    while IFS= read -r child_pid; do
        terminate_process_tree "${child_pid}"
    done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)

    kill -TERM "${parent_pid}" 2>/dev/null || true
}

stop_all_workers() {
    local pid

    trap - INT TERM
    echo
    echo "Stop requested; terminating all TPNN workers."
    for pid in "${pids[@]}"; do
        terminate_process_tree "${pid}"
    done
    wait 2>/dev/null || true
    echo "All launcher child processes have been stopped."
    exit 130
}

trap stop_all_workers INT TERM

for worker_index in "${!GPUS[@]}"; do
    run_gpu_worker "${worker_index}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [ "${status}" -eq 0 ]; then
    echo "All ${NUM_MODELS} TPNN ensemble members completed successfully."
    echo "Final ensemble directory: ${STEP2_ROOT}"
else
    echo "One or more members failed. Inspect logs under ${LOG_ROOT}."
fi

exit "${status}"
