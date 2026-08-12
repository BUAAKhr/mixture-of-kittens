#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${MASTER_ADDR:?set MASTER_ADDR to node 0's bootstrap address}"
: "${NCCL_SOCKET_IFNAME:?set NCCL_SOCKET_IFNAME explicitly}"
: "${NCCL_IB_HCA:?set NCCL_IB_HCA explicitly}"
: "${NCCL_IB_GID_INDEX:?set NCCL_IB_GID_INDEX explicitly}"

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi

MASTER_PORT="${MASTER_PORT:-29500}"
BACKEND_LABEL="${BACKEND_LABEL:-nccl-native}"
RESULT_TAG="${RESULT_TAG:-manual}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/pdl_gemm_ar_results/${RESULT_TAG}}"
TRIAL_ID="${TRIAL_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "${RESULT_ROOT}"

if [[ ! "${TRIAL_ID}" =~ ^[0-9]+$ ]]; then
  echo "TRIAL_ID must be a non-negative integer" >&2
  exit 2
fi
RUN_RANDOM_SEED="${RANDOM_SEED:-$((20260812 + TRIAL_ID))}"

# Preflight and approval must observe the exact communication environment used
# by torchrun, including defaults supplied by this launcher.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX

LOCK_FILE="${TK_BROKER_LOCK_FILE:-/tmp/pdl_gemm_ar_tk_broker.lock}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another TKParallelTensor experiment holds ${LOCK_FILE}" >&2
  exit 3
fi

"${PYTHON_BIN}" -m benchmarks.pdl_gemm_ar.multinode.preflight \
  --backend-label "${BACKEND_LABEL}" \
  --node-rank "${NODE_RANK}" \
  --output "${RESULT_ROOT}/preflight_node${NODE_RANK}.json"

if [[ "${RUN_BENCHMARK:-0}" != "1" ]]; then
  echo "Preflight complete. Set RUN_BENCHMARK=1 only after reviewing both nodes."
  exit 0
fi

: "${PREFLIGHT_APPROVAL_FILE:?set PREFLIGHT_APPROVAL_FILE to the reviewed comparison JSON}"
"${PYTHON_BIN}" -m benchmarks.pdl_gemm_ar.multinode.verify_approval \
  "${PREFLIGHT_APPROVAL_FILE}" \
  --current-preflight "${RESULT_ROOT}/preflight_node${NODE_RANK}.json" \
  --backend-label "${BACKEND_LABEL}" \
  --node-rank "${NODE_RANK}" \
  --max-age-seconds "${PREFLIGHT_APPROVAL_MAX_AGE_SECONDS:-3600}"

RUN_PHASE="${RUN_PHASE:-latency}"
case "${RUN_PHASE}" in
  correctness)
    RUN_MODULE="benchmarks.pdl_gemm_ar.multinode.pk_benchmark"
    DEFAULT_WINDOW_TILES="16"
    DEFAULT_MAX_INFLIGHT="1"
    RUN_EXTRA_ARGS=(
      --modes "${MODES:-flat,default,pdl_grid,pdl_tile}"
      --correctness-only
      --randomize
      --random-seed "${RUN_RANDOM_SEED}"
    )
    ;;
  latency)
    RUN_MODULE="benchmarks.pdl_gemm_ar.multinode.pk_benchmark"
    DEFAULT_WINDOW_TILES="1,2,4,8,16"
    DEFAULT_MAX_INFLIGHT="1,2,4,8"
    RUN_EXTRA_ARGS=(
      --modes "${MODES:-flat,default,pdl_grid,pdl_tile}"
      --randomize
      --random-seed "${RUN_RANDOM_SEED}"
    )
    ;;
  instrument)
    RUN_MODULE="benchmarks.pdl_gemm_ar.multinode.pk_benchmark"
    DEFAULT_WINDOW_TILES="16"
    DEFAULT_MAX_INFLIGHT="1"
    RUN_EXTRA_ARGS=(
      --modes "${MODES:-pdl_tile}"
      --instrument
      --correctness-only
      --randomize
      --random-seed "${RUN_RANDOM_SEED}"
    )
    ;;
  stages)
    RUN_MODULE="benchmarks.pdl_gemm_ar.multinode.stage_decomposition"
    DEFAULT_WINDOW_TILES="16"
    DEFAULT_MAX_INFLIGHT="1"
    RUN_EXTRA_ARGS=()
    ;;
  *)
    echo "RUN_PHASE must be correctness, latency, instrument, or stages" >&2
    exit 4
    ;;
esac

timeout "${RUN_TIMEOUT:-30m}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m "${RUN_MODULE}" \
  --m "${M:-8192}" \
  --k "${K:-1024}" \
  --n "${N:-8192}" \
  --window-tiles "${WINDOW_TILES:-${DEFAULT_WINDOW_TILES}}" \
  --max-inflight "${MAX_INFLIGHT:-${DEFAULT_MAX_INFLIGHT}}" \
  --num-pack-ctas "${NUM_PACK_CTAS:-20}" \
  --pack-threads "${PACK_THREADS:-1024}" \
  --unpack-threads "${UNPACK_THREADS:-1024}" \
  --ready-timeout-ms "${READY_TIMEOUT_MS:-30000}" \
  --warmups "${WARMUPS:-20}" \
  --iterations "${ITERATIONS:-100}" \
  --trial-id "${TRIAL_ID}" \
  --backend-label "${BACKEND_LABEL}" \
  --output "${RESULT_ROOT}/${RUN_PHASE}_trial${TRIAL_ID}_global_rank0.jsonl" \
  "${RUN_EXTRA_ARGS[@]}"
