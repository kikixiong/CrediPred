#!/usr/bin/env bash
# Submit the fixed clean-UQ smoke, pretest, or sealed-test DAG.

set -euo pipefail

MODE=${1:-}
case "${MODE}" in
  smoke|pretest-representative|pretest|sealed-test-representative|sealed-test) ;;
  *)
    echo "first argument must be smoke, pretest-representative, pretest, sealed-test-representative, or sealed-test" >&2
    exit 64
    ;;
esac
shift

SOURCE_DIR=
RUN_ROOT=
PYTHON_BIN=
DATA_PT=
FIT_SELECT_TARGETS_PT=
CALIBRATE_TARGETS_PT=
SEALED_TEST_TARGETS_PT=
PARTITION_JSONL=
PROTOCOL_JSON=
TARGET_LEDGER=
NUM_NODES=
REPRESENTATIVE_JOB_ID=
SEEDS=42,43,44,45,46
BASE_EPOCHS=100
CORRECTION_EPOCHS=100
BATCH_SIZE=1024
NEIGHBOR_K=30
NUM_WORKERS=4
PROBE_NODES=32768
MAX_CACHE_HOURS=20.0
LP_CHUNK_SIZE=8388608
MAX_LP_HOURS=20.0

while (( $# )); do
  case "$1" in
    --source-dir) SOURCE_DIR=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --python-bin) PYTHON_BIN=$2; shift 2 ;;
    --data-pt) DATA_PT=$2; shift 2 ;;
    --fit-select-targets-pt) FIT_SELECT_TARGETS_PT=$2; shift 2 ;;
    --calibrate-targets-pt) CALIBRATE_TARGETS_PT=$2; shift 2 ;;
    --sealed-test-targets-pt) SEALED_TEST_TARGETS_PT=$2; shift 2 ;;
    --partition-jsonl) PARTITION_JSONL=$2; shift 2 ;;
    --protocol-json) PROTOCOL_JSON=$2; shift 2 ;;
    --target-ledger) TARGET_LEDGER=$2; shift 2 ;;
    --num-nodes) NUM_NODES=$2; shift 2 ;;
    --representative-job-id) REPRESENTATIVE_JOB_ID=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --base-epochs) BASE_EPOCHS=$2; shift 2 ;;
    --correction-epochs) CORRECTION_EPOCHS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --neighbor-k) NEIGHBOR_K=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --probe-nodes) PROBE_NODES=$2; shift 2 ;;
    --max-cache-hours) MAX_CACHE_HOURS=$2; shift 2 ;;
    --lp-chunk-size) LP_CHUNK_SIZE=$2; shift 2 ;;
    --max-lp-hours) MAX_LP_HOURS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

: "${SOURCE_DIR:?--source-dir is required}"
: "${RUN_ROOT:?--run-root is required}"
: "${PYTHON_BIN:?--python-bin is required}"
test -d "${SOURCE_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${RUN_ROOT}/logs"

GPU_STAGE_SCRIPT="${SOURCE_DIR}/scripts/noether/clean_uq_stage.sbatch"
CPU_STAGE_SCRIPT="${SOURCE_DIR}/scripts/noether/clean_uq_cpu_stage.sbatch"
test -f "${GPU_STAGE_SCRIPT}"
test -f "${CPU_STAGE_SCRIPT}"

submit_job() {
  local stage_script=$1
  local name=$2
  local dependency=$3
  shift 3
  local options=(
    --parsable
    --job-name="${name}"
    --output="${RUN_ROOT}/logs/%x-%j.out"
    --error="${RUN_ROOT}/logs/%x-%j.err"
  )
  if [[ -n "${dependency}" ]]; then
    options+=(--dependency="afterok:${dependency}")
  fi
  local submitted
  submitted=$(sbatch "${options[@]}" "${stage_script}" \
    "${SOURCE_DIR}" "${RUN_ROOT}" "${PYTHON_BIN}" "$@")
  echo "${submitted%%;*}"
}

submit_gpu() {
  local name=$1
  local dependency=$2
  shift 2
  submit_job "${GPU_STAGE_SCRIPT}" "${name}" "${dependency}" "$@"
}

submit_cpu() {
  local name=$1
  local dependency=$2
  shift 2
  submit_job "${CPU_STAGE_SCRIPT}" "${name}" "${dependency}" "$@"
}

join_dependencies() {
  local IFS=:
  echo "$*"
}

if [[ "${MODE}" == smoke ]]; then
  smoke_job=$(submit_gpu uq-smoke "" \
    "${SOURCE_DIR}/scripts/noether/clean_uq_smoke.py" \
    --output-dir "${RUN_ROOT}/synthetic-smoke" --seed 42)
  echo "synthetic strong-baseline smoke submitted: ${smoke_job}"
  exit 0
fi

: "${DATA_PT:?--data-pt is required}"
: "${PARTITION_JSONL:?--partition-jsonl is required}"
: "${PROTOCOL_JSON:?--protocol-json is required}"
: "${TARGET_LEDGER:?--target-ledger is required}"
: "${NUM_NODES:?--num-nodes is required}"
test -f "${DATA_PT}"
test -f "${PARTITION_JSONL}"
test -f "${PROTOCOL_JSON}"
test -f "${TARGET_LEDGER}"

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
if [[ "${SEEDS}" != "42,43,44,45,46" ]]; then
  echo "WWW RQ1 requires frozen seeds 42,43,44,45,46" >&2
  exit 64
fi

case "${MODE}" in
  pretest-representative|pretest)
    : "${FIT_SELECT_TARGETS_PT:?--fit-select-targets-pt is required}"
    if [[ -n "${SEALED_TEST_TARGETS_PT}" ]]; then
      echo "pretest modes reject --sealed-test-targets-pt" >&2
      exit 64
    fi
    test -f "${FIT_SELECT_TARGETS_PT}"
    PHASE=pretest
    ;;
  sealed-test-representative|sealed-test)
    : "${SEALED_TEST_TARGETS_PT:?--sealed-test-targets-pt is required}"
    test -f "${SEALED_TEST_TARGETS_PT}"
    PHASE=sealed-test
    ;;
esac
if [[ "${MODE}" == pretest ]]; then
  : "${CALIBRATE_TARGETS_PT:?--calibrate-targets-pt is required}"
  : "${REPRESENTATIVE_JOB_ID:?--representative-job-id is required}"
  test -f "${CALIBRATE_TARGETS_PT}"
fi
if [[ "${MODE}" == sealed-test ]]; then
  : "${REPRESENTATIVE_JOB_ID:?--representative-job-id is required}"
fi

require_canonical_target() {
  local supplied=$1
  local expected=$2
  local label=$3
  if [[ "$(realpath -e -- "${supplied}")" != "$(realpath -e -- "${expected}")" ]]; then
    echo "${label} must be the canonical role target under RUN_ROOT" >&2
    exit 64
  fi
}

case "${MODE}" in
  pretest-representative)
    require_canonical_target "${FIT_SELECT_TARGETS_PT}" \
      "${RUN_ROOT}/protocol/targets/fit_select.pt" fit-select
    ;;
  pretest)
    require_canonical_target "${FIT_SELECT_TARGETS_PT}" \
      "${RUN_ROOT}/protocol/targets/fit_select.pt" fit-select
    require_canonical_target "${CALIBRATE_TARGETS_PT}" \
      "${RUN_ROOT}/protocol/targets/calibrate.pt" calibrate
    ;;
  sealed-test-representative|sealed-test)
    require_canonical_target "${SEALED_TEST_TARGETS_PT}" \
      "${RUN_ROOT}/sealed/targets/test.pt" sealed-test
    ;;
esac

PYTHONPATH="${SOURCE_DIR}" "${PYTHON_BIN}" \
  "${SOURCE_DIR}/scripts/noether/clean_uq_phase_check.py" "${PHASE}" \
  --protocol-json "${PROTOCOL_JSON}" --run-root "${RUN_ROOT}" \
  --target-ledger "${TARGET_LEDGER}" \
  --seeds "${SEED_VALUES[@]}"

TRAIN_MODULE=credipred.experiments.gnn_experiments.clean_uq_train
BASELINE_MODULE=credipred.experiments.gnn_experiments.clean_uq_baselines
AUDIT_MODULE=credipred.experiments.gnn_experiments.clean_uq_audit
ARMS=(FF GCN SAGE GAT GAT-mlp GAT-topology)
BASELINE_ARMS=(GlobalMedian XGBoost RegressionLP)
FIRST_SEED=${SEED_VALUES[0]}

arm_neighbor_k() {
  case "$1" in
    FF|GAT-mlp) echo 0 ;;
    *) echo "${NEIGHBOR_K}" ;;
  esac
}

if [[ "${MODE}" == pretest-representative ]]; then
  representative_job=$(submit_gpu "uq-rep-GAT-${FIRST_SEED}" "" \
    -m "${TRAIN_MODULE}" base-fit \
    --run-dir "${RUN_ROOT}" --seed "${FIRST_SEED}" \
    --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}" --model GAT \
    --epochs "${BASE_EPOCHS}" --batch-size "${BATCH_SIZE}" \
    --neighbor-k "${NEIGHBOR_K}" --num-workers "${NUM_WORKERS}" \
    --hidden-channels 256 --embedding-dim 128 --num-layers 3 \
    --dropout 0.1 --normalization BatchNorm)
  echo "pretest representative submitted: ${representative_job}"
  exit 0
fi

if [[ "${MODE}" == pretest ]]; then
  declare -A BASE_JOB CACHE_JOB CORRECTION_JOB CAL_PRED_JOB
  declare -a CAL_AUDIT_JOBS
  BASE_JOB["${FIRST_SEED}:GAT"]=${REPRESENTATIVE_JOB_ID}

  for seed in "${SEED_VALUES[@]}"; do
    for base_arm in FF GCN SAGE GAT; do
      key="${seed}:${base_arm}"
      if [[ -n "${BASE_JOB[${key}]:-}" ]]; then
        continue
      fi
      arm_k=$(arm_neighbor_k "${base_arm}")
      BASE_JOB["${key}"]=$(submit_gpu "uq-${base_arm}-${seed}" "" \
        -m "${TRAIN_MODULE}" base-fit \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --model "${base_arm}" \
        --epochs "${BASE_EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}" \
        --hidden-channels 256 --embedding-dim 128 --num-layers 3 \
        --dropout 0.1 --normalization BatchNorm)
    done
  done

  cache_probe_job=$(submit_gpu "uq-cache-probe-${FIRST_SEED}" \
    "${BASE_JOB[${FIRST_SEED}:GAT]}" \
    "${SOURCE_DIR}/scripts/noether/clean_uq_cache_probe.py" \
    --run-dir "${RUN_ROOT}" --seed "${FIRST_SEED}" --data-pt "${DATA_PT}" \
    --sample-nodes "${PROBE_NODES}" --batch-size "${BATCH_SIZE}" \
    --neighbor-k "${NEIGHBOR_K}" --num-workers "${NUM_WORKERS}" \
    --max-projected-hours "${MAX_CACHE_HOURS}")

  lp_probe_job=$(submit_cpu uq-lp-probe "" \
    -m "${BASELINE_MODULE}" lp-probe --run-dir "${RUN_ROOT}" \
    --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}" --chunk-size "${LP_CHUNK_SIZE}" \
    --passes 2 --max-projected-hours "${MAX_LP_HOURS}")

  median_fit_job=$(submit_cpu uq-median-fit "" \
    -m "${BASELINE_MODULE}" median-fit --run-dir "${RUN_ROOT}" \
    --targets-pt "${FIT_SELECT_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}" --num-nodes "${NUM_NODES}")
  xgboost_fit_job=$(submit_cpu uq-xgboost-fit "" \
    -m "${BASELINE_MODULE}" xgboost-fit --run-dir "${RUN_ROOT}" \
    --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}")
  lp_fit_job=$(submit_cpu uq-lp-fit "${lp_probe_job}" \
    -m "${BASELINE_MODULE}" lp-fit --run-dir "${RUN_ROOT}" \
    --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}" --chunk-size "${LP_CHUNK_SIZE}")

  for seed in "${SEED_VALUES[@]}"; do
    cache_dependency=$(join_dependencies \
      "${BASE_JOB["${seed}:GAT"]}" "${cache_probe_job}")
    CACHE_JOB["${seed}"]=$(submit_gpu "uq-cache-${seed}" "${cache_dependency}" \
      -m "${TRAIN_MODULE}" parent-cache \
      --run-dir "${RUN_ROOT}" --seed "${seed}" --data-pt "${DATA_PT}" \
      --batch-size "${BATCH_SIZE}" --neighbor-k "${NEIGHBOR_K}" \
      --num-workers "${NUM_WORKERS}")
    for correction_arm in mlp topology; do
      arm_k=$(arm_neighbor_k "GAT-${correction_arm}")
      CORRECTION_JOB["${seed}:${correction_arm}"]=$(submit_gpu \
        "uq-${correction_arm}-${seed}" "${CACHE_JOB[${seed}]}" \
        -m "${TRAIN_MODULE}" correction-fit \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --targets-pt "${FIT_SELECT_TARGETS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --arm "${correction_arm}" \
        --epochs "${CORRECTION_EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}" \
        --hidden-channels 64 --num-layers 2 --dropout 0.1 \
        --normalization BatchNorm)
    done
  done

  for seed in "${SEED_VALUES[@]}"; do
    for arm in "${ARMS[@]}"; do
      case "${arm}" in
        FF|GCN|SAGE) dependency=${BASE_JOB["${seed}:${arm}"]} ;;
        GAT) dependency=${CACHE_JOB["${seed}"]} ;;
        GAT-mlp) dependency=${CORRECTION_JOB["${seed}:mlp"]} ;;
        GAT-topology) dependency=${CORRECTION_JOB["${seed}:topology"]} ;;
      esac
      arm_k=$(arm_neighbor_k "${arm}")
      prediction_job=$(submit_gpu "uq-cal-pred-${arm}-${seed}" "${dependency}" \
        -m "${TRAIN_MODULE}" predict-role \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --targets-pt "${CALIBRATE_TARGETS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --arm "${arm}" \
        --role calibrate --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}")
      CAL_PRED_JOB["${seed}:${arm}"]=${prediction_job}
      audit_job=$(submit_cpu "uq-cal-${arm}-${seed}" "${prediction_job}" \
        -m "${AUDIT_MODULE}" calibrate \
        --run-dir "${RUN_ROOT}" --seed "${seed}" --arm "${arm}" \
        --predictions-pt "${RUN_ROOT}/predictions/${arm}/seed-${seed}/calibrate.pt")
      CAL_AUDIT_JOBS+=("${audit_job}")
    done
  done

  declare -A BASELINE_FIT_JOB=(
    [GlobalMedian]="${median_fit_job}"
    [XGBoost]="${xgboost_fit_job}"
    [RegressionLP]="${lp_fit_job}"
  )
  for arm in "${BASELINE_ARMS[@]}"; do
    predict_arguments=(
      -m "${BASELINE_MODULE}" predict-role --run-dir "${RUN_ROOT}"
      --targets-pt "${CALIBRATE_TARGETS_PT}"
      --partition-jsonl "${PARTITION_JSONL}" --num-nodes "${NUM_NODES}"
      --arm "${arm}" --role calibrate
    )
    if [[ "${arm}" == XGBoost ]]; then
      predict_arguments+=(--data-pt "${DATA_PT}")
    fi
    prediction_job=$(submit_cpu "uq-cal-pred-${arm}" \
      "${BASELINE_FIT_JOB[${arm}]}" "${predict_arguments[@]}")
    audit_job=$(submit_cpu "uq-cal-${arm}" "${prediction_job}" \
      -m "${AUDIT_MODULE}" calibrate --run-dir "${RUN_ROOT}" \
      --seed deterministic --arm "${arm}" \
      --predictions-pt "${RUN_ROOT}/predictions/${arm}/deterministic/calibrate.pt")
    CAL_AUDIT_JOBS+=("${audit_job}")
  done

  ensemble_calibrate=(
    -m "${AUDIT_MODULE}" ensemble --run-dir "${RUN_ROOT}" --role calibrate
  )
  declare -a ENSEMBLE_DEPENDENCIES
  for seed in "${SEED_VALUES[@]}"; do
    ensemble_calibrate+=(
      --predictions-pt "${RUN_ROOT}/predictions/GAT/seed-${seed}/calibrate.pt"
    )
    ENSEMBLE_DEPENDENCIES+=("${CAL_PRED_JOB[${seed}:GAT]}")
  done
  ensemble_prediction_job=$(submit_cpu uq-ensemble-cal \
    "$(join_dependencies "${ENSEMBLE_DEPENDENCIES[@]}")" \
    "${ensemble_calibrate[@]}")
  ensemble_audit_job=$(submit_cpu uq-cal-ensemble "${ensemble_prediction_job}" \
    -m "${AUDIT_MODULE}" calibrate --run-dir "${RUN_ROOT}" \
    --seed ensemble --arm GAT-ensemble \
    --predictions-pt "${RUN_ROOT}/predictions/GAT-ensemble/calibrate.pt")
  CAL_AUDIT_JOBS+=("${ensemble_audit_job}")
  collect_job=$(submit_cpu uq-collect-pretest \
    "$(join_dependencies "${CAL_AUDIT_JOBS[@]}")" \
    "${SOURCE_DIR}/scripts/noether/clean_uq_collect.py" pretest \
    --run-root "${RUN_ROOT}")
  echo "pretest DAG submitted; LP probe ${lp_probe_job}; terminal collect ${collect_job}"
  exit 0
fi

if [[ "${MODE}" == sealed-test-representative ]]; then
  representative_job=$(submit_gpu "uq-rep-test-GAT-${FIRST_SEED}" "" \
    -m "${TRAIN_MODULE}" predict-role \
    --run-dir "${RUN_ROOT}" --seed "${FIRST_SEED}" \
    --data-pt "${DATA_PT}" --targets-pt "${SEALED_TEST_TARGETS_PT}" \
    --partition-jsonl "${PARTITION_JSONL}" --arm GAT --role test \
    --batch-size "${BATCH_SIZE}" --neighbor-k "${NEIGHBOR_K}" \
    --num-workers "${NUM_WORKERS}")
  echo "sealed-test representative submitted: ${representative_job}"
  exit 0
fi

declare -A TEST_PRED_JOB
declare -a TEST_AUDIT_JOBS
TEST_PRED_JOB["${FIRST_SEED}:GAT"]=${REPRESENTATIVE_JOB_ID}
for seed in "${SEED_VALUES[@]}"; do
  for arm in "${ARMS[@]}"; do
    key="${seed}:${arm}"
    if [[ -z "${TEST_PRED_JOB[${key}]:-}" ]]; then
      arm_k=$(arm_neighbor_k "${arm}")
      TEST_PRED_JOB["${key}"]=$(submit_gpu "uq-test-pred-${arm}-${seed}" "" \
        -m "${TRAIN_MODULE}" predict-role \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --targets-pt "${SEALED_TEST_TARGETS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --arm "${arm}" --role test \
        --batch-size "${BATCH_SIZE}" --neighbor-k "${arm_k}" \
        --num-workers "${NUM_WORKERS}")
    fi
    audit_job=$(submit_cpu "uq-test-${arm}-${seed}" "${TEST_PRED_JOB[${key}]}" \
      -m "${AUDIT_MODULE}" test --run-dir "${RUN_ROOT}" \
      --seed "${seed}" --arm "${arm}" \
      --predictions-pt "${RUN_ROOT}/predictions/${arm}/seed-${seed}/test.pt")
    TEST_AUDIT_JOBS+=("${audit_job}")
  done
done

for arm in "${BASELINE_ARMS[@]}"; do
  predict_arguments=(
    -m "${BASELINE_MODULE}" predict-role --run-dir "${RUN_ROOT}"
    --targets-pt "${SEALED_TEST_TARGETS_PT}"
    --partition-jsonl "${PARTITION_JSONL}" --num-nodes "${NUM_NODES}"
    --arm "${arm}" --role test
  )
  if [[ "${arm}" == XGBoost ]]; then
    predict_arguments+=(--data-pt "${DATA_PT}")
  fi
  prediction_job=$(submit_cpu "uq-test-pred-${arm}" "" "${predict_arguments[@]}")
  audit_job=$(submit_cpu "uq-test-${arm}" "${prediction_job}" \
    -m "${AUDIT_MODULE}" test --run-dir "${RUN_ROOT}" \
    --seed deterministic --arm "${arm}" \
    --predictions-pt "${RUN_ROOT}/predictions/${arm}/deterministic/test.pt")
  TEST_AUDIT_JOBS+=("${audit_job}")
done

ensemble_test=(
  -m "${AUDIT_MODULE}" ensemble --run-dir "${RUN_ROOT}" --role test
)
declare -a ENSEMBLE_TEST_DEPENDENCIES
for seed in "${SEED_VALUES[@]}"; do
  ensemble_test+=(
    --predictions-pt "${RUN_ROOT}/predictions/GAT/seed-${seed}/test.pt"
  )
  ENSEMBLE_TEST_DEPENDENCIES+=("${TEST_PRED_JOB[${seed}:GAT]}")
done
ensemble_prediction_job=$(submit_cpu uq-ensemble-test \
  "$(join_dependencies "${ENSEMBLE_TEST_DEPENDENCIES[@]}")" \
  "${ensemble_test[@]}")
ensemble_audit_job=$(submit_cpu uq-test-ensemble "${ensemble_prediction_job}" \
  -m "${AUDIT_MODULE}" test --run-dir "${RUN_ROOT}" \
  --seed ensemble --arm GAT-ensemble \
  --predictions-pt "${RUN_ROOT}/predictions/GAT-ensemble/test.pt")
TEST_AUDIT_JOBS+=("${ensemble_audit_job}")
collect_job=$(submit_cpu uq-collect-sealed \
  "$(join_dependencies "${TEST_AUDIT_JOBS[@]}")" \
  "${SOURCE_DIR}/scripts/noether/clean_uq_collect.py" sealed-test \
  --run-root "${RUN_ROOT}")
echo "sealed-test DAG submitted; terminal collect ${collect_job}"
