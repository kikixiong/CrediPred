#!/usr/bin/env bash
# Submit the synthetic smoke or one sealed clean-UQ phase.
#
# Production usage:
#   clean_uq_submit.sh pretest ... --seeds 42,43,44,45,46
#   clean_uq_submit.sh sealed-test ... --seeds 42,43,44,45,46
# The retired `formal` mode is intentionally rejected.

set -euo pipefail

MODE=${1:-}
if [[ "${MODE}" != smoke && "${MODE}" != pretest && "${MODE}" != sealed-test ]]; then
  echo "first argument must be smoke, pretest, or sealed-test; formal is retired" >&2
  exit 64
fi
shift

SOURCE_DIR=
RUN_ROOT=
PYTHON_BIN=
DATA_PT=
LABELS_PT=
PARTITION_JSONL=
PROTOCOL_JSON=
SEEDS=42,43,44,45,46
BASE_EPOCHS=100
CORRECTION_EPOCHS=100
BATCH_SIZE=1024
NEIGHBOR_K=30
NUM_WORKERS=4
PROBE_NODES=32768
MAX_CACHE_HOURS=20.0

while (( $# )); do
  case "$1" in
    --source-dir) SOURCE_DIR=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --python-bin) PYTHON_BIN=$2; shift 2 ;;
    --data-pt) DATA_PT=$2; shift 2 ;;
    --labels-pt) LABELS_PT=$2; shift 2 ;;
    --partition-jsonl) PARTITION_JSONL=$2; shift 2 ;;
    --protocol-json) PROTOCOL_JSON=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --base-epochs) BASE_EPOCHS=$2; shift 2 ;;
    --correction-epochs) CORRECTION_EPOCHS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --neighbor-k) NEIGHBOR_K=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --probe-nodes) PROBE_NODES=$2; shift 2 ;;
    --max-cache-hours) MAX_CACHE_HOURS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

: "${SOURCE_DIR:?--source-dir is required}"
: "${RUN_ROOT:?--run-root is required}"
: "${PYTHON_BIN:?--python-bin is required}"
test -d "${SOURCE_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${RUN_ROOT}/logs"

STAGE_SCRIPT="${SOURCE_DIR}/scripts/noether/clean_uq_stage.sbatch"
test -f "${STAGE_SCRIPT}"

submit_job() {
  local name=$1
  local dependency=$2
  shift 2
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
  submitted=$(sbatch "${options[@]}" "${STAGE_SCRIPT}" \
    "${SOURCE_DIR}" "${RUN_ROOT}" "${PYTHON_BIN}" "$@")
  echo "${submitted%%;*}"
}

if [[ "${MODE}" == smoke ]]; then
  smoke_job=$(submit_job uq-smoke "" \
    "${SOURCE_DIR}/scripts/noether/clean_uq_smoke.py" \
    --output-dir "${RUN_ROOT}/synthetic-smoke" --seed 42)
  echo "synthetic smoke submitted: ${smoke_job}"
  exit 0
fi

: "${DATA_PT:?--data-pt is required}"
: "${LABELS_PT:?--labels-pt is required}"
: "${PARTITION_JSONL:?--partition-jsonl is required}"
: "${PROTOCOL_JSON:?--protocol-json is required}"
test -f "${DATA_PT}"
test -f "${LABELS_PT}"
test -f "${PARTITION_JSONL}"
test -f "${PROTOCOL_JSON}"

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
if (( ${#SEED_VALUES[@]} < 3 )); then
  echo "pre-registered ensemble requires at least three seeds" >&2
  exit 64
fi

PYTHONPATH="${SOURCE_DIR}" "${PYTHON_BIN}" \
  "${SOURCE_DIR}/scripts/noether/clean_uq_phase_check.py" "${MODE}" \
  --protocol-json "${PROTOCOL_JSON}" --run-root "${RUN_ROOT}" \
  --seeds "${SEED_VALUES[@]}"

TRAIN_MODULE=credipred.experiments.gnn_experiments.clean_uq_train
AUDIT_MODULE=credipred.experiments.gnn_experiments.clean_uq_audit
ARMS=(FF GCN SAGE GAT GAT-mlp GAT-topology)
previous_job=

arm_neighbor_k() {
  case "$1" in
    FF|GAT-mlp) echo 0 ;;
    *) echo "${NEIGHBOR_K}" ;;
  esac
}

if [[ "${MODE}" == pretest ]]; then
  for seed in "${SEED_VALUES[@]}"; do
    for base_arm in FF GCN SAGE GAT; do
      arm_k=$(arm_neighbor_k "${base_arm}")
      previous_job=$(submit_job "uq-${base_arm}-${seed}" "${previous_job}" \
        -m "${TRAIN_MODULE}" base-fit \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --labels-pt "${LABELS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --model "${base_arm}" \
        --epochs "${BASE_EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}" \
        --hidden-channels 256 --embedding-dim 128 --num-layers 3 \
        --dropout 0.1 --normalization BatchNorm)
    done

    if [[ "${seed}" == "${SEED_VALUES[0]}" ]]; then
      previous_job=$(submit_job "uq-probe-${seed}" "${previous_job}" \
        "${SOURCE_DIR}/scripts/noether/clean_uq_cache_probe.py" \
        --run-dir "${RUN_ROOT}" --seed "${seed}" --data-pt "${DATA_PT}" \
        --sample-nodes "${PROBE_NODES}" --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${NEIGHBOR_K}" --num-workers "${NUM_WORKERS}" \
        --max-projected-hours "${MAX_CACHE_HOURS}")
    fi
    previous_job=$(submit_job "uq-cache-${seed}" "${previous_job}" \
      -m "${TRAIN_MODULE}" parent-cache \
      --run-dir "${RUN_ROOT}" --seed "${seed}" --data-pt "${DATA_PT}" \
      --batch-size "${BATCH_SIZE}" --neighbor-k "${NEIGHBOR_K}" \
      --num-workers "${NUM_WORKERS}")

    for correction_arm in mlp topology; do
      arm_k=$(arm_neighbor_k "GAT-${correction_arm}")
      previous_job=$(submit_job "uq-${correction_arm}-${seed}" "${previous_job}" \
        -m "${TRAIN_MODULE}" correction-fit \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --labels-pt "${LABELS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --arm "${correction_arm}" \
        --epochs "${CORRECTION_EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}" \
        --hidden-channels 64 --num-layers 2 --dropout 0.1 \
        --normalization BatchNorm)
    done

    for arm in "${ARMS[@]}"; do
      arm_k=$(arm_neighbor_k "${arm}")
      previous_job=$(submit_job "uq-cal-pred-${arm}-${seed}" "${previous_job}" \
        -m "${TRAIN_MODULE}" predict-role \
        --run-dir "${RUN_ROOT}" --seed "${seed}" \
        --data-pt "${DATA_PT}" --labels-pt "${LABELS_PT}" \
        --partition-jsonl "${PARTITION_JSONL}" --arm "${arm}" \
        --role calibrate --batch-size "${BATCH_SIZE}" \
        --neighbor-k "${arm_k}" --num-workers "${NUM_WORKERS}")
      previous_job=$(submit_job "uq-cal-${arm}-${seed}" "${previous_job}" \
        -m "${AUDIT_MODULE}" calibrate \
        --run-dir "${RUN_ROOT}" --seed "${seed}" --arm "${arm}" \
        --predictions-pt "${RUN_ROOT}/predictions/${arm}/seed-${seed}/calibrate.pt")
    done
  done

  ensemble_calibrate=(
    -m "${AUDIT_MODULE}" ensemble --run-dir "${RUN_ROOT}" --role calibrate
  )
  for seed in "${SEED_VALUES[@]}"; do
    ensemble_calibrate+=(
      --predictions-pt "${RUN_ROOT}/predictions/GAT/seed-${seed}/calibrate.pt"
    )
  done
  previous_job=$(submit_job "uq-ensemble-cal" "${previous_job}" \
    "${ensemble_calibrate[@]}")
  previous_job=$(submit_job "uq-cal-ensemble" "${previous_job}" \
    -m "${AUDIT_MODULE}" calibrate \
    --run-dir "${RUN_ROOT}" --seed ensemble --arm GAT-ensemble \
    --predictions-pt "${RUN_ROOT}/predictions/GAT-ensemble/calibrate.pt")
  echo "pretest chain submitted for seeds [${SEEDS}]; terminal job: ${previous_job}"
  exit 0
fi

for seed in "${SEED_VALUES[@]}"; do
  for arm in "${ARMS[@]}"; do
    arm_k=$(arm_neighbor_k "${arm}")
    previous_job=$(submit_job "uq-test-pred-${arm}-${seed}" "${previous_job}" \
      -m "${TRAIN_MODULE}" predict-role \
      --run-dir "${RUN_ROOT}" --seed "${seed}" \
      --data-pt "${DATA_PT}" --labels-pt "${LABELS_PT}" \
      --partition-jsonl "${PARTITION_JSONL}" --arm "${arm}" --role test \
      --batch-size "${BATCH_SIZE}" --neighbor-k "${arm_k}" \
      --num-workers "${NUM_WORKERS}")
    previous_job=$(submit_job "uq-test-${arm}-${seed}" "${previous_job}" \
      -m "${AUDIT_MODULE}" test \
      --run-dir "${RUN_ROOT}" --seed "${seed}" --arm "${arm}" \
      --predictions-pt "${RUN_ROOT}/predictions/${arm}/seed-${seed}/test.pt")
  done
done

ensemble_test=(
  -m "${AUDIT_MODULE}" ensemble --run-dir "${RUN_ROOT}" --role test
)
for seed in "${SEED_VALUES[@]}"; do
  ensemble_test+=(
    --predictions-pt "${RUN_ROOT}/predictions/GAT/seed-${seed}/test.pt"
  )
done
previous_job=$(submit_job "uq-ensemble-test" "${previous_job}" \
  "${ensemble_test[@]}")
previous_job=$(submit_job "uq-test-ensemble" "${previous_job}" \
  -m "${AUDIT_MODULE}" test \
  --run-dir "${RUN_ROOT}" --seed ensemble --arm GAT-ensemble \
  --predictions-pt "${RUN_ROOT}/predictions/GAT-ensemble/test.pt")

echo "sealed-test chain submitted for frozen seeds [${SEEDS}]; terminal job: ${previous_job}"
