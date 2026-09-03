#!/usr/bin/env bash
#SBATCH --job-name=mu-star-fit
#SBATCH --output=/mnt/data/slurm-storage/jiaxio/ws/artifacts/credipred/mu-star/logs/fit-%j.out
#SBATCH --error=/mnt/data/slurm-storage/jiaxio/ws/artifacts/credipred/mu-star/logs/fit-%j.err
#SBATCH --partition=h100
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=${REPO:-/mnt/data/slurm-storage/jiaxio/ws/CrediPred}
PYTHON=${PYTHON:-${REPO}/.venv/bin/python}
DATA_PT=${DATA_PT:?set the frozen processed graph data.pt}
MAPPING_PT=${MAPPING_PT:?set the matching graph mapping.pt}
LABELS_PT=${LABELS_PT:?set the DQR labels tensor aligned to graph nodes}
MU_STAR_PARTITIONS=${MU_STAR_PARTITIONS:?set the reviewed mu_star_partitions.jsonl}
MU_STAR_PLAN=${MU_STAR_PLAN:?set the matching preflight mu_star_lineage.json}
MU_STAR_OUTPUT_DIR=${MU_STAR_OUTPUT_DIR:?set a new output directory}
TRAINER_SCRIPT=${TRAINER_SCRIPT:?set the isolated train_mu_star.py path}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-1024}
NEIGHBOR_K=${NEIGHBOR_K:-30}
NUM_WORKERS=${NUM_WORKERS:-4}

test -x "${PYTHON}"
test -f "${DATA_PT}"
test -f "${MAPPING_PT}"
test -f "${LABELS_PT}"
test -f "${MU_STAR_PARTITIONS}"
test -f "${MU_STAR_PLAN}"
test ! -e "${MU_STAR_OUTPUT_DIR}" || { echo "refusing to reuse output" >&2; exit 2; }

PYTHONPATH="${REPO}" \
OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
"${PYTHON}" "${TRAINER_SCRIPT}" \
  --data-pt "${DATA_PT}" \
  --mapping-pt "${MAPPING_PT}" \
  --labels-pt "${LABELS_PT}" \
  --partition-jsonl "${MU_STAR_PARTITIONS}" \
  --plan-json "${MU_STAR_PLAN}" \
  --output-dir "${MU_STAR_OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --neighbor-k "${NEIGHBOR_K}" \
  --num-workers "${NUM_WORKERS}" \
  --seed 42

test -f "${MU_STAR_OUTPUT_DIR}/mu_star_checkpoint.pt"
test -f "${MU_STAR_OUTPUT_DIR}/mu_star_midpoint.pt"
test -f "${MU_STAR_OUTPUT_DIR}/mu_star_lineage.json"
