#!/usr/bin/env bash
#SBATCH --job-name=mu-star-split
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

set -euo pipefail

: "${MU_STAR_REGISTRY_JSONL:?set the reviewed canonical-domain registry}"
: "${MU_STAR_REGISTRY_SHA256:?set the out-of-band registry SHA-256}"
: "${MU_STAR_OUTPUT_DIR:?set a fresh output directory}"
: "${DATA_PT:?set the frozen processed graph data.pt}"
: "${DATA_PT_SHA256:?set the reviewed data.pt SHA-256}"
: "${MAPPING_PT:?set the matching graph mapping.pt}"
: "${MAPPING_PT_SHA256:?set the reviewed mapping.pt SHA-256}"
: "${LABELS_PT:?set the DQR labels tensor aligned to graph nodes}"
: "${LABELS_PT_SHA256:?set the reviewed labels.pt SHA-256}"

REPO=${REPO:-/mnt/data/slurm-storage/jiaxio/ws/CrediPred}
PYTHON=${PYTHON:-${REPO}/.venv/bin/python}

if [[ -e "${MU_STAR_OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output: ${MU_STAR_OUTPUT_DIR}" >&2
  exit 2
fi

ACTUAL_REGISTRY_SHA256=$(sha256sum "${MU_STAR_REGISTRY_JSONL}" | cut -d' ' -f1)
if [[ "${ACTUAL_REGISTRY_SHA256}" != "${MU_STAR_REGISTRY_SHA256}" ]]; then
  echo "registry SHA-256 does not match the reviewed anchor" >&2
  exit 2
fi

PYTHONPATH="${REPO}" "${PYTHON}" -m credipred.experiments.gnn_experiments.mu_star_base_fit \
  --registry-jsonl "${MU_STAR_REGISTRY_JSONL}" \
  --output-dir "${MU_STAR_OUTPUT_DIR}" \
  --data-pt "${DATA_PT}" \
  --mapping-pt "${MAPPING_PT}" \
  --expected-data-sha256 "${DATA_PT_SHA256}" \
  --expected-mapping-sha256 "${MAPPING_PT_SHA256}" \
  --labels-pt "${LABELS_PT}" \
  --expected-labels-sha256 "${LABELS_PT_SHA256}" \
  --seed 42

echo "partition plan approved against reviewed data.pt and mapping.pt hashes; training remains a separate job"
