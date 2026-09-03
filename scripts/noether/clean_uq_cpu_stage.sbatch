#!/usr/bin/env bash
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if (( $# < 4 )); then
  echo "usage: clean_uq_cpu_stage.sbatch SOURCE_DIR RUN_ROOT PYTHON_BIN PYTHON_ARGS..." >&2
  exit 64
fi

SOURCE_DIR=$1
RUN_ROOT=$2
PYTHON_BIN=$3
shift 3

test -d "${SOURCE_DIR}"
test -d "${RUN_ROOT}"
test -x "${PYTHON_BIN}"

cd "${SOURCE_DIR}"
export PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

exec "${PYTHON_BIN}" "$@"
