#!/bin/bash
#SBATCH --job-name=simul_sa
#SBATCH --partition=standard
#SBATCH --account=irinagn0
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

if [ -z "${SIM_N:-}" ]; then
  echo "ERROR: SIM_N must be provided, e.g. sbatch --export=SIM_N=50 slurm/additional_simulations_sa_array.sh" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate JY
    PYTHON_RUN=(python)
  else
    PYTHON_RUN=(conda run -n JY python)
  fi
else
  echo "ERROR: conda command not found in PATH on this node." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export N_JOBS=${SLURM_CPUS_PER_TASK:-36}

mkdir -p logs

echo "[$(date)] Starting SA array task ${SLURM_ARRAY_TASK_ID} with SIM_N=${SIM_N} on node $(hostname)"
"${PYTHON_RUN[@]}" tools/run_additional_simulations.py --group sa --n "${SIM_N}"
echo "[$(date)] Finished SA array task ${SLURM_ARRAY_TASK_ID}"
