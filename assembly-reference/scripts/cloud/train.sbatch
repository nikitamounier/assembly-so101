#!/usr/bin/env bash
#SBATCH --job-name=so101-learner-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_RUN="${RUN_DIR:-runs/oracle-${SLURM_JOB_ID}}"
srun .venv/bin/python scripts/cloud/run.py --run-dir "$TASK_RUN" --threads "${SLURM_CPUS_PER_TASK:-32}"
