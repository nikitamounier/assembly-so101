#!/usr/bin/env bash
#SBATCH --job-name=wowrobo-reset1800
#SBATCH --partition=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --exclude=CPUDFDC84,CPUDFDE34,CPUDFE024,CPUDFE044,CPUDFE410,GPU105E,GPU122C,GPU1298,GPU12FE,GPU70DC,GPU71E4,GPU71FM,GPU724A,GPU726E,GPUCACE
#SBATCH --output=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4/logs/slurm-%j.log
set -euo pipefail
reset_task_root=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4
reset_python=/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python
reset_step_pid=''
requeue_reset() {
  trap - USR1
  # Every candidate is already checkpointed with atomic NPZ + receipt rename.
  # An interrupted in-flight candidate has no receipt and is resampled on resume.
  if [[ -n "$reset_step_pid" ]]; then
    kill -TERM "$reset_step_pid" 2>/dev/null || true
    wait "$reset_step_pid" || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_reset USR1
export PYTHONPATH="$reset_task_root/robotics-assembly-3.0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$reset_python" "$reset_task_root/launch/driver.py" --root "$reset_task_root" --workers 8 &
reset_step_pid=$!
wait "$reset_step_pid"
