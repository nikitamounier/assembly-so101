#!/usr/bin/env bash
#SBATCH --job-name=wowrobo-reset-packed
#SBATCH --partition=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4/logs/packed-%j.log
set -euo pipefail
reset_root=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4
reset_python=/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python
reset_child=''
checkpoint_requeue() {
 trap - USR1
 if [[ -n "$reset_child" ]]; then kill -TERM "$reset_child" 2>/dev/null || true; wait "$reset_child" || true; fi
 scontrol requeue "$SLURM_JOB_ID"
}
trap checkpoint_requeue USR1
export PYTHONPATH="$reset_root/robotics-assembly-3.0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
reset_task_cpus="${SLURM_TRES_PER_TASK#cpu=}"
[[ "$reset_task_cpus" =~ ^[0-9]+$ ]]
srun --ntasks=1 --cpus-per-task="$reset_task_cpus" --cpu-bind=cores "$reset_python" "$reset_root/launch/packed_worker.py" --root "$reset_root" --indices "$1" &
reset_child=$!
wait "$reset_child"
