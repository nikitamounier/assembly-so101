#!/usr/bin/env bash
#SBATCH --job-name=wowrobo-reset19200-control
#SBATCH --partition=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1/logs/control-%j.log
set -euo pipefail
reset_root=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1
reset_python=/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python
reset_child=''
requeue_reset() { trap - USR1; if [[ -n "$reset_child" ]]; then kill -TERM "$reset_child" 2>/dev/null || true; wait "$reset_child" || true; fi; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue_reset USR1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores "$reset_python" "$reset_root/launch/setup.py" --root "$reset_root"
for reset_tick in $(seq 1 120); do [[ -f "$reset_root/READY" ]] && break; sleep 10; done
[[ -f "$reset_root/READY" ]]
srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores "$reset_python" "$reset_root/launch/coordinator.py" --root "$reset_root" &
reset_child=$!
wait "$reset_child"
