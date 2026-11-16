#!/usr/bin/env bash
#SBATCH --job-name=wowrobo-reset19200-cpu
#SBATCH --partition=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1/logs/pool-%j.log
set -euo pipefail
reset_root=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1
reset_python=/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python
reset_child=''
requeue_reset() { trap - USR1; if [[ -n "$reset_child" ]]; then kill -TERM "$reset_child" 2>/dev/null || true; wait "$reset_child" || true; fi; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue_reset USR1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for reset_tick in $(seq 1 180); do [[ -f "$reset_root/VALIDATED" ]] && break; [[ -f "$reset_root/STOP" ]] && exit 2; sleep 10; done
[[ -f "$reset_root/VALIDATED" ]]
reset_cpus="${SLURM_TRES_PER_TASK#cpu=}"
[[ "$reset_cpus" =~ ^[0-9]+$ ]]
srun --ntasks=1 --cpus-per-task="$reset_cpus" --cpu-bind=cores "$reset_python" "$reset_root/launch/pool.py" --root "$reset_root" --workers "$1" &
reset_child=$!
wait "$reset_child"
