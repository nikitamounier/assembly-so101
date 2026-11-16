#!/usr/bin/env bash
#SBATCH --partition=cpu_preemptible
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
set -uo pipefail
ROOT=/large_storage/goodarzilab/nikitamounier/assembly/assembly-nist-eb5d15e
PYTHON=/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv/bin/python
cd "$ROOT"
for binding in ${!SLURM_CPU_BIND@} ${!SRUN_@}; do unset "$binding"; done
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
unset SLURM_STEP_ID SLURM_STEPID SLURM_STEP_LAUNCHER_PORT SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_STEP_GPUS SLURM_GTIDS
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
renew() {
    trap - USR1
    if [[ -n "${CHILD:-}" ]]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; fi
    scontrol requeue "$SLURM_JOB_ID"
}
trap renew USR1
if (( $1 < 20 )); then
    "$PYTHON" operational/baseline_banks.py "$1" &
    CHILD=$!; wait "$CHILD" || exit $?
fi
while [[ ! -f artifacts_local/chimera/ALL_BANKS_READY ]]; do sleep 5 & CHILD=$!; wait "$CHILD"; done
srun --cpu-bind=cores --ntasks=5 --cpus-per-task=2 --kill-on-bad-exit=1 "$PYTHON" operational/worker_entry.py --run-dir "$ROOT/runs/profile" --nthread 2 --wait 7200 --poll 1 &
CHILD=$!; wait "$CHILD"
