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
ROOT=/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a
RUN="$ROOT/runs/chimera-v13-8192-20261106"
cd "$ROOT"
for binding in ${!SLURM_CPU_BIND@} ${!SRUN_@}; do unset "$binding"; done
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
unset SLURM_STEP_ID SLURM_STEPID SLURM_STEP_LAUNCHER_PORT SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_STEP_GPUS SLURM_GTIDS
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
renew() {
    trap - USR1
    if [[ -n "${WORKER_STEP_PID:-}" ]]; then kill -TERM "$WORKER_STEP_PID" 2>/dev/null || true; wait "$WORKER_STEP_PID" || true; fi
    scontrol requeue "$SLURM_JOB_ID"
}
trap renew USR1
printf '[assembly-node-attempt] job=%s node=%s restart=%s workers=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "${SLURM_RESTART_COUNT:-0}" "$SLURM_NTASKS"
srun --cpu-bind=cores --ntasks="$SLURM_NTASKS" --cpus-per-task=2 --kill-on-bad-exit=1 "$ROOT/.venv/bin/python" -m assembly_policy.rl.remote --run-dir "$RUN" --nthread 2 --wait 7200 --poll 1 &
WORKER_STEP_PID=$!
wait "$WORKER_STEP_PID"
