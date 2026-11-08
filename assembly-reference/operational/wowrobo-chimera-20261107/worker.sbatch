#!/usr/bin/env bash
set -euo pipefail
cd "${SO101_ROOT:?}"
for binding in ${!SLURM_CPU_BIND@} ${!SRUN_@}; do unset "$binding"; done
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
unset SLURM_STEP_ID SLURM_STEPID SLURM_STEP_LAUNCHER_PORT SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_STEP_GPUS SLURM_GTIDS
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1
renew() {
 trap - USR1
 if [[ -n "${worker_pid:-}" ]]; then kill -TERM "$worker_pid" 2>/dev/null || true; wait "$worker_pid" || true; fi
 scontrol requeue "$SLURM_JOB_ID"
}
trap renew USR1
srun --ntasks=1 --cpus-per-task=10 --cpu-bind=cores --kill-on-bad-exit=1 "${SO101_PYTHON:?}" -m assembly_policy.rl.remote --run-dir "${SO101_RUN:?}" --nthread 10 --wait 86400 --poll 1 &
worker_pid=$!
wait "$worker_pid"
