#!/usr/bin/env bash
# Run inside the allocated Linux x86-64 GPU node, after bootstrap.sh.
set -euo pipefail
CLOUD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CLOUD_ROOT"
CLOUD_PYTHON="${CLOUD_PYTHON:-$CLOUD_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
CLOUD_CPUS="$("$CLOUD_PYTHON" -c 'import os; n=len(os.sched_getaffinity(0)); print(min(n,int(os.environ.get("SLURM_CPUS_PER_TASK",n))))')"
CLOUD_THREADS="${H100_THREADS:-$((CLOUD_CPUS>2 ? CLOUD_CPUS-2 : CLOUD_CPUS))}"
[[ "$CLOUD_THREADS" -gt 0 && "$CLOUD_THREADS" -le "$CLOUD_CPUS" ]] || { echo 'H100_THREADS must fit the allocated CPU budget.' >&2; exit 1; }
CLOUD_ENVS="${H100_ENVS:-$("$CLOUD_PYTHON" -c 'import sys; n=max(512,16*int(sys.argv[1])); print(min(4096,1<<(n-1).bit_length()))' "$CLOUD_THREADS")}"
CLOUD_HOURS="${H100_HOURS:-4}"
CLOUD_ITERS="${H100_ITERS:-10000}"
CLOUD_TRIALS="${H100_TRIALS:-16}"
CLOUD_RUN="${H100_RUN_ROOT:-runs/h100-$(date -u +%Y%m%dT%H%M%SZ)}"
printf 'Allocated CPUs: %s | simulator threads: %s | worlds: %s | training hours: %s | output: %s\n' "$CLOUD_CPUS" "$CLOUD_THREADS" "$CLOUD_ENVS" "$CLOUD_HOURS" "$CLOUD_RUN"
if [[ "${1:-}" == --dry-run ]]; then
  exec "$CLOUD_PYTHON" scripts/cloud/run.py --run-dir "$CLOUD_RUN/train" --envs "$CLOUD_ENVS" --threads "$CLOUD_THREADS" --iterations "$CLOUD_ITERS" --max-hours "$CLOUD_HOURS" --dry-run
fi
"$CLOUD_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "An allocated CUDA GPU is required"; print("GPU:",torch.cuda.get_device_name(0))'
"$CLOUD_PYTHON" scripts/cloud/healthcheck.py --output "$CLOUD_RUN/smoke" --threads "$((CLOUD_THREADS<4 ? CLOUD_THREADS : 4))"
"$CLOUD_PYTHON" scripts/cloud/run.py --run-dir "$CLOUD_RUN/train" --envs "$CLOUD_ENVS" --threads "$CLOUD_THREADS" --iterations "$CLOUD_ITERS" --max-hours "$CLOUD_HOURS"
"$CLOUD_PYTHON" scripts/cloud/assess_learning.py --run-dir "$CLOUD_RUN/train" --trials "$CLOUD_TRIALS" --workers "$((CLOUD_THREADS<8 ? CLOUD_THREADS : 8))"
printf 'Inspect health: %s/smoke/report.json\nTask improvement: %s/train/learning-assessment/comparison.json\n' "$CLOUD_RUN" "$CLOUD_RUN"
