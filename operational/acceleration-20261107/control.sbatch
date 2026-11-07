#!/bin/bash
#SBATCH --partition=cpu_preemptible
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=assembly-acceleration-control
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
ROOT=/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107
trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
cd "$ROOT"
if [[ ! -f .runtime-ready ]]; then
    BASE=/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv
    "$BASE/bin/python" -m venv .venv
    printf '%s\n' "$BASE/lib/python3.12/site-packages" > .venv/lib/python3.12/site-packages/assembly-base.pth
    .venv/bin/python -m pip install --no-deps mujoco==4.0.0 mujoco-core==4.0.0 mujoco-studio==4.0.0
    bash scripts/so101_build_plugins.sh
    .venv/bin/python -c 'import mujoco,torch; assert mujoco.__version__=="4.0.0"; print(torch.__version__,mujoco.__version__)'
    touch .runtime-ready
fi
.venv/bin/python operational/acceleration-20261107/benchmark.py &
wait $!
