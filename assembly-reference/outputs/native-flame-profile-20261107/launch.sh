#!/usr/bin/env bash
set -euo pipefail
# Exact profiling inputs. Review the spare lease before rerunning.
profile_root=/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4
profile_python=/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python
srun --jobid=3606264 --overlap --exact -n1 -c8 --mem=24G --gpus=1 --cpu-bind=cores \
  env PROFILE_ROOT="$profile_root" PROFILE_OUT="$profile_root/native-flame-profile-20261107" \
  PYTHONPATH="$profile_root/assembly-so101" WOWROBO_CANONICAL_ROOT="$profile_root/robotics-assembly-3.0" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1 \
  "$profile_python" "$profile_root/native-flame-profile-20261107/run.py"
# Additional phase-clock pass, run after the PPO capture.
srun --jobid=3606264 --overlap --exact -n1 -c2 --mem=12G --gres=none --cpu-bind=cores \
  env PROFILE_ROOT="$profile_root" PROFILE_OUT="$profile_root/native-flame-profile-20261107" \
  PYTHONPATH="$profile_root/assembly-so101" WOWROBO_CANONICAL_ROOT="$profile_root/robotics-assembly-3.0" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1 \
  "$profile_python" "$profile_root/native-flame-profile-20261107/run_timers.py"
