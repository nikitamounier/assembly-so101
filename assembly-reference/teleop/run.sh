#!/usr/bin/env bash
set -euo pipefail
teleop_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$teleop_root"
teleop_python=${TELEOP_PYTHON:-$teleop_root/.venv-teleop/bin/python}
[[ -x "$teleop_python" ]] || { echo 'Run bash teleop/setup.sh first (or set TELEOP_PYTHON).' >&2; exit 1; }
if [[ "$(uname -s)" == Darwin ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-cgl}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
exec "$teleop_python" -m teleop.serve "$@"
