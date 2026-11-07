#!/usr/bin/env bash
# Install the standalone simulator runtime without changing the training venv.
set -euo pipefail
teleop_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$teleop_root"
case "$(uname -s):$(uname -m)" in
  Linux:x86_64) teleop_gl=egl;;
  Darwin:arm64) teleop_gl=cgl;;
  *) echo 'This native runtime supports Linux x86-64 and macOS ARM64.' >&2; exit 1;;
esac
command -v cc >/dev/null && command -v g++ >/dev/null || { echo 'Install a C/C++ compiler (cc and g++), then rerun setup.' >&2; exit 1; }
if command -v uv >/dev/null; then
  [[ -x .venv-teleop/bin/python ]] || uv venv --python 3.12 .venv-teleop
  uv pip install --python .venv-teleop/bin/python -r teleop/requirements.lock
else
  [[ -x .venv-teleop/bin/python ]] || python3.12 -m venv .venv-teleop
  .venv-teleop/bin/python -m pip install -r teleop/requirements.lock
fi
teleop_runtime=()
while IFS= read -r teleop_value; do teleop_runtime+=("$teleop_value"); done < <(.venv-teleop/bin/python -c 'import json; c=json.load(open("configs/runtime_bundle.json")); print(c["repository"]); print(c["release"]); print(c["asset"])')
[[ ${#teleop_runtime[@]} == 3 ]] || { echo 'Invalid runtime configuration.' >&2; exit 1; }
teleop_archive=${RUNTIME_ARCHIVE:-$teleop_root/artifacts_local/download/${teleop_runtime[2]}}
if [[ ! -f "$teleop_archive" ]]; then
  command -v gh >/dev/null || { echo 'Install/authenticate gh for this private repo, or set RUNTIME_ARCHIVE to its downloaded runtime asset.' >&2; exit 1; }
  mkdir -p "$(dirname -- "$teleop_archive")"
  gh release download "${teleop_runtime[1]}" --repo "${teleop_runtime[0]}" --pattern "${teleop_runtime[2]}" --dir "$(dirname -- "$teleop_archive")"
  [[ "$teleop_archive" == "$(dirname -- "$teleop_archive")/${teleop_runtime[2]}" ]] || mv -- "$(dirname -- "$teleop_archive")/${teleop_runtime[2]}" "$teleop_archive"
fi
.venv-teleop/bin/python scripts/cloud/runtime_bundle.py install "$teleop_archive"
PYTHON="$teleop_root/.venv-teleop/bin/python" bash scripts/so101_build_plugins.sh
MUJOCO_GL="${MUJOCO_GL:-$teleop_gl}" .venv-teleop/bin/python -m teleop.doctor
echo 'Ready: connect/authorize your Quest USB cable, then run bash teleop/run.sh --quest'
