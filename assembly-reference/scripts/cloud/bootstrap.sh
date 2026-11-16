#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || { echo 'This pinned build currently supports Linux x86-64.' >&2; exit 1; }
command -v g++ >/dev/null || { echo 'Install g++ before building the MuJoCo plugins.' >&2; exit 1; }
if command -v uv >/dev/null; then
  [[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements-cloud.lock -e .
else
  [[ -x .venv/bin/python ]] || python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements-cloud.lock -e .
fi
mapfile -t TASK_RUNTIME < <(.venv/bin/python -c 'import json; c=json.load(open("configs/runtime_bundle.json")); print(c["repository"]); print(c["release"]); print(c["asset"])')
[[ "${#TASK_RUNTIME[@]}" == 3 ]] || { echo 'Invalid runtime configuration.' >&2; exit 1; }
TASK_ARCHIVE="${RUNTIME_ARCHIVE:-$TASK_ROOT/artifacts_local/download/${TASK_RUNTIME[2]}}"
if [[ ! -f "$TASK_ARCHIVE" ]]; then
  command -v gh >/dev/null || { echo 'Install/authenticate gh for this private repo, or set RUNTIME_ARCHIVE to the downloaded release asset.' >&2; exit 1; }
  mkdir -p "$(dirname -- "$TASK_ARCHIVE")"
  gh release download "${TASK_RUNTIME[1]}" --repo "${TASK_RUNTIME[0]}" --pattern "${TASK_RUNTIME[2]}" --dir "$(dirname -- "$TASK_ARCHIVE")"
  [[ "$TASK_ARCHIVE" == "$(dirname -- "$TASK_ARCHIVE")/${TASK_RUNTIME[2]}" ]] || mv -- "$(dirname -- "$TASK_ARCHIVE")/${TASK_RUNTIME[2]}" "$TASK_ARCHIVE"
fi
.venv/bin/python scripts/cloud/runtime_bundle.py install "$TASK_ARCHIVE"
PYTHON="$TASK_ROOT/.venv/bin/python" bash scripts/so101_build_plugins.sh
.venv/bin/python -m scripts.prepare_so101_workcell --workers 2
.venv/bin/python -c 'import torch; print("CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available(), "Compatible NVIDIA driver/CUDA PyTorch required"'
echo 'Setup complete. Run the smoke check, then submit the stress session; see docs/ORACLE_HANDOFF.md.'
