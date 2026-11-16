#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if command -v uv >/dev/null 2>&1; then
    if [[ ! -x .venv-viewer/bin/python ]]; then
        uv venv --python 3.12 .venv-viewer
    fi
    uv pip install --python .venv-viewer/bin/python -e '.[viewer]' -c requirements-viewer.txt
else
    if [[ ! -x .venv-viewer/bin/python ]]; then
        python3.12 -m venv .venv-viewer
    fi
    .venv-viewer/bin/python -m pip install -e '.[viewer]' -c requirements-viewer.txt
fi
printf '\nViewer ready. Run: bash scripts/run_viewer.sh\n'
