#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
viewer_python="${VIEWER_PYTHON:-$repo_dir/.venv-viewer/bin/python}"
if [[ ! -x "$viewer_python" ]]; then
    printf 'Run bash scripts/setup_viewer.sh first.\n' >&2
    exit 1
fi
cd "$repo_dir"
exec "$viewer_python" -m assembly_policy.viewer \
    --snapshot "$repo_dir/runtime/wowrobo-viewer" --port 8876 "$@"
