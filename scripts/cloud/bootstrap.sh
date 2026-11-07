#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
command -v uv >/dev/null || { echo 'Install uv before running bootstrap.' >&2; exit 1; }
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[train]'
TASK_ARCHIVE="${RUNTIME_ARCHIVE:-$TASK_ROOT/artifacts_local/so101-geometry-v1.tar.gz}"
mkdir -p "$(dirname -- "$TASK_ARCHIVE")"
if [[ ! -f "$TASK_ARCHIVE" ]]; then
    .venv/bin/python - "$TASK_ARCHIVE" <<'PY'
import json,sys,urllib.request
from pathlib import Path
c=json.loads(Path('configs/runtime_bundle.json').read_text())
url=f"https://github.com/{c['repository']}/releases/download/{c['release']}/{c['asset']}"
urllib.request.urlretrieve(url,sys.argv[1])
PY
fi
.venv/bin/python scripts/cloud/runtime_bundle.py install "$TASK_ARCHIVE"
PYTHON="$TASK_ROOT/.venv/bin/python" bash scripts/so101_build_plugins.sh
echo 'Native geometry and plugins are ready. Generate a qualified bank before training.'
