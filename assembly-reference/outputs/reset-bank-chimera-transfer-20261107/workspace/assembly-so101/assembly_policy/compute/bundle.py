"""Build a frozen, credential-free worker image input once per source snapshot."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_bundle(output: Path, root: Path = ROOT) -> dict:
    required = root / "assets_local/parts/manifest.json"
    if not required.is_file():
        raise FileNotFoundError("Install the native geometry runtime before building worker images")
    paths = []
    for directory, suffixes in (
        ("assembly_policy", {".py"}),
        ("plugin", {".c", ".cc", ".sh"}),
        ("configs", {".json"}),
        ("assets_local", {".json", ".obj", ".npz", ".bin", ".stl", ".xml", ".txt"}),
    ):
        for path in sorted((root / directory).rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix in suffixes
                and "build" not in path.parts
            ):
                paths.append(path)
    paths += [root / "pyproject.toml", root / "requirements-worker.txt"]
    build_script = root / "scripts/so101_build_plugins.sh"
    if build_script.exists():
        paths.append(build_script)
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w|") as archive,
    ):
        for path in paths:
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(root)))
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("worker-manifest.json")
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
    return {
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(paths),
        "bytes": output.stat().st_size,
    }


BUILD = (
    "mkdir -p /opt/assembly && tar -xzf /tmp/worker.tar.gz -C /opt/assembly && "
    "cd /opt/assembly && python -m pip install -r requirements-worker.txt && "
    "python -m pip install --no-deps -e . && "
    "PYTHON=python bash scripts/so101_build_plugins.sh"
)
