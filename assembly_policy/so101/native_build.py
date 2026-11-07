"""Build native plugins together when their sources, checkout or runtime change."""

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "plugin/assemblenist_sdf.cc",
    "plugin/so101_grid_bounds.h",
    "plugin/so101_contact_search.c",
    "plugin/so101_surface_tree.h",
    "plugin/so101_screw_head.cc",
    "plugin/build.sh",
    "plugin/build_contact_search.sh",
    "scripts/so101_build_plugins.sh",
)
LIBRARIES = ("libassemblenist_sdf.so", "libso101_contact_search.so", "libso101_screw_head.so")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_plugins():
    """Serialize builders and bind all three outputs to the complete input set."""
    build = ROOT / "plugin/build"
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = dict(
            checkout=str(ROOT),
            platform=platform.system(),
            machine=platform.machine(),
            mujoco_version=mujoco.__version__,
            mujoco_directory=str(Path(mujoco.__file__).resolve().parent),
            sources={name: digest(ROOT / name) for name in SOURCES},
        )
        manifest = build / "build-manifest.json"
        try:
            saved = json.loads(manifest.read_text())
            current = {name: digest(build / name) for name in LIBRARIES}
            if saved == dict(inputs=inputs, libraries=current):
                return
        except (OSError, ValueError):
            pass
        subprocess.run(
            ["bash", str(ROOT / "scripts/so101_build_plugins.sh")],
            env=dict(os.environ, PYTHON=sys.executable),
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = {name: digest(build / name) for name in LIBRARIES}
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(inputs=inputs, libraries=outputs), indent=2) + "\n")
        temporary.replace(manifest)
