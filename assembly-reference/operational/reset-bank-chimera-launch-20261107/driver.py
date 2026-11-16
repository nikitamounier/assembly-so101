"""Allocated CPU-only reset-bank relocation, qualification and exact-quota build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=25)
    a = p.parse_args()
    root = a.root.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Reset computation requires an allocated Slurm step")
    ready_deadline = time.monotonic() + 1200
    while not (root / "READY").exists():
        if time.monotonic() > ready_deadline:
            raise RuntimeError("Package READY marker did not arrive within20minutes")
        time.sleep(3)
    atomic(
        root / "allocated.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "host": os.uname().nodename,
            "cpus": len(os.sched_getaffinity(0)),
            "workers": a.workers,
            "pid": os.getpid(),
            "python": sys.executable,
            "allocated_at_unix": time.time(),
        },
    )
    # The relocator and its exact invocation are supplied with the frozen
    # source/asset package, before READY is written by the release operator.
    release = json.loads((root / "release.json").read_text())
    if not (root / "extract-complete.json").exists():
        archive = root / release.get("archive_filename", "reset-workspace.tar.gz")
        if (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            != release["archive_sha256"]
        ):
            raise RuntimeError("Transferred archive SHA256 differs from frozen release")
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                for value in [member.name] + (
                    [member.linkname] if member.issym() or member.islnk() else []
                ):
                    if Path(value).is_absolute() or ".." in Path(value).parts:
                        raise ValueError(
                            "Archive path is outside the isolated reset workspace"
                        )
            bundle.extractall(root, filter="data")
        atomic(
            root / "extract-complete.json",
            {"archive_sha256": release["archive_sha256"]},
        )
    if not (root / "relocation-complete.json").exists():
        subprocess.run(
            [sys.executable, *release["relocation_args"]], cwd=root, check=True
        )
    project = root / "robotics-assembly-3.0"
    sys.path.insert(0, str(project))
    os.environ["PYTHONPATH"] = str(project)
    os.chdir(project)
    import mujoco
    import numpy as np

    if mujoco.__version__ != "4.0.0":
        raise RuntimeError("Native source contacts require MuJoCo4.0.0")
    atomic(
        root / "environment.json",
        {
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "executable": sys.executable,
        },
    )
    preflight = release.get("preflight_args")
    if preflight:
        deadline = time.monotonic() + 1200
        retry = root / "launch/PREFLIGHT_RETRY"
        while True:
            result = subprocess.run([sys.executable, *preflight], cwd=root, check=False)
            if result.returncode == 0:
                atomic(
                    root / "run-status.json",
                    {
                        "status": "preflight_passed",
                        "job_id": os.environ["SLURM_JOB_ID"],
                        "unix": time.time(),
                    },
                )
                break
            atomic(
                root / "run-status.json",
                {
                    "status": "waiting_for_preflight_repair",
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "exit_code": result.returncode,
                    "unix": time.time(),
                },
            )
            prior = retry.stat().st_mtime_ns if retry.exists() else None
            while (retry.stat().st_mtime_ns if retry.exists() else None) == prior:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Preflight did not pass within its bounded repair window"
                    )
                time.sleep(3)
    farm_gate = root / "launch/farm_control.py"
    if farm_gate.exists():
        subprocess.run([sys.executable, str(farm_gate), "gate-coordinator", "--root", str(root), "--pid", str(os.getpid())], check=True)
    spec = importlib.util.spec_from_file_location(
        "reset_framework_cluster", project / "scripts/build_wowrobo_reset_framework.py"
    )
    framework = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(framework)
    plan = project / release["plan_relative"]
    for attempts in (20, 40, 80, 160):
        result = framework.run(plan, None, workers=a.workers, attempts=attempts)
        atomic(root / "progress.json", result)
        if result["complete"]:
            break
    if not result["complete"]:
        atomic(
            root / "result.json",
            {"status": "incomplete", "progress": result, "completed_unix": time.time()},
        )
        return 2
    finalize = root / "launch/finalize.py"
    if finalize.exists():
        subprocess.run(
            [sys.executable, str(finalize), "--root", str(root)], cwd=root, check=True
        )
    atomic(
        root / "result.json",
        {
            "status": "complete",
            "accepted": result["accepted"],
            "target": result["target"],
            "bank": str(plan / "bank"),
            "completed_unix": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--root") + 1])
            atomic(
                root / "result.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "failed_unix": time.time(),
                },
            )
        raise
