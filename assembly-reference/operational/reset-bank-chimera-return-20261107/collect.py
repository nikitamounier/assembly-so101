"""One-shot completion transfer for one authorized reset-bank job; no alerts/jobs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REMOTE = "/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4"


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fetch(relative, destination):
    return (
        subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                f"chimera-login:{REMOTE}/{relative}",
                str(destination),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        ).returncode
        == 0
    )


def available_viewer_port():
    """Leave existing viewers alone; select a bindable local review port."""
    for port in range(8878, 8891):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("No available review viewer port in8878..8890")


def wait_for_viewer(viewer, port, display, *, timeout_seconds=90):
    """Require the live child to serve this exact completed bank, not just a port."""
    expected_bank = hashlib.sha256(
        str((display / "stages").resolve()).encode()
    ).hexdigest()[:12]
    deadline = time.monotonic() + timeout_seconds
    last_error = "Viewer has not answered its catalog request"
    while time.monotonic() < deadline:
        code = viewer.poll()
        if code is not None:
            raise RuntimeError(f"Review viewer exited with code{code}; see viewer.log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/catalog", timeout=2
            ) as response:
                catalog = json.load(response)
            banks = [
                bank
                for bank in catalog.get("banks", [])
                if bank.get("id") == expected_bank
            ]
            entries = [
                entry
                for entry in catalog.get("entries", [])
                if entry.get("bank") == expected_bank
            ]
            variations = [
                entry
                for entry in entries
                if (entry.get("variation") or {}).get("kind") == "variation"
            ]
            if (
                len(banks) == 1
                and banks[0].get("compatible")
                and len(variations) == 1800
                and all(
                    (entry.get("variation") or {}).get("training_eligible")
                    for entry in variations
                )
            ):
                if viewer.poll() is not None:
                    raise RuntimeError(
                        "Review viewer exited after its catalog response; see viewer.log"
                    )
                return {
                    "viewer_ready": True,
                    "viewer_catalog_entries": len(entries),
                    "viewer_accepted_variations": len(variations),
                    "viewer_verified_unix": time.time(),
                }
            last_error = f"Catalog is missing the completed bank{expected_bank} or its1800 admitted variations"
        except (OSError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.5)
    if viewer.poll() is None:
        viewer.terminate()
        try:
            viewer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            viewer.kill()
            viewer.wait(timeout=5)
    raise RuntimeError(
        f"Review viewer was not ready within{timeout_seconds}s: {last_error}; see viewer.log"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=int, required=True)
    p.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[3]
    )
    p.add_argument("--timeout-hours", type=float, default=12)
    a = p.parse_args()
    workspace = a.workspace.resolve()
    project = workspace / "robotics-assembly-3.0"
    out = project / "outputs/reset-bank-chimera-completion-20261107"
    out.mkdir(exist_ok=True)
    with (out / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            out / "status.json",
            {"status": "waiting_for_complete_bank", "job": a.job, "pid": os.getpid()},
        )
        end = time.monotonic() + a.timeout_hours * 3600
        terminal = 0
        while time.monotonic() < end:
            target = out / "return-receipt.partial.json"
            if fetch("exports/return-receipt.json", target):
                receipt = json.loads(target.read_text())
                if (
                    str(receipt.get("slurm_job_id")) != str(a.job)
                    or receipt.get("accepted") != 1800
                    or receipt.get("status") != "complete"
                ):
                    raise ValueError(
                        "Return receipt does not belong to the complete authorized job"
                    )
                target.replace(out / "return-receipt.json")
                break
            query = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "chimera-login",
                    f"sacct -n -X -j {a.job} --format=State --parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=35,
                check=False,
            )
            state = (
                query.stdout.strip().split("|")[0].strip()
                if query.returncode == 0
                else "UNKNOWN"
            )
            terminal = (
                terminal + 1
                if state
                in {
                    "FAILED",
                    "CANCELLED",
                    "COMPLETED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                }
                else 0
            )
            write(
                out / "status.json",
                {
                    "status": "waiting_for_complete_bank",
                    "job": a.job,
                    "scheduler_state": state,
                    "checked_unix": time.time(),
                    "pid": os.getpid(),
                },
            )
            if terminal >= 3:
                fetch("result.json", out / "remote-result.json")
                write(
                    out / "status.json",
                    {
                        "status": "remote_job_ended_without_complete_export",
                        "job": a.job,
                        "scheduler_state": state,
                    },
                )
                return 2
            time.sleep(60)
        else:
            write(
                out / "status.json",
                {"status": "bounded_transfer_wait_expired", "job": a.job},
            )
            return 2
        archive = out / "reset-bank-result.partial.tar.gz"
        if not fetch("exports/reset-bank-result.tar.gz", archive):
            raise RuntimeError("Completed bank download failed; retained the receipt")
        archive.replace(out / "reset-bank-result.tar.gz")
        display = (
            project / "assets_local/wowrobo/randomization-bank-1800-valid-chimera-v4"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("ingest.py")),
            "--archive",
            str(out / "reset-bank-result.tar.gz"),
            "--sha256",
            receipt["archive_sha256"],
            "--output",
            str(out / "linux-qualified-result"),
            "--display",
            str(display),
            "--workspace",
            str(workspace),
        ]
        with (out / "ingest.log").open("a") as log:
            subprocess.run(
                command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        port = available_viewer_port()
        env = dict(os.environ, PYTHONPATH=str(workspace / "assembly-so101"))
        with (out / "viewer.log").open("a") as log:
            viewer = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "assembly_policy.viewer",
                    "--scene",
                    "wowrobo-layout",
                    "--layout",
                    str(display),
                    "--port",
                    str(port),
                ],
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        readiness = wait_for_viewer(viewer, port, display)
        write(
            out / "status.json",
            {
                "status": "complete_and_published",
                **readiness,
                "accepted": 1800,
                "job": a.job,
                "display": str(display),
                "qualified_platform": "Linux",
                "local_native_admission_repeated": False,
                "viewer_url": f"http://localhost:{port}/",
                "viewer_pid": viewer.pid,
            },
        )
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        p = (
            Path(__file__).resolve().parents[3]
            / "robotics-assembly-3.0/outputs/reset-bank-chimera-completion-20261107"
        )
        if p.exists():
            write(
                p / "status.json",
                {
                    "status": "transfer_or_publication_failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
