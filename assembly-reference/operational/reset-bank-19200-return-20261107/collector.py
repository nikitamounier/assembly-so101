"""One-shot completion handoff: allocated remote read, download, ingest and viewer.

No Slurm submission/cancellation, training mutation or prior-bank overwrite.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_ROOT = (
    "/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1"
)
WORKSPACE = Path(__file__).resolve().parents[3]


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def read_completion_receipt(host, root, directory):
    """File transfer stays available after the allocated controller exits.

    Only the exact completion receipt is fetched on the login transport. Until
    it exists, all progress inspection still runs inside the supplied allocation.
    """
    remote = str(Path(root) / "exports/return-receipt.json")
    temporary = directory / ".completion-receipt.partial.json"
    result = subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"{host}:{remote}",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        if "No such file or directory" in result.stderr:
            return None
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    receipt = json.loads(temporary.read_text())
    return {
        "kind": "ready",
        "receipt": receipt,
        "observation": "completion_receipt_file_transfer",
    }


def read_remote(host, job_id, root, remote_python):
    code = """import json,pathlib
r=pathlib.Path(__import__('sys').argv[1]);p=r/'exports/return-receipt.json'
if p.exists():
 d=json.loads(p.read_text());print(json.dumps(dict(kind='ready',receipt=d)))
else:
 d=dict(kind='waiting')
 for name in ['result.json','progress.json','controller-state.json']:
  p=r/name
  if p.exists():d[name]=json.loads(p.read_text())
 print(json.dumps(d))
"""
    command = shlex.join(
        [
            "srun",
            "--jobid=" + job_id,
            "--overlap",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--cpu-bind=none",
            remote_python,
            "-c",
            code,
            root,
        ]
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def unused_port(start):
    for port in range(start, start + 20):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)):
                return port
    raise RuntimeError("No free viewer port in requested range")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--job-id", required=True, help="Live allocated controller/read job ID"
    )
    p.add_argument("--host", default="chimera-login")
    p.add_argument("--remote-root", default=DEFAULT_ROOT)
    p.add_argument(
        "--remote-python",
        default="/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python",
    )
    p.add_argument(
        "--directory",
        type=Path,
        default=WORKSPACE
        / "robotics-assembly-3.0/outputs/reset-bank-19200-completion-20261107",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "robotics-assembly-3.0/outputs/reset-bank-19200-chimera-returned-v1",
    )
    p.add_argument(
        "--display",
        type=Path,
        default=WORKSPACE
        / "robotics-assembly-3.0/assets_local/wowrobo/randomization-bank-19200-chimera-display-v1",
    )
    p.add_argument("--port", type=int, default=8891)
    p.add_argument("--train", type=int, default=640)
    p.add_argument("--eval", type=int, default=128)
    p.add_argument("--poll-seconds", type=float, default=30)
    p.add_argument(
        "--once",
        action="store_true",
        help="Read status once; return without fetching when not complete",
    )
    a = p.parse_args()
    if (
        not re.fullmatch(r"\d+", a.job_id)
        or min(a.train, a.eval) <= 0
        or a.poll_seconds < 1
    ):
        raise ValueError(
            "Numeric allocation ID, positive quotas and polling interval required"
        )
    directory = a.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    with (directory / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            status.exists()
            and json.loads(status.read_text()).get("status") == "complete"
        ):
            prior = json.loads(status.read_text())
            if (
                prior.get("accepted") != 25 * (a.train + a.eval)
                or prior.get("train_per_stage") != a.train
                or prior.get("eval_per_stage") != a.eval
            ):
                raise ValueError(
                    "Collector directory belongs to a different bank quota"
                )
            print(status.read_text())
            return 0
        failures = 0
        while True:
            try:
                remote = read_completion_receipt(a.host, a.remote_root, directory)
                if remote is None:
                    remote = read_remote(
                        a.host, a.job_id, a.remote_root, a.remote_python
                    )
                failures = 0
            except (subprocess.SubprocessError, ValueError) as error:
                failures += 1
                write(
                    status,
                    {
                        "status": "observation_failed",
                        "job_id": a.job_id,
                        "error": str(error),
                        "consecutive_failures": failures,
                    },
                )
                if failures >= 3 or a.once:
                    return 2
                time.sleep(a.poll_seconds)
                continue
            write(
                status,
                {
                    "status": "waiting",
                    "job_id": a.job_id,
                    "remote": remote,
                    "observed_unix": time.time(),
                    "target": 25 * (a.train + a.eval),
                },
            )
            if remote["kind"] == "ready":
                break
            if a.once:
                return 0
            time.sleep(a.poll_seconds)
        receipt = remote["receipt"]
        target = 25 * (a.train + a.eval)
        if (
            receipt.get("status") != "complete"
            or receipt.get("accepted") != target
            or receipt.get("train_per_stage") != a.train
            or receipt.get("eval_per_stage") != a.eval
        ):
            raise ValueError(
                "Return receipt does not match the requested completed bank"
            )
        remote_archive = Path(receipt["archive"])
        if (
            not remote_archive.is_absolute()
            or ".." in remote_archive.parts
            or not remote_archive.is_relative_to(Path(a.remote_root) / "exports")
        ):
            raise ValueError("Return receipt archive path is outside isolated exports")
        write(directory / "return-receipt.json", receipt)
        archive = directory / "reset-bank-result.tar.gz"
        temporary = directory / "reset-bank-result.partial.tar.gz"
        subprocess.run(
            ["scp", f"{a.host}:{receipt['archive']}", str(temporary)], check=True
        )
        temporary.replace(archive)
        command = [
            sys.executable,
            str(Path(__file__).with_name("ingest.py")),
            "--archive",
            str(archive),
            "--sha256",
            receipt["archive_sha256"],
            "--output",
            str(a.output.resolve()),
            "--display",
            str(a.display.resolve()),
            "--train",
            str(a.train),
            "--eval",
            str(a.eval),
        ]
        with (directory / "ingest.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        port = unused_port(a.port)
        environment = dict(os.environ, PYTHONPATH=str(WORKSPACE / "assembly-so101"))
        command = [
            sys.executable,
            "-m",
            "assembly_policy.viewer",
            "--scene",
            "wowrobo-layout",
            "--layout",
            str(a.display.resolve()),
            "--port",
            str(port),
        ]
        with (directory / "viewer.log").open("a") as log:
            viewer = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 120
        url = f"http://localhost:{port}"
        while time.monotonic() < deadline:
            if viewer.poll() is not None:
                raise RuntimeError("New viewer exited; inspect viewer.log")
            try:
                with urllib.request.urlopen(
                    url + "/api/catalog", timeout=10
                ) as response:
                    catalog = json.load(response)
                rows = [
                    row
                    for row in catalog["entries"]
                    if row.get("review_kind") == "variation"
                ]
                if len(rows) != target:
                    raise ValueError(
                        f"Viewer has{len(rows)} variations, expected{target}"
                    )
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("New viewer did not become ready")
        result = {
            "status": "complete",
            "accepted": target,
            "train_per_stage": a.train,
            "eval_per_stage": a.eval,
            "job_id": a.job_id,
            "viewer_pid": viewer.pid,
            "viewer_url": url + "/",
            "display": str(a.display.resolve()),
            "returned_artifacts": str(a.output.resolve()),
            "completed_unix": time.time(),
        }
        write(status, result)
        print(json.dumps(result, indent=2))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--directory" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--directory") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            write(
                directory / "status.json",
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
        raise
