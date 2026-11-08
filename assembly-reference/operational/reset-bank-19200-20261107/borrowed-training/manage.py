"""Recover borrowed lease attachments; only the science owner restores training."""

import concurrent.futures
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUT = PROJECT / "outputs/reset-bank-19200-borrowed-training"
STATE = "/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1"
TRAIN = "/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4"
PY = "/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python"


def write(value):
    temporary = OUT / f"status.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUT / "status.json")


def scheduler(jobs):
    p = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "chimera-login",
            shlex.join(["squeue", "-j", ",".join(jobs), "-h", "-o", "%i|%T"]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return dict(line.split("|", 1) for line in p.stdout.splitlines() if "|" in line)


def remote(job, arguments):
    command = shlex.join(
        [
            "srun",
            "--jobid=" + job,
            "--overlap",
            "--exact",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=1G",
            "--cpu-bind=none",
            PY,
            *arguments,
        ]
    )
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "chimera-login", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )


def complete():
    receipt = OUT / "return-receipt.probe.json"
    p = subprocess.run(
        [
            "scp",
            "-q",
            "chimera-login:" + STATE + "/exports/return-receipt.json",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if p.returncode != 0:
        return False
    result = json.loads(receipt.read_text())
    return result.get("status") == "complete" and result.get("accepted") == 19200


def main():
    OUT.mkdir(exist_ok=True)
    lease = (OUT / "manager.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [
        str(j)
        for j in json.loads(
            (
                PROJECT / "operational/wowrobo-chimera-20261107/modal-live/ops-v10.json"
            ).read_text()
        )["config"]["cpu_jobs"]
    ]
    assert len(jobs) == 40
    handles, attempts, exits = {}, {}, {}
    deadline = time.monotonic() + 5 * 3600
    while time.monotonic() < deadline:
        if complete():
            break
        states = scheduler(jobs)
        for job in jobs:
            child = handles.get(job)
            if child is not None and child.poll() is None:
                continue
            if child is not None:
                exits[job] = child.returncode
                handles.pop(job)
            # Exit zero is only a process exit. Completion is the sealed bank above.
            if states.get(job) != "RUNNING":
                continue
            command = shlex.join(
                [
                    "srun",
                    "--jobid=" + job,
                    "--overlap",
                    "--exact",
                    "--ntasks=1",
                    "--cpus-per-task=8",
                    "--mem=16G",
                    "--cpu-bind=cores",
                    PY,
                    STATE + "/launch/borrowed-entry.py",
                    "--state-root",
                    STATE,
                    "--training-root",
                    TRAIN,
                ]
            )
            log = (OUT / f"job-{job}.log").open("a")
            handles[job] = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "chimera-login", command],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            attempts[job] = attempts.get(job, 0) + 1
        marker = (
            PROJECT
            / "operational/wowrobo-chimera-20261107/modal-live/handoff-active-v10.json"
        )
        value = json.loads(marker.read_text())
        value.update(
            status="intentional_handoff",
            time_unix=time.time(),
            state_build_priority=True,
            state_build_root=STATE,
            reason="State build owns CPUs; checkpoint32 preserved; science owner activates successor",
        )
        marker.write_text(json.dumps(value) + "\n")
        write(
            {
                "status": "state_build_priority_active",
                "running_borrowed": sum(p.poll() is None for p in handles.values()),
                "attempts": attempts,
                "last_exits": exits,
                "scheduler": states,
                "pid": os.getpid(),
                "unix": time.time(),
                "automatic_resume": False,
            }
        )
        time.sleep(15)
    else:
        raise RuntimeError(
            "Resource-priority wait expired; preserve pause for owner review"
        )
    closure = None
    for _ in range(40):
        states = scheduler(jobs)

        def probe(job, states=states):
            if states.get(job) != "RUNNING":
                return job, {
                    "status": "allocation_not_running",
                    "scheduler_state": states.get(job, "ABSENT"),
                    "unix": time.time(),
                }
            try:
                result = remote(
                    job,
                    [
                        STATE + "/launch/priority-restore.py",
                        "--state-root",
                        STATE,
                        "--training-root",
                        TRAIN,
                        "--probe-workers",
                    ],
                )
                if result.returncode == 0:
                    return job, json.loads(result.stdout)
            except (subprocess.SubprocessError, ValueError):
                pass
            return job, {"status": "probe_failed", "unix": time.time()}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            rows = dict(executor.map(probe, jobs))
        passed = all(
            row["status"] in ("no_live_borrowed_workers", "allocation_not_running")
            for row in rows.values()
        )
        closure = {
            "status": "PASS" if passed else "WAITING",
            "leases": rows,
            "unix": time.time(),
            "automatic_resume": False,
        }
        path = OUT / "worker-closure-proof.json"
        path.write_text(json.dumps(closure, indent=2) + "\n")
        copied = subprocess.run(
            [
                "scp",
                "-q",
                str(path),
                "chimera-login:"
                + STATE
                + "/borrowed-training/worker-closure-proof.json",
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if passed and copied.returncode == 0:
            break
        write(
            {
                "status": "bank_complete_waiting_worker_closure",
                "closure": closure,
                "pid": os.getpid(),
                "automatic_resume": False,
            }
        )
        time.sleep(15)
    if closure is None or closure["status"] != "PASS":
        raise RuntimeError(
            "Bank complete but borrowed workers not yet confirmed closed; retain training pause"
        )
    for job in ["3625780", *jobs, "3598717", "3606264"]:
        if scheduler([job]).get(job) != "RUNNING":
            continue
        result = remote(
            job,
            [
                STATE + "/launch/priority-restore.py",
                "--state-root",
                STATE,
                "--training-root",
                TRAIN,
            ],
        )
        if result.returncode == 0:
            write(
                {
                    "status": "resources_ready_for_science_owner_handoff",
                    "restore": json.loads(result.stdout),
                    "pid": os.getpid(),
                    "unix": time.time(),
                    "automatic_resume": False,
                }
            )
            return 0
    raise RuntimeError(
        "Bank complete; owner handoff proof needs attention; training remains paused"
    )


if __name__ == "__main__":
    raise SystemExit(main())
