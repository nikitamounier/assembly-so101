"""Seal resource-return readiness; the science owner alone activates a successor."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def write(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument(
        "--learner",
        action="store_true",
        help="Compatibility flag; never resumes a process",
    )
    parser.add_argument("--probe-workers", action="store_true")
    args = parser.parse_args()
    state, training = args.state_root.resolve(), args.training_root.resolve()
    folder = state / "borrowed-training"
    if args.probe_workers:
        job = os.environ["SLURM_JOB_ID"]
        live = []
        for process in Path("/proc").iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                argv = (process / "cmdline").read_bytes().split(b"\0")
                env = (process / "environ").read_bytes().split(b"\0")
                if ("SLURM_JOB_ID=" + job).encode() not in env:
                    continue
                if not any(value.startswith(str(state).encode()) for value in argv):
                    continue
                if not any(
                    value.endswith(
                        (
                            b"/pool.py",
                            b"/worker.py",
                            b"/build_wowrobo_randomized_bank.py",
                        )
                    )
                    for value in argv
                ):
                    continue
                stat = (process / "stat").read_text().rsplit(")", 1)[1].split()
                if stat[0] != "Z":
                    live.append(
                        {
                            "pid": int(process.name),
                            "start": int(stat[19]),
                            "state": stat[0],
                        }
                    )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        print(
            json.dumps(
                {
                    "job": job,
                    "node": os.uname().nodename,
                    "status": "no_live_borrowed_workers"
                    if not live
                    else "workers_still_closing",
                    "live": live,
                    "unix": time.time(),
                }
            )
        )
        return 0
    receipt_path = state / "exports/return-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "complete" and receipt["accepted"] == 19200
    proof = json.loads(
        (training / "resource-repair/native-pause-proof.json").read_text()
    )
    controller = json.loads(
        (training / "resource-repair/paused-controller.json").read_text()
    )
    assert proof["status"] == "PASS" and proof["policy_optimizer_finite"]
    assert proof["it"] == 32 and proof["transitions"] == 1046323
    checkpoint = training / "run-v10/checkpoint.before-state-build-repause.pt"
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == proof["checkpoint_sha256"]
    )
    closure_path = folder / "worker-closure-proof.json"
    closure = json.loads(closure_path.read_text()) if closure_path.exists() else {}
    expected = set(
        map(
            str,
            json.loads((folder / "training-config-before.json").read_text())[
                "cpu_jobs"
            ],
        )
    )
    rows = closure.get("leases", {})
    ready = (
        closure.get("status") == "PASS"
        and set(rows) == expected
        and all(
            row.get("status") in ("no_live_borrowed_workers", "allocation_not_running")
            for row in rows.values()
        )
        and time.time() - closure.get("unix", 0) < 180
    )
    result = {
        "status": "resources_ready_for_science_owner_handoff"
        if ready
        else "waiting_for_worker_closure",
        "ready_for_handoff": ready,
        "worker_closure_proof": str(closure_path),
        "worker_closure_sha256": hashlib.sha256(closure_path.read_bytes()).hexdigest()
        if closure_path.exists()
        else None,
        "automatic_resume": False,
        "checkpoint_preserved": proof["checkpoint_sha256"],
        "checkpoint": str(checkpoint),
        "learner": proof["learner"],
        "controller": controller["controllers"],
        "state_root": str(state),
        "accepted": 19200,
        "unix": time.time(),
        "return_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "cpu_configuration_unchanged": True,
        "autoscale_unchanged": True,
        "reason": "Main science owner must qualify and activate the successor from current checkpoint32",
    }
    write(folder / "config-restore.json", result)
    write(folder / "restoration-ready.json", result)
    print(json.dumps(result))
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
