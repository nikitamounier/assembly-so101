"""Singleton attachment to a retained lease; adopt existing native pool safely."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def identity(pid):
    p = Path("/proc") / str(pid)
    try:
        if p.stat().st_uid != os.getuid():
            return None
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "start": int(stat[19]),
            "state": stat[0],
            "args": (p / "cmdline").read_bytes().split(b"\0"),
            "env": (p / "environ").read_bytes().split(b"\0"),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-root", type=Path, required=True)
    p.add_argument("--training-root", type=Path, required=True)
    a = p.parse_args()
    state, training = a.state_root.resolve(), a.training_root.resolve()
    job = os.environ["SLURM_JOB_ID"]
    folder = state / "borrowed-training"
    backup = json.loads((folder / "training-config-before.json").read_text())
    assert job in backup["cpu_jobs"]
    original = Path(
        "/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/runs/chimera-v13-8192-20261106"
    )
    assert (
        json.loads((original / "user-pause.json").read_text())["automatic_resume"]
        is False
    )
    lock = (folder / f"entry-{job}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "existing_attachment_owns_lease", "job": job}))
        return 75

    def priority():
        cfg = json.loads((training / "ops-v10/cluster.json").read_text())
        return not cfg["cpu_jobs"] and cfg.get("state_build_priority", {}).get(
            "state_root"
        ) == str(state)

    closed = []

    def guard_primary():
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            row = identity(proc.name)
            if row is None or row["args"][1:3] != [b"-m", b"assembly_policy.rl.remote"]:
                continue
            if (
                str(original).encode() not in row["args"]
                or ("SLURM_JOB_ID=" + job).encode() not in row["env"]
            ):
                continue
            if row["state"] not in ("T", "t", "Z"):
                os.kill(row["pid"], signal.SIGSTOP)
                closed.append({"pid": row["pid"], "start": row["start"]})

    assert priority()
    guard_primary()
    pools = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        row = identity(proc.name)
        if (
            row
            and row["state"] != "Z"
            and str(state / "launch/pool.py").encode() in row["args"]
            and str(state).encode() in row["args"]
            and ("SLURM_JOB_ID=" + job).encode() in row["env"]
        ):
            pools.append(row)
    if len(pools) > 1:
        raise RuntimeError(
            "Multiple native pools already occupy this lease; refusing another"
        )
    child = None
    if pools:
        pool = pools[0]
        adopted = True
    else:
        log = (folder / f"pool-{job}.log").open("a")
        child = subprocess.Popen(
            [
                sys.executable,
                str(state / "launch/pool.py"),
                "--root",
                str(state),
                "--workers",
                "1",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pool = identity(child.pid)
        assert pool is not None
        adopted = False
    proof = {
        "job": job,
        "node": os.uname().nodename,
        "pool_pid": pool["pid"],
        "pool_start": pool["start"],
        "entry_pid": os.getpid(),
        "workers": 1,
        "adopted": adopted,
        "lease_retained": True,
        "old_primary_never_resumed": True,
        "unix": time.time(),
    }
    receipt = folder / f"attached-{job}.json"
    write(receipt, proof)
    stopping = False

    def terminate(*_):
        nonlocal stopping
        stopping = True
        current = identity(pool["pid"])
        if current and current["start"] == pool["start"]:
            os.killpg(pool["pid"], signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while True:
        current = identity(pool["pid"])
        if (
            current is None
            or current["start"] != pool["start"]
            or current["state"] == "Z"
        ):
            break
        if not priority():
            terminate()
        guard_primary()
        proof.update(heartbeat_unix=time.time())
        write(receipt, proof)
        time.sleep(5)
    code = child.wait() if child else 0
    proof.update(
        exit=code,
        finished_unix=time.time(),
        preserved_old_primary=closed,
        stopped=stopping,
    )
    write(receipt, proof)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
