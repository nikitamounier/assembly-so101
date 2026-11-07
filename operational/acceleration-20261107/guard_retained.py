"""Suspend only exact original workers inside the five transferred allocations."""

import json
import os
import signal
from pathlib import Path

OLD_RUN = "/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/runs/chimera-v13-8192-20261106"


def handoff(cfg):
    """Suspend only positively identified original worker PIDs; never kill steps."""
    job = os.environ["SLURM_JOB_ID"]
    assert job in cfg["cpu_jobs"]
    pause = json.loads((Path(OLD_RUN) / "user-pause.json").read_text())
    assert pause["automatic_resume"] is False and pause["checkpoint"]["it"] == 64
    group = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    while group != Path("/sys/fs/cgroup") and group.name != f"job_{job}":
        group = group.parent
    if group.name != f"job_{job}":
        raise RuntimeError("Cannot identify the allocation's memory cgroup")
    limit = (group / "memory.max").read_text().strip()
    used = int((group / "memory.current").read_text())
    free = None if limit == "max" else int(limit) - used
    if free is not None and free < 4 * 1024**3:
        raise RuntimeError(f"Insufficient memory headroom for cube workers: {free} bytes")
    workers = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            command = (p / "cmdline").read_bytes().split(b"\0")
            if (
                len(command) < 3
                or command[1:3] != [b"-m", b"assembly_policy.rl.remote"]
                or OLD_RUN.encode() not in command
            ):
                continue
            env = (p / "environ").read_bytes().split(b"\0")
            if f"SLURM_JOB_ID={job}".encode() not in env:
                continue
            start = (p / "stat").read_text().split()[21]
            if (p / "stat").read_text().split()[21] != start:
                continue
            workers.append(dict(pid=int(p.name), start=start))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(workers) != 5:
        raise RuntimeError(f"Expected five original workers in job {job}; found {len(workers)}")
    for worker in workers:
        p = Path("/proc") / str(worker["pid"])
        if (p / "stat").read_text().split()[21] != worker["start"]:
            raise RuntimeError("Original worker PID identity changed before suspension")
        os.kill(worker["pid"], signal.SIGSTOP)
    print(json.dumps(dict(job=job, suspended_original_workers=workers, memory_free_bytes=free)), flush=True)


handoff({"cpu_jobs": ["3602571", "3602586", "3602595", "3602600", "3602602"]})
