"""Bounded worker processes, shared by local, Modal, and Sail launchers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_workers(ticket: dict, processes: int, threads: int, seconds: float) -> dict:
    if ticket.get("kind") == "probe":
        from .probe import measure

        return measure(ticket)
    if processes < 1 or threads < 1 or seconds <= 0:
        raise ValueError("Positive workers, threads and duration are required")
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, stop)
    started = time.monotonic()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        with tempfile.TemporaryDirectory(prefix="so101-workers-") as directory:
            path = Path(directory) / "ticket.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(ticket, stream)
            command = [
                sys.executable,
                "-m",
                "assembly_policy.rl.remote",
                "--ticket",
                str(path),
                "--nthread",
                str(threads),
                "--wait",
                str(seconds),
                "--poll",
                "1",
            ]
            for _ in range(processes):
                children.append(subprocess.Popen(command, env=env))
            while not stopping and time.monotonic() - started < seconds:
                failed = [child.returncode for child in children if child.poll() is not None]
                if failed:
                    raise RuntimeError(f"Worker process exited: {failed}")
                time.sleep(0.25)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {"workers": processes, "threads": threads, "elapsed_s": time.monotonic() - started}
