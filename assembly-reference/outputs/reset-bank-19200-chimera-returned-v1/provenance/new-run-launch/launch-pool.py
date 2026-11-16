"""Packed, physical-core-pinned work stealing for the immutable native reset plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def physical_cpus(allowed, topology=Path("/sys/devices/system/cpu")):
    """Choose one allowed hardware thread per distinct physical package/core."""
    seen, result = set(), []
    for cpu in sorted(allowed):
        folder = topology / f"cpu{cpu}" / "topology"
        key = (
            int((folder / "physical_package_id").read_text()),
            int((folder / "core_id").read_text()),
        )
        if key not in seen:
            seen.add(key)
            result.append(cpu)
    return result


def verify_done(root, task, namespace):
    path = root / "done" / f"{task['index']}.json"
    if not path.exists():
        return False
    proof = json.loads(path.read_text())
    expected = {key: task[key] for key in ("index", "stage", "part", "parts")}
    if (
        proof.get("status") != "complete"
        or proof.get("task") != expected
        or proof.get("accepted_train") != 10
        or proof.get("accepted_eval") != 2
        or proof.get("rng_namespace") != namespace
    ):
        raise ValueError(f"Invalid authoritative completion proof:{path}")
    for key in ("sample_identities", "input_sample_identities"):
        values = proof.get(key, [])
        if len(values) != 12 or len(set(values)) != 12:
            raise ValueError(f"Incomplete/repeated completion identities:{path}/{key}")
    return True


class Pool:
    def __init__(self, root, workers):
        self.root = root.resolve()
        self.settings = json.loads((root / "settings.json").read_text())
        if (self.settings["train"], self.settings["eval"], self.settings["parts"]) != (
            640,
            128,
            64,
        ):
            raise ValueError("Packed pool requires640/128 quotas and64 partitions")
        self.plan = Path(self.settings.get("plan", self.root / "plan")).resolve()
        self.tasks = json.loads((root / "assignment.json").read_text())["tasks"]
        if len(self.tasks) != 1600 or {t["index"] for t in self.tasks} != set(
            range(1600)
        ):
            raise ValueError("Expected exactly1600 unique task indices")
        self.tasks = sorted(self.tasks, key=lambda task: task["index"])
        if (
            len({t["stage"] for t in self.tasks}) != 25
            or len({(t["stage"], t["part"]) for t in self.tasks}) != 1600
            or any(t["parts"] != 64 or not 0 <= t["part"] < 64 for t in self.tasks)
        ):
            raise ValueError("Assignment does not partition25 stages into64 parts each")
        self.job = os.environ.get("SLURM_JOB_ID")
        if not self.job:
            raise RuntimeError("Packed computation requires a Slurm allocation")
        allowed = sorted(os.sched_getaffinity(0))
        cores = physical_cpus(allowed)
        if workers < 1 or workers > len(cores):
            raise ValueError(
                f"Requested{workers} workers but only{len(cores)} physical cores allowed"
            )
        self.cpus = cores[:workers]
        self.stop = threading.Event()
        self.children = {}
        self.children_lock = threading.Lock()
        self.thread_errors = []
        self.pool_id = uuid.uuid4().hex
        self.proof = {
            "pool_id": self.pool_id,
            "job": self.job,
            "node": socket.gethostname(),
            "pid": os.getpid(),
            "allowed_affinity": allowed,
            "distinct_physical_core_cpus": cores,
            "selected_worker_cpus": self.cpus,
            "workers": workers,
            "settings_sha256": hashlib.sha256(
                (root / "settings.json").read_bytes()
            ).hexdigest(),
            "assignment_sha256": hashlib.sha256(
                (root / "assignment.json").read_bytes()
            ).hexdigest(),
            "pool_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "started_unix": time.time(),
        }
        self.record("starting")

    def record(self, status, **extra):
        atomic(
            self.root / "pools" / f"{self.job}.json",
            dict(self.proof, status=status, updated_unix=time.time(), **extra),
        )
        # Per-process history survives requeue reuse of the same scheduler job ID.
        atomic(
            self.root / "pools" / f"{self.job}-{self.pool_id}.json",
            dict(self.proof, status=status, updated_unix=time.time(), **extra),
        )

        with (self.root / ".pool-registry.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            registry_path = self.root / "pool.json"
            registry = (
                json.loads(registry_path.read_text())
                if registry_path.exists()
                else {"pools": {}}
            )
            registry["pools"][str(self.job)] = {
                "pool_id": self.pool_id,
                "pid": os.getpid(),
                "job": self.job,
                "node": self.proof["node"],
                "workers": len(self.cpus),
                "cpus": self.cpus,
                "status": status,
                "updated_unix": time.time(),
            }
            atomic(registry_path, registry)

    @contextmanager
    def dispatch(self):
        with (self.root / ".dispatch.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            state = json.loads((self.root / "dispatch.json").read_text())
            state.setdefault("completed", [])
            try:
                yield state
            finally:
                atomic(self.root / "dispatch.json", state)

    def lease(self, task):
        folder = self.plan / "partitions" / task["stage"] / str(task["part"])
        folder.mkdir(parents=True, exist_ok=True)
        fd = os.open(folder / ".lease", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        os.set_inheritable(fd, True)
        return fd

    def done(self, task):
        return verify_done(self.root, task, self.settings["rng_namespace"])

    def claim(self, cpu):
        with self.dispatch() as state:
            if self.stop.is_set() or (self.root / "STOP").exists():
                return None, "stopped"
            completed = set(state["completed"])
            # Reconcile completed owners only, rather than rescanning1600 JSONs
            # on every claim across300 workers on a shared filesystem.
            for key in list(state["active"]):
                index = int(key)
                if self.done(self.tasks[index]):
                    completed.add(index)
                    state["active"].pop(key, None)
            state["completed"] = sorted(completed)
            if len(completed) == len(self.tasks):
                if not all(self.done(task) for task in self.tasks):
                    raise RuntimeError(
                        "Cached completion list lost an authoritative done proof"
                    )
                return None, "complete"
            candidates = []
            while state["cursor"] < len(self.tasks):
                index = state["cursor"]
                state["cursor"] += 1
                candidates.append(index)
                # New work takes priority; abandoned leases are revisited once
                # every original task has been offered at least once.
                task = self.tasks[index]
                if index in completed or self.done(task):
                    completed.add(index)
                    continue
                if state["failures"].get(str(index), {}).get("count", 0) >= 3:
                    continue
                fd = self.lease(task)
                if fd is not None:
                    return self.grant(state, task, fd, cpu, completed), "claimed"
                state["active"].setdefault(
                    str(index), {"status": "lease_held_by_prior_owner"}
                )
            for key in list(state["active"]):
                index = int(key)
                if index in completed:
                    continue
                if state["failures"].get(key, {}).get("count", 0) >= 3:
                    continue
                fd = self.lease(self.tasks[index])
                if fd is not None:
                    return self.grant(
                        state, self.tasks[index], fd, cpu, completed
                    ), "reclaimed"
            state["completed"] = sorted(completed)
            terminal = [
                key
                for key, failure in state["failures"].items()
                if failure.get("count", 0) >= 3 and int(key) not in completed
            ]
            return None, "blocked" if terminal and not state["active"] else "idle"

    def grant(self, state, task, fd, cpu, completed):
        key = str(task["index"])
        claim_id = uuid.uuid4().hex
        state["attempts"][key] = state["attempts"].get(key, 0) + 1
        record = {
            "claim_id": claim_id,
            "pool_id": self.pool_id,
            "job": self.job,
            "node": socket.gethostname(),
            "pool_pid": os.getpid(),
            "cpu": cpu,
            "task": task,
            "status": "claimed",
            "claimed_unix": time.time(),
        }
        state["active"][key] = record
        state["completed"] = sorted(completed)
        return task, fd, claim_id

    def finished(self, task, claim_id, code, error=None):
        with self.dispatch() as state:
            key = str(task["index"])
            complete = self.done(task)
            active = state["active"].get(key)
            if active and active.get("claim_id") != claim_id:
                raise RuntimeError(
                    "Another claimant replaced an inherited active lease"
                )
            if complete:
                state["completed"] = sorted(
                    set(state.get("completed", [])) | {task["index"]}
                )
                state["active"].pop(key, None)
            elif self.stop.is_set() or (self.root / "STOP").exists():
                if active:
                    active.update(
                        status="interrupted_reclaim_after_lease_release",
                        unix=time.time(),
                    )
            else:
                previous = state["failures"].get(key, {})
                failure = {
                    "count": previous.get("count", 0) + 1,
                    "last_exit": code,
                    "error": error,
                    "job": self.job,
                    "node": socket.gethostname(),
                    "last_failure_unix": time.time(),
                }
                state["failures"][key] = failure
                if failure["count"] >= 3:
                    state["active"].pop(key, None)
                elif active:
                    active.update(status="failed_retry_available", last_exit=code)

    def stop_children(self, *_):
        self.stop.set()
        with self.children_lock:
            children = list(self.children.values())
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def worker(self, cpu):
        try:
            while not self.stop.is_set() and not (self.root / "STOP").exists():
                granted, reason = self.claim(cpu)
                if granted is None:
                    if reason in ("complete", "blocked", "stopped"):
                        return
                    self.stop.wait(10)
                    continue
                task, fd, claim_id = granted
                child, code, error = None, None, None
                folder = self.plan / "partitions" / task["stage"] / str(task["part"])
                try:
                    env = dict(
                        os.environ,
                        PYTHONPATH=str(self.settings["project"]),
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1",
                    )
                    command = [
                        "taskset",
                        "-c",
                        str(cpu),
                        sys.executable,
                        str(self.root / "launch/worker.py"),
                        "--root",
                        str(self.root),
                        "--index",
                        str(task["index"]),
                        "--lease-fd",
                        str(fd),
                    ]
                    with (folder / "pool-worker.log").open("a") as log:
                        child = subprocess.Popen(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            pass_fds=(fd,),
                            start_new_session=True,
                        )
                        with self.children_lock:
                            self.children[cpu] = child
                        interrupted_at = None
                        while child.poll() is None:
                            if self.stop.is_set() or (self.root / "STOP").exists():
                                if interrupted_at is None:
                                    interrupted_at = time.monotonic()
                                    try:
                                        os.killpg(child.pid, signal.SIGTERM)
                                    except ProcessLookupError:
                                        pass
                                elif time.monotonic() - interrupted_at > 10:
                                    try:
                                        os.killpg(child.pid, signal.SIGKILL)
                                    except ProcessLookupError:
                                        pass
                            time.sleep(0.5)
                        code = child.returncode
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if child is not None and child.poll() is None:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=10)
                finally:
                    with self.children_lock:
                        self.children.pop(cpu, None)
                    try:
                        self.finished(task, claim_id, code, error)
                    finally:
                        os.close(fd)
        except Exception as exc:  # noqa: BLE001 - stop all owned children on a fatal thread error.
            self.thread_errors.append(f"{type(exc).__name__}: {exc}")
            self.stop_children()

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop_children)
        threads = [
            threading.Thread(target=self.worker, args=(cpu,)) for cpu in self.cpus
        ]
        self.record("running")
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.thread_errors:
            self.record("failed", errors=self.thread_errors)
            return 2
        with self.dispatch() as state:
            done_count = len(state.get("completed", []))
            failures = state["failures"]
        result = (
            "complete"
            if done_count == len(self.tasks)
            else (
                "interrupted"
                if self.stop.is_set() or (self.root / "STOP").exists()
                else "blocked"
            )
        )
        self.record(result, complete_tasks=done_count, failed_tasks=failures)
        return 0 if result in ("complete", "interrupted") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    return Pool(args.root, args.workers).run()


if __name__ == "__main__":
    raise SystemExit(main())
