"""Allocated cube-trial orchestration; original leases and science stay separate."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
CONFIG = OPS / "cluster.json"
STATE = ROOT / "artifacts_local/chimera/controller.json"
OLD_RUN = "/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/runs/chimera-v13-8192-20261106"
TERMINAL = {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", "OUT_OF_MEMORY", "NODE_FAIL"}


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def native(job):
    r = subprocess.run(
        ["scontrol", "show", "job", str(job), "-o"], capture_output=True, text=True, timeout=20
    )
    if r.returncode:
        return {"JobState": "UNKNOWN", "error": r.stderr.strip()}
    return dict(x.split("=", 1) for x in r.stdout.split() if "=" in x)


def clean_env():
    return {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("SLURM_", "SRUN_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"
        },
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MUJOCO_GL": "egl",
    }


def launch(job, cpus, args, log, tasks=1):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps(dict(launch_utc=time.time(), job=job, tasks=tasks, cpus=cpus)) + "\n")
        stream.flush()
        return subprocess.Popen(
            [
                "srun",
                f"--jobid={job}",
                "--overlap",
                "--exact",
                f"--ntasks={tasks}",
                f"--cpus-per-task={cpus}",
                *args,
            ],
            cwd=ROOT,
            env=clean_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def remaining(row):
    try:
        return time.mktime(time.strptime(row["EndTime"], "%Y-%m-%dT%H:%M:%S")) - time.time()
    except (ValueError, KeyError):
        return 0


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


def learner(cfg, stop=False):
    run = ROOT / cfg["run"]
    identity = run / "learner-process.json"
    assert os.environ["SLURM_JOB_ID"] == cfg["gpu_job"]
    if stop:
        row = json.loads(identity.read_text())
        assert row["host"] == socket.gethostname()
        p = Path("/proc") / str(row["pid"])
        if p.exists():
            command = (p / "cmdline").read_bytes().split(b"\0")
            assert b"assembly_policy.cube_trial.train" in command and str(run).encode() in command
            os.kill(row["pid"], signal.SIGTERM)
        return
    atomic(identity, dict(pid=os.getpid(), host=socket.gethostname(), job=os.environ["SLURM_JOB_ID"]))
    command = [
        cfg["python"],
        "-m",
        "assembly_policy.cube_trial.train",
        "--run-dir",
        str(run),
        "--bank-dir",
        str(ROOT / cfg["bank"]),
        "--envs",
        "1024",
        "--n-part",
        "256",
        "--n-fix",
        "128",
        "--steps",
        "32",
        "--iters",
        "10000",
        "--device",
        "cuda",
        "--remote-workers",
        str(len(cfg["cpu_jobs"]) * 5),
        "--nthread",
        "8",
        "--accept-timeout",
        "60",
        "--min-conn-frac",
        "0.125",
        "--worker-init-wait",
        "300",
        "--bind-host",
        "0.0.0.0",
        "--advertise-host",
        socket.gethostbyname(socket.gethostname()),
        "--trusted-network",
        "--save-every",
        "1",
        "--keep-every",
        "500",
        "--no-eval",
        "--cost-coef",
        "0.00002",
        "--action-rate-coef",
        "0",
        "--sampler",
        "beta",
        "--eps-q",
        ".01",
        "--seed",
        "20261107",
    ]
    atomic(run / "command.json", command)
    os.execv(command[0], command)


def last_record(run):
    p = run / "log.jsonl"
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    return json.loads(lines[-1]) if lines else None


class AdoptedProcess:
    """Observe an exact existing srun without disturbing its owning Slurm step."""

    def __init__(self, record, run, kind):
        self.pid = record["pid"]
        self.start = record["start"]
        self.returncode = None
        self.run = run
        self.kind = kind
        path = Path("/proc") / str(self.pid)
        if path.exists():
            argv = (path / "cmdline").read_bytes().split(b"\0")
            assert (path / "stat").read_text().split()[21] == self.start
            assert Path(argv[0].decode()).name == "srun"
            if kind == "worker":
                assert b"assembly_policy.cube_trial.remote" in argv and str(run).encode() in argv
            else:
                assert b"learner" in argv and str(Path(__file__)).encode() in argv

    def poll(self):
        path = Path("/proc") / str(self.pid) / "stat"
        try:
            row = path.read_text().split()
            if row[21] == self.start and row[2] != "Z":
                return None
        except FileNotFoundError:
            pass
        self.returncode = (
            0 if self.kind == "learner" and (last_record(self.run) or {}).get("it", 0) >= 10000 else 255
        )
        return self.returncode


def control(cfg, adopt=False):
    assert os.environ.get("SLURM_JOB_ID")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lock = (STATE.parent / cfg.get("controller_lock", "controller.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists() and not adopt:
        raise RuntimeError(
            "Existing controller state requires explicit process reconciliation before restart"
        )
    if not (ROOT / "artifacts_local/chimera/READY.json").is_file():
        raise RuntimeError("Allocated native/CUDA preflight has not completed")
    run = ROOT / cfg["run"]
    run.mkdir(parents=True, exist_ok=True)
    worker_steps, handoffs, errors, exited_since = {}, {}, [], {}
    process = None
    generation = None
    draining = False
    if adopt:
        receipt = json.loads((STATE.parent / "controller-adoption.json").read_text())
        assert receipt["control_job"] == os.environ["SLURM_JOB_ID"]
        old = Path("/proc") / str(receipt["old_controller_pid"])
        row = (old / "stat").read_text().split()
        assert row[21] == receipt["old_controller_start"] and row[2] == "T"
        argv = (old / "cmdline").read_bytes().split(b"\0")
        assert any(x.endswith(b"operational/cube-trial-20261107/cluster.py") for x in argv) and b"control" in argv
        for j, rec in receipt["workers"].items():
            assert j in cfg["cpu_jobs"]
            worker_steps[j] = (AdoptedProcess(rec, run, "worker"), rec["epoch"])
            handoffs[j] = rec["epoch"]
        if receipt.get("learner"):
            process = AdoptedProcess(receipt["learner"], run, "learner")
            generation = receipt["gpu_epoch"]
            draining = receipt["draining"]
    while True:
        errors = []
        jobs = {j: native(j) for j in [cfg["gpu_job"], *cfg["cpu_jobs"]]}
        gpu = jobs[cfg["gpu_job"]]
        terminal = [j for j, r in jobs.items() if r["JobState"] in TERMINAL]
        phase = "running" if process is not None else "waiting_allocations"
        for j in cfg["cpu_jobs"]:
            row = jobs[j]
            epoch = row.get("Restarts")
            old = worker_steps.get(j)
            if old and old[0].poll() is not None:
                if old[1] == epoch and row["JobState"] == "RUNNING":
                    exited_since.setdefault(j, time.time())
                    if time.time() - exited_since[j] > 15:
                        errors.append(f"worker_exit:{j}:{old[0].returncode}")
                    continue
                worker_steps.pop(j)
                exited_since.pop(j, None)
            if row["JobState"] != "RUNNING" or remaining(row) < 60 or j in worker_steps:
                continue
            if handoffs.get(j) != epoch:
                result = launch(
                    j, 1, [cfg["python"], str(Path(__file__)), "handoff"], run / f"handoff-{j}.log"
                )
                if result.wait(timeout=60) != 0:
                    errors.append(f"handoff_failed:{j}")
                    continue
                handoffs[j] = epoch
            worker_steps[j] = (
                launch(
                    j,
                    2,
                    [
                        cfg["python"],
                        "-m",
                        "assembly_policy.cube_trial.remote",
                        "--run-dir",
                        str(run),
                        "--nthread",
                        "2",
                        "--wait",
                        "7200",
                        "--poll",
                        "1",
                    ],
                    run / f"workers-{j}.log",
                    tasks=5,
                ),
                epoch,
            )
        if process is not None and process.poll() is not None:
            code = process.returncode
            if (
                draining
                or gpu["JobState"] in ("PENDING", "CONFIGURING", "REQUEUED")
                or gpu.get("Restarts") != generation
                or code == 3
            ):
                process = None
                phase = "waiting_allocations"
            elif code == 0 and (last_record(run) or {}).get("it", 0) >= 10000:
                phase = "complete"
            else:
                phase = "learner_failed"
                errors.append(f"learner_exit:{code}")
        if process is None and gpu["JobState"] == "RUNNING" and remaining(gpu) > 360 and worker_steps:
            process = launch(
                cfg["gpu_job"], 8, [cfg["python"], str(Path(__file__)), "learner"], run / "trainer.log"
            )
            generation = gpu.get("Restarts")
            draining = False
            phase = "learner_starting"
        if (
            process is not None
            and process.poll() is None
            and not draining
            and gpu["JobState"] == "RUNNING"
            and remaining(gpu) < 360
        ):
            stop = launch(cfg["gpu_job"], 1, [cfg["python"], str(Path(__file__)), "stop"], run / "stop.log")
            if stop.wait(timeout=60):
                errors.append("checkpoint_request_failed")
            draining = True
        if cfg["gpu_job"] in terminal:
            phase = "allocation_terminal"
        atomic(
            STATE,
            dict(
                phase=phase,
                observed_utc=time.time(),
                controller_pid=os.getpid(),
                control_job=os.environ["SLURM_JOB_ID"],
                gpu_job=cfg["gpu_job"],
                cpu_jobs=cfg["cpu_jobs"],
                jobs=jobs,
                terminal_jobs=terminal,
                worker_steps={
                    j: dict(pid=p.pid, epoch=e, running=p.poll() is None)
                    for j, (p, e) in worker_steps.items()
                },
                learner_srun_pid=process.pid if process else None,
                draining=draining,
                errors=sorted(set(errors)),
                run=str(run),
            ),
        )
        if phase in ("complete", "learner_failed", "allocation_terminal"):
            return 0 if phase == "complete" else 2
        time.sleep(5)


def observe(cfg):
    assert os.environ.get("SLURM_JOB_ID")
    hashes = json.loads((ROOT / "artifacts_local/chimera/source-hashes.json").read_text())
    bad = [
        p
        for p, sha in hashes.items()
        if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha
    ]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    run = ROOT / cfg["run"]
    latest = last_record(run)
    errors = list(state.get("errors", []))
    if not state or time.time() - state.get("observed_utc", 0) > 120:
        errors.append("controller_stale")
    if state.get("phase") in ("learner_failed", "allocation_terminal"):
        errors.append(state["phase"])
    if latest and any(
        not __import__("math").isfinite(latest[k]) for k in ("fps", "lr", "std", "success") if k in latest
    ):
        errors.append("nonfinite_update")
    checkpoint = run / "checkpoint.pt"
    print(
        json.dumps(
            dict(
                allocated_monitor_job=os.environ["SLURM_JOB_ID"],
                run=str(run),
                source_manifest_sha256=hashlib.sha256(
                    (ROOT / "artifacts_local/chimera/source-hashes.json").read_bytes()
                ).hexdigest(),
                source_valid=not bad,
                source_errors=bad,
                controller=state,
                last_update=latest,
                progress_age_s=time.time() - (run / "log.jsonl").stat().st_mtime if latest else None,
                checkpoint_bytes=checkpoint.stat().st_size if checkpoint.exists() else 0,
                errors=errors,
                physics=json.loads((ROOT / "artifacts_local/chimera/READY.json").read_text()),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["handoff", "learner", "stop", "control", "observe"])
    a.add_argument("--adopt", action="store_true")
    args = a.parse_args()
    mode = args.mode
    cfg = json.loads(CONFIG.read_text())
    if mode == "handoff":
        handoff(cfg)
    elif mode in ("learner", "stop"):
        learner(cfg, stop=mode == "stop")
    elif mode == "control":
        sys.exit(control(cfg, adopt=args.adopt))
    else:
        observe(cfg)
