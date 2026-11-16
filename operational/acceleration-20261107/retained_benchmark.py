"""Full PPO comparison on a fixed available cohort; no original training jobs used."""

import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

BASE = Path("/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107")
SPARSE = BASE.with_name("acceleration-20261107-sparse")
OUT = BASE / "artifacts_local/retained-cohort"
assert os.environ.get("SLURM_JOB_ID"), "Controller must be allocated"
ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("SLURM_", "SRUN_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"
}
ENV.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl")


def row(job):
    native = subprocess.run(["scontrol", "show", "job", job, "-o"], capture_output=True, text=True)
    if native.returncode == 0:
        return dict(x.split("=", 1) for x in native.stdout.split() if "=" in x)
    accounting = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State,JobName%80"], text=True
    )
    matches = [line.split("|") for line in accounting.splitlines() if line.split("|")[0] == job]
    assert len(matches) == 1, dict(job=job, unavailable=native.stderr)
    item = matches[0]
    return dict(JobState=item[1].split()[0].rstrip("+"), JobName=item[2])


assert os.environ["SLURM_JOB_ID"] == "3602602"
lock = open(BASE / "artifacts_local/retained-controller.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if OUT.exists():
    OUT.rename(OUT.with_name("retained-cohort-retired-" + str(time.time_ns())))
OUT.mkdir(parents=True)
GPU = json.loads((BASE / "artifacts_local/retained-resources.json").read_text())["gpu"]
RETAINED = ["3602571", "3602586", "3602595", "3602600", "3602602"]
jobs = [dict(job=GPU, kind="gpu"), *[dict(job=j, kind="retained_cpu") for j in RETAINED]]
(OUT / "allocations.json").write_text(json.dumps(jobs, indent=2) + "\n")


def state(status, **fields):
    record = dict(
        status=status,
        controller_job="3602602",
        pid=os.getpid(),
        host=socket.gethostname(),
        start=Path("/proc/self/stat").read_text().split()[21],
        time=time.time(),
        **fields,
    )
    temp = OUT / "controller-state.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(OUT / "controller-state.json")


state("STARTING")


def launch(job, cpus, command, root, log, ntasks=1, mode="1"):
    return subprocess.Popen(
        [
            "srun",
            f"--jobid={job}",
            "--overlap",
            "--exact",
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpus}",
            "--cpu-bind=none",
            *command,
        ],
        cwd=root,
        env=dict(ENV, SO101_SURFACE_MODE=mode),
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
    )


initial = {}
cohort = []
gpu = None


def snapshot():
    rows = {job: row(job) for job in [gpu, *cohort]}
    assert all(item["JobState"] == "RUNNING" for item in rows.values()), rows
    if initial:
        for job, native in rows.items():
            for field in ["NodeList", "Restarts", "NumCPUs", "AllocTRES"]:
                assert native.get(field) == initial[job].get(field), dict(job=job, changed=field)
    return rows


try:
    gpu = GPU
    cohort = RETAINED
    deadline = time.time() + 1200
    while not all(row(j)["JobState"] == "RUNNING" for j in [gpu, *cohort]):
        assert time.time() < deadline, "Retained leases or GPU did not become RUNNING"
        time.sleep(5)
    for job in cohort:
        subprocess.run(
            [
                "srun",
                f"--jobid={job}",
                "--overlap",
                "--exact",
                "--ntasks=1",
                "--cpus-per-task=1",
                str(BASE / ".venv/bin/python"),
                str(BASE / "operational/acceleration-20261107/guard_retained.py"),
            ],
            env=ENV,
            check=True,
        )
    initial = snapshot()
    assert all(int(initial[job]["NumCPUs"]) == 10 for job in cohort)
    conns = 23
    (OUT / "cohort.json").write_text(
        json.dumps(
            dict(
                control_job=os.environ["SLURM_JOB_ID"],
                controller_cpus=4,
                gpu=gpu,
                cpu_jobs=cohort,
                simulator_cpus=46,
                workers=conns,
                initial_jobs=initial,
            ),
            indent=2,
        )
        + "\n"
    )
    cases = [
        ("smoke", BASE, "1", 8, 4, 1),
        ("flat-dense", BASE, "0", 8192, 32, 2),
        ("tree-dense", BASE, "1", 8192, 32, 2),
        ("tree-sparse", SPARSE, "1", 8192, 32, 2),
    ]
    microbatch = 65536
    for name, root, mode, envs, steps, iters in cases:
        before = snapshot()
        state("RUNNING", phase=name, jobs=initial)
        run = OUT / name
        run.mkdir()
        worker_jobs = cohort[:1] if name == "smoke" else cohort
        worker_count = 1 if name == "smoke" else conns
        python = str(root / ".venv/bin/python")
        bank = json.loads((root / "configs/so101_scene.json").read_text())["bank_directory"]
        command = [
            python,
            "-m",
            "operational.acceleration-20261107.train_measured",
            "--run-dir",
            str(run),
            "--bank-dir",
            str(root / bank),
            "--device",
            "cuda",
            "--envs",
            str(envs),
            "--steps",
            str(steps),
            "--iters",
            str(iters),
            "--remote-workers",
            str(worker_count),
            "--worker-init-wait",
            "900",
            "--accept-timeout",
            "900",
            "--min-conn-frac",
            "1",
            "--nthread",
            "8",
            "--bind-host",
            "0.0.0.0",
            "--trusted-network",
            "--seed",
            "20261105",
            "--eps-q",
            ".01",
            "--so101-max-contact-depth",
            ".0004",
            "--so101-stability",
            "pose",
            "--so101-safety-penalty",
            "1",
            "--save-every",
            "1",
            "--keep-every",
            "1000000",
            "--no-eval",
            "--micro-batch",
            str(microbatch),
        ]
        if name == "smoke":
            command.extend(["--epochs", "1", "--minibatches", "1"])
        (run / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        learner = launch(gpu, 8, command, root, run / "trainer.log", mode=mode)
        workers = [
            launch(
                job,
                2,
                [python, str(root / "operational/acceleration-20261107/one_worker.py"), str(run)],
                root,
                run / f"worker-{job}.log",
                1 if name == "smoke" else (3 if job == "3602602" else 5),
                mode,
            )
            for job in worker_jobs
        ]
        try:
            deadline = time.time() + 4200
            while learner.poll() is None:
                snapshot()
                assert time.time() < deadline, "Phase timeout"
                time.sleep(2)
            assert learner.returncode == 0, dict(phase=name, log=str(run / "trainer.log"))
            for worker in workers:
                assert worker.wait(timeout=90) == 0
            logs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
            assert len(logs) == iters and all(item["valid_transitions"] == envs * steps for item in logs)
            assert logs[-1]["transitions"] == iters * envs * steps
            after = snapshot()
            (run / "validated.json").write_text(
                json.dumps(dict(log=logs, jobs_before=before, jobs_after=after), indent=2) + "\n"
            )
            if name == "smoke":
                hardware = json.loads((run / "hardware.json").read_text())
                microbatch = 65536 if hardware["gpu_memory_bytes"] >= 60 * 1024**3 else 16384
                assert hardware["args"]["advertise_host"] not in ["0.0.0.0", "127.0.0.1", "localhost"]
        finally:
            for process in [learner, *workers]:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    (OUT / "complete.json").write_text(
        json.dumps(dict(complete=True, cohort=cohort, gpu=gpu, time=time.time())) + "\n"
    )
    state("COMPLETE", jobs=initial, benchmark_steps_closed=True)
except BaseException:
    state("FAILED", error=traceback.format_exc(), jobs=initial)
    raise
