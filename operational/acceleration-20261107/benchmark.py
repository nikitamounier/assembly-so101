"""Matched full PPO comparison; only newly recorded benchmark allocations owned."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts_local/cluster-comparison"
OUT.mkdir(parents=True, exist_ok=True)
PYTHON = str(ROOT / ".venv/bin/python")
ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("SLURM_", "SRUN_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"
}
ENV.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl")
JOURNAL = OUT / "allocations.json"
assert os.environ.get("SLURM_JOB_ID"), "Benchmark controller must be allocated"
if JOURNAL.exists():
    # A control-allocation requeue invalidates the old partial comparison.
    old = json.loads(JOURNAL.read_text())
    for job in old:
        native = subprocess.run(
            ["scontrol", "show", "job", job["job"], "-o"], capture_output=True, text=True, check=True
        ).stdout
        assert "JobName=assembly-acceleration-" in native
        subprocess.run(["scancel", job["job"]], check=True)
    archive = OUT.with_name("cluster-comparison-retired-" + str(time.time_ns()))
    OUT.rename(archive)
    OUT.mkdir()
jobs = []


def save():
    JOURNAL.write_text(json.dumps(jobs, indent=2) + "\n")


def submit(kind, args):
    jid = (
        subprocess.check_output(
            ["sbatch", "--parsable", *args, str(ROOT / "operational/acceleration-20261107/hold.sbatch")],
            env=ENV,
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    assert jid.isdigit(), jid
    jobs.append(dict(job=jid, kind=kind))
    save()
    return jid


def row(jid):
    text = subprocess.check_output(["scontrol", "show", "job", jid, "-o"], text=True)
    return dict(x.split("=", 1) for x in text.split() if "=" in x)


def launch(jid, cpus, cmd, log, ntasks=1, mode="1"):
    return subprocess.Popen(
        [
            "srun",
            f"--jobid={jid}",
            "--overlap",
            "--exact",
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpus}",
            *cmd,
        ],
        cwd=ROOT,
        env=dict(ENV, SO101_SURFACE_MODE=mode),
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
    )


try:
    gpu = submit(
        "gpu",
        [
            "-p",
            "quick_preemptible",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--gres=gpu:1",
            "--job-name=assembly-acceleration-gpu",
            "--output=" + str(OUT / "gpu-%j.log"),
        ],
    )
    cpus = [
        submit(
            "cpu",
            [
                "-p",
                "cpu_preemptible",
                "--qos=preempt",
                "--nodes=1",
                "--ntasks=5",
                "--cpus-per-task=2",
                "--mem-per-cpu=2G",
                "--job-name=assembly-acceleration-cpu10",
                "--output=" + str(OUT / "cpu-%j.log"),
            ],
        )
        for _ in range(40)
    ]
    deadline = time.time() + 3600
    while True:
        rows = {j["job"]: row(j["job"]) for j in jobs}
        if all(v["JobState"] == "RUNNING" for v in rows.values()):
            break
        assert all(v["JobState"] in ("PENDING", "RUNNING", "CONFIGURING") for v in rows.values()), rows
        assert time.time() < deadline, "Full matched capacity did not start in one hour"
        (OUT / "capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
        time.sleep(20)
    for mode in ["0", "1"]:
        phase = OUT / ("flat" if mode == "0" else "tree")
        phase.mkdir()
        command = [
            PYTHON,
            "-m",
            "assembly_policy.train",
            "--run-dir",
            str(phase),
            "--bank-dir",
            str(ROOT / "data/so101/banks_shared_v14_mj4_surface_tree_v4"),
            "--device",
            "cuda",
            "--envs",
            "8192",
            "--steps",
            "32",
            "--iters",
            "2",
            "--remote-workers",
            "200",
            "--accept-timeout",
            "900",
            "--worker-init-wait",
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
        ]
        (phase / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        learner = launch(gpu, 8, command, phase / "trainer.log", mode=mode)
        workers = [
            launch(
                jid,
                2,
                [PYTHON, str(ROOT / "operational/acceleration-20261107/one_worker.py"), str(phase)],
                phase / f"worker-{jid}.log",
                5,
                mode,
            )
            for jid in cpus
        ]
        code = learner.wait(timeout=2400)
        for w in workers:
            try:
                w.wait(timeout=60)
            except subprocess.TimeoutExpired:
                w.terminate()
                w.wait(timeout=30)
        assert code == 0, dict(mode=mode, learner_exit=code)
        logs = [json.loads(x) for x in (phase / "log.jsonl").read_text().splitlines()]
        assert len(logs) == 2 and logs[-1]["transitions"] == 2 * 8192 * 32, logs
        (phase / "validated.json").write_text(
            json.dumps(dict(mode=mode, jobs=rows, log=logs), indent=2) + "\n"
        )
    (OUT / "complete.json").write_text(json.dumps(dict(complete=True, time=time.time())) + "\n")
finally:
    # Only IDs returned by this controller's own submissions are ever released.
    for job in jobs:
        subprocess.run(["scancel", job["job"]], check=True)
