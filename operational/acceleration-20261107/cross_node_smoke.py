"""Allocated cross-node benchmark smoke; setup remains immutable."""

import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path("/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107")
assert os.environ.get("SLURM_JOB_ID") == "3077705"
journal = json.loads((BASE / "artifacts_local/cluster-comparison/allocations.json").read_text())


def row(job):
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    return dict(field.split("=", 1) for field in raw.split() if "=" in field)


gpu = next(item["job"] for item in journal if item["kind"] == "gpu")
deadline = time.time() + 600
while row(gpu)["JobState"] != "RUNNING":
    assert row(gpu)["JobState"] in ("PENDING", "CONFIGURING", "REQUEUED")
    assert time.time() < deadline, "GPU did not renew within ten minutes"
    time.sleep(10)
assert not (BASE / "artifacts_local/cluster-comparison/flat/trainer.log").exists(), (
    "Full timing already began"
)
cpus = [
    item["job"] for item in journal if item["kind"] == "cpu" and row(item["job"])["JobState"] == "RUNNING"
]
assert cpus, "Need one allocated CPU job for smoke"
ROOT = BASE
RUN = BASE / "artifacts_local/cross-node-smoke"
assert not RUN.exists()
PYTHON = str(BASE / ".venv/bin/python")
ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("SLURM_", "SRUN_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"
}
ENV.update(
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MUJOCO_GL="egl",
    SO101_SURFACE_MODE="1",
)
RUN.mkdir()
learner_cmd = [
    PYTHON,
    "-m",
    "assembly_policy.train",
    "--run-dir",
    str(RUN),
    "--bank-dir",
    str(BASE / "data/so101/banks_shared_v14_mj4_surface_tree_v4"),
    "--device",
    "cuda",
    "--envs",
    "8",
    "--steps",
    "4",
    "--iters",
    "1",
    "--epochs",
    "1",
    "--minibatches",
    "1",
    "--remote-workers",
    "1",
    "--worker-init-wait",
    "60",
    "--min-conn-frac",
    "1",
    "--accept-timeout",
    "60",
    "--bind-host",
    "0.0.0.0",
    "--trusted-network",
    "--nthread",
    "8",
    "--no-eval",
    "--keep-every",
    "1000000",
    "--seed",
    "31",
    "--eps-q",
    ".01",
    "--so101-max-contact-depth",
    ".0004",
    "--so101-safety-penalty",
    "1",
]


def launch(job, cpus, cmd, log):
    return subprocess.Popen(
        ["srun", f"--jobid={job}", "--overlap", "--exact", "--ntasks=1", f"--cpus-per-task={cpus}", *cmd],
        cwd=BASE,
        env=ENV,
        stdout=open(RUN / log, "a"),
        stderr=subprocess.STDOUT,
    )


learner = worker = None
try:
    learner = launch(gpu, 8, learner_cmd, "learner.log")
    worker = launch(
        cpus[0],
        2,
        [PYTHON, str(BASE / "operational/acceleration-20261107/one_worker.py"), str(RUN)],
        "worker.log",
    )
    assert learner.wait(timeout=180) == 0
    assert worker.wait(timeout=60) == 0
    logs = [json.loads(x) for x in (RUN / "log.jsonl").read_text().splitlines()]
    assert logs[-1]["transitions"] == 32
    config = json.loads((RUN / "config.json").read_text())
    assert config["advertise_host"] not in ["0.0.0.0", "127.0.0.1", "localhost"]
    (RUN / "validated.json").write_text(
        json.dumps(
            dict(log=logs, advertise_host=config["advertise_host"], gpu=gpu, cpu=cpus[0], passed=True),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(smoke_passed=True, gpu=gpu, cpu=cpus[0], transitions=32)), flush=True)
finally:
    for process in [learner, worker]:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
