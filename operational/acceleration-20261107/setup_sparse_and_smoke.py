"""Allocated setup and cross-node smoke before the three-phase benchmark begins."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

BASE = Path("/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107")
CANDIDATE = BASE.with_name("acceleration-20261107-sparse")
assert os.environ.get("SLURM_JOB_ID") == "3077705"
assert not (BASE / "artifacts_local/cluster-comparison/flat/trainer.log").exists()
archive = BASE.with_name("acceleration-20261107-sparse-candidate.tar.gz")
assert (
    hashlib.sha256(archive.read_bytes()).hexdigest()
    == "7a0bd31ff303fa3fba2c7135d5cea4f909ffe47a74190c584cfa2bbad6ed1b79"
)
assert not CANDIDATE.exists()
shutil.copytree(
    BASE,
    CANDIDATE,
    ignore=shutil.ignore_patterns(".venv", "assets_local", "artifacts_local", "data", ".runtime-ready"),
)
(CANDIDATE / "assets_local").symlink_to(BASE / "assets_local", target_is_directory=True)
(CANDIDATE / ".venv").symlink_to(BASE / ".venv", target_is_directory=True)
with tarfile.open(archive) as tar:
    tar.extractall(CANDIDATE, filter="data")
# The third phase is declared before either timing phase starts. Only this
# isolated benchmark entry point gets the postlude; production training is untouched.
p = BASE / "assembly_policy/train.py"
before = p.read_bytes()
assert (
    hashlib.sha256(before).hexdigest() == "cf243836b60ebfa1cb7e46068f33cb17112a2381812f3ffe22b46f685ea8674c"
)
postlude = (BASE / "artifacts_local/sparse_postlude.py").read_bytes()
p.write_bytes(before + b"\n" + postlude)
(BASE / "artifacts_local/three-phase-amendment.json").write_text(
    json.dumps(
        dict(
            before=hashlib.sha256(before).hexdigest(),
            after=hashlib.sha256(p.read_bytes()).hexdigest(),
            candidate_root=str(CANDIDATE),
            declared_before_timing=True,
            time=time.time(),
        ),
        indent=2,
    )
    + "\n"
)

journal = json.loads((BASE / "artifacts_local/cluster-comparison/allocations.json").read_text())


def row(job):
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    return dict(field.split("=", 1) for field in raw.split() if "=" in field)


gpu = next(item["job"] for item in journal if item["kind"] == "gpu")
assert row(gpu)["JobState"] == "RUNNING"
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
