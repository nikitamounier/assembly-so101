"""Third paired PPO phase inside the existing benchmark GPU step."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.with_name("acceleration-20261107")
OUT = BASE / "artifacts_local/cluster-comparison"
RUN = OUT / "sparse"
assert ROOT.name == "acceleration-20261107-sparse"
assert not RUN.exists(), "Never overwrite a sparse benchmark phase"
journal = json.loads((OUT / "allocations.json").read_text())
gpu = [row["job"] for row in journal if row["kind"] == "gpu"]
assert len(gpu) == 1
assert os.environ.get("SLURM_JOB_ID") == gpu[0], "Must run inside the owned benchmark GPU step"
cpus = [row["job"] for row in journal if row["kind"] == "cpu"]
assert len(cpus) == 40
initial = json.loads((OUT / "flat/validated.json").read_text())["jobs"]


def snapshot():
    rows = {}
    for row in journal:
        job = row["job"]
        raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
        native = dict(field.split("=", 1) for field in raw.split() if "=" in field)
        assert native["JobState"] == "RUNNING", dict(job=job, state=native["JobState"])
        for field in ["NodeList", "Restarts", "NumCPUs", "AllocTRES"]:
            assert native.get(field) == initial[job].get(field), dict(job=job, changed=field)
        rows[job] = native
    return rows


before = snapshot()
RUN.mkdir()
python = str(ROOT / ".venv/bin/python")
command = [
    python,
    "-m",
    "assembly_policy.train",
    "--run-dir",
    str(RUN),
    "--bank-dir",
    str(ROOT / "data/so101/banks_shared_v14_mj4_surface_tree_sparse"),
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
]
(RUN / "command.json").write_text(json.dumps(command, indent=2) + "\n")
clean = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("SLURM_", "SRUN_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"
}
clean.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    MUJOCO_GL="egl",
    SO101_SURFACE_MODE="1",
)
workers = []
learner = None
try:
    learner = subprocess.Popen(
        command, cwd=ROOT, stdout=open(RUN / "trainer.log", "a"), stderr=subprocess.STDOUT
    )
    for job in cpus:
        workers.append(
            subprocess.Popen(
                [
                    "srun",
                    f"--jobid={job}",
                    "--overlap",
                    "--exact",
                    "--ntasks=5",
                    "--cpus-per-task=2",
                    python,
                    str(ROOT / "operational/acceleration-20261107/one_worker.py"),
                    str(RUN),
                ],
                env=clean,
                cwd=ROOT,
                stdout=open(RUN / f"worker-{job}.log", "a"),
                stderr=subprocess.STDOUT,
            )
        )
    code = learner.wait(timeout=2400)
    assert code == 0, code
    for worker in workers:
        assert worker.wait(timeout=60) == 0
    logs = [json.loads(x) for x in (RUN / "log.jsonl").read_text().splitlines()]
    assert len(logs) == 2 and [row["transitions"] for row in logs] == [262144, 524288]
    after = snapshot()
    (RUN / "validated.json").write_text(
        json.dumps(
            dict(
                log=logs,
                jobs_before=before,
                jobs_after=after,
                scope="Same leased GPU and CPU hosts/epochs as flat/dense and tree/dense. Sparse uses a separately qualified bank with the same generation distribution/seed.",
                completed=time.time(),
            ),
            indent=2,
        )
        + "\n"
    )
finally:
    for process in [learner, *workers]:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
