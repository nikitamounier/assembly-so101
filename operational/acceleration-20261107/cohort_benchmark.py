"""Full PPO comparison on a fixed available cohort; no original training jobs used."""

import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path("/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107")
SPARSE = BASE.with_name("acceleration-20261107-sparse")
OUT = BASE / "artifacts_local/available-cohort"
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


if (OUT / "allocations.json").exists():
    previous = json.loads((OUT / "allocations.json").read_text())
    for item in previous:
        native = row(item["job"])
        assert native["JobName"].startswith("assembly-cohort-")
        subprocess.run(["scancel", item["job"]], check=True)
if OUT.exists():
    OUT.rename(OUT.with_name("available-cohort-retired-" + str(time.time_ns())))
OUT.mkdir(parents=True)
# Camera-only serialization is canonical across ARM64/x86; regenerate actual
# starts/references under each Linux model instead of changing old bank labels.
python = str(BASE / ".venv/bin/python")
subprocess.run(
    [
        python,
        str(BASE / "operational/acceleration-20261107/portable_cameras.py"),
        str(BASE / "assembly_policy/so101/workcell.py"),
        str(SPARSE / "assembly_policy/so101/workcell.py"),
    ],
    check=True,
)
for root, tag in [(BASE, "tree_portable"), (SPARSE, "tree_sparse_portable")]:
    config = root / "configs/so101_scene.json"
    cfg = json.loads(config.read_text())
    cfg["bank_directory"] = "data/so101/banks_shared_v14_mj4_" + tag
    cfg["reference_prefix"] = "data/so101/shared_v14_mj4_" + tag + "_reference"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    with (OUT / ("prepare-" + tag + ".log")).open("a") as log:
        subprocess.run(
            [python, "-m", "scripts.prepare_so101_workcell", "--count", "32", "--workers", "2"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    identity = subprocess.check_output(
        [python, "-c", "from assembly_policy.so101.resets import signature; print(signature())"],
        cwd=root,
        text=True,
    ).strip()
    if root == SPARSE:
        assert identity == "bc0bef2686ce0eb6b7d22cc0caf27b28abad51010933a57f91bc0e0d00125104", identity
jobs = []


def submit(kind, flags):
    jid = (
        subprocess.check_output(
            ["sbatch", "--parsable", *flags, str(BASE / "operational/acceleration-20261107/hold.sbatch")],
            env=ENV,
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    assert jid.isdigit()
    jobs.append(dict(job=jid, kind=kind))
    (OUT / "allocations.json").write_text(json.dumps(jobs, indent=2) + "\n")
    return jid


def launch(job, cpus, command, root, log, ntasks=1, mode="1"):
    return subprocess.Popen(
        [
            "srun",
            f"--jobid={job}",
            "--overlap",
            "--exact",
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpus}",
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
            "--job-name=assembly-cohort-gpu",
            "--output=" + str(OUT / "gpu-%j.log"),
        ],
    )
    requested = [
        submit(
            "cpu",
            [
                "-p",
                "cpu_preemptible",
                "--qos=preempt",
                "--nodes=1",
                "--ntasks=5",
                "--cpus-per-task=2",
                "--mem-per-cpu=4G",
                "--job-name=assembly-cohort-cpu10",
                "--output=" + str(OUT / "cpu-%j.log"),
            ],
        )
        for _ in range(4)
    ]
    start = time.time()
    while True:
        native = {job: row(job) for job in [gpu, *requested]}
        assert native[gpu]["JobState"] in ("PENDING", "RUNNING", "CONFIGURING", "REQUEUED"), native[gpu]
        # A CPU job lost before cohort admission is not part of the timed
        # experiment. Select a fixed live cohort from the remaining jobs.
        assert any(
            native[job]["JobState"] in ("PENDING", "RUNNING", "CONFIGURING", "REQUEUED") for job in requested
        ), native
        ready = [job for job in requested if native[job]["JobState"] == "RUNNING"]
        (OUT / "capacity.json").write_text(json.dumps(native, indent=2) + "\n")
        if (
            native[gpu]["JobState"] == "RUNNING"
            and len(ready) >= 2
            and (len(ready) == 4 or time.time() - start > 120)
        ):
            cohort = ready
            break
        assert time.time() - start < 1200, (
            "GPU plus two CPU jobs did not become available within twenty minutes"
        )
        time.sleep(10)
    for job in requested:
        if job not in cohort:
            subprocess.run(["scancel", job], check=True)
    initial = snapshot()
    assert all(int(initial[job]["NumCPUs"]) == 10 for job in cohort)
    conns = 5 * len(cohort)
    (OUT / "cohort.json").write_text(
        json.dumps(
            dict(
                control_job=os.environ["SLURM_JOB_ID"],
                gpu=gpu,
                cpu_jobs=cohort,
                simulator_cpus=10 * len(cohort),
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
                1 if name == "smoke" else 5,
                mode,
            )
            for job in worker_jobs
        ]
        try:
            assert learner.wait(timeout=4200) == 0, dict(phase=name, log=str(run / "trainer.log"))
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
finally:
    for item in jobs:
        subprocess.run(["scancel", item["job"]], check=True)
