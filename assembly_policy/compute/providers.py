"""Cloud API calls happen at pool startup, never on the rollout hot path."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .bundle import BUILD, ROOT, build_bundle
from .pool import run_workers


def modal_pool(args, ticket):
    import modal

    with tempfile.TemporaryDirectory(prefix="so101-modal-") as tmp:
        bundle = Path(tmp) / "worker.tar.gz"
        identity = build_bundle(bundle)
        print(json.dumps({"provider": "modal", "image_input": identity}), flush=True)
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("g++", "libgl1")
            .add_local_file(str(bundle), "/tmp/worker.tar.gz", copy=True)
            .run_commands(BUILD)
        )
        app = modal.App("assembly-so101-cpu")
        worker = app.function(
            image=image,
            cpu=args.cpus,
            memory=args.memory_mib,
            region=args.region,
            timeout=int(args.seconds) + 60,
            max_containers=args.hosts,
            serialized=True,
            include_source=False,
        )(run_workers)
        # An attached app owns exactly this pool; leaving the context releases it.
        with app.run():
            calls = []
            try:
                for _ in range(args.hosts):
                    calls.append(worker.spawn(ticket, args.workers, args.threads, args.seconds))
                for call in calls:
                    print(json.dumps(call.get()), flush=True)
            finally:
                for call in calls:
                    call.cancel()


def sail_pool(args, ticket):
    import sail

    app = sail.App.find(name="assembly-so101", mint_if_missing=True)
    boxes = []
    with tempfile.TemporaryDirectory(prefix="so101-sail-") as tmp:
        bundle = Path(tmp) / "worker.tar.gz"
        identity = build_bundle(bundle)
        print(json.dumps({"provider": "sail", "image_input": identity}), flush=True)
        image = (
            sail.Image.debian("amd64")
            .apt_install("g++", "libgl1")
            .add_local_file(str(bundle), "/tmp/worker.tar.gz")
            .run_commands(BUILD)
            .build()
        )
        try:
            # Capture each created handle immediately so partial startup is cleaned up.
            for _ in range(args.hosts):
                kwargs = dict(
                    app=app,
                    name="so101-" + uuid.uuid4().hex[:10],
                    image=image,
                    size=args.sail_size,
                    memory_limit_gib=(args.memory_mib + 1023) // 1024,
                    visibility="private",
                    auto_sleep=sail.AutoSleep.never(),
                    max_lifetime_seconds=int(args.seconds) + 600,
                )
                if args.region:
                    kwargs["region"] = args.region
                box = sail.Sailbox.create(**kwargs)
                boxes.append(box)
            # Ticket upload is private and happens once; source/runtime are image-cached.
            ticket_path = Path(tmp) / "ticket.json"
            ticket_path.write_text(json.dumps(ticket))
            ticket_path.chmod(0o600)
            processes = []
            for box in boxes:
                box.fs.upload_file(ticket_path, "/tmp/so101-ticket.json")
                command = [
                    "python",
                    "-m",
                    "assembly_policy.compute",
                    "local",
                    "--ticket",
                    "/tmp/so101-ticket.json",
                    "--workers",
                    str(args.workers),
                    "--threads",
                    str(args.threads),
                    "--seconds",
                    str(args.seconds),
                ]
                processes.append(box.exec(command, cwd="/opt/assembly", timeout=args.seconds + 60))

            # Drain output concurrently; one full buffer must not stall another host.
            def finish(proc):
                for chunk in proc.output:
                    print(chunk, end="", flush=True)
                result = proc.wait()
                if result.exit_code:
                    raise RuntimeError(f"Sail worker exited {result.exit_code}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.hosts) as executor:
                list(executor.map(finish, processes))
        finally:
            errors = []
            for box in boxes:
                try:
                    box.terminate()
                except Exception as error:
                    errors.append(f"{box.sailbox_id}: {error}")
            if errors:
                raise RuntimeError("Sail cleanup needs attention: " + "; ".join(errors))


def chimera_pool(args, ticket):
    if args.run_dir is None:
        raise ValueError("Chimera requires --run-dir on the cluster's shared filesystem")
    # This command runs inside a CPU allocation; it never computes on a login node.
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the Chimera launcher inside an allocated CPU step")
    run = args.run_dir.resolve()
    if args.ticket.resolve() != run / "server.json":
        raise ValueError("Chimera --ticket must be server.json in the shared --run-dir")
    run.mkdir(parents=True, exist_ok=True)
    jobs = []
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SBATCH_"))}
    env.update(
        SO101_ROOT=str(ROOT), SO101_RUN=str(run), SO101_THREADS=str(args.threads), SO101_PYTHON=sys.executable
    )
    try:
        for _ in range(args.hosts):
            command = [
                "sbatch",
                "--parsable",
                "--job-name=assembly-so101-cpu",
                "--partition=" + args.partition,
                "--qos=" + args.qos,
                "--nodes=1",
                f"--ntasks={args.workers}",
                f"--cpus-per-task={args.threads}",
                f"--mem={args.memory_mib}M",
                "--time=02:00:00",
                "--requeue",
                "--signal=B:USR1@120",
                "--output=" + str(run / "workers-%j.log"),
                str(ROOT / "scripts/compute/worker.sbatch"),
            ]
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            job = result.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError("Unexpected sbatch response")
            jobs.append(job)
            (run / "cpu-pool.json").write_text(
                json.dumps({"jobs": jobs, "workers_per_job": args.workers}) + "\n"
            )
        print(json.dumps({"provider": "chimera", "jobs": jobs}), flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["squeue", "--noheader", "--jobs=" + ",".join(jobs), "--format=%i"],
                capture_output=True,
                text=True,
                check=True,
            )
            if not result.stdout.strip():
                break
            time.sleep(5)
    finally:
        if jobs:
            subprocess.run(["scancel", *jobs], check=True)
