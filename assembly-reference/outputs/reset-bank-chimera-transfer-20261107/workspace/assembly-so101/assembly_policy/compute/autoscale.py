"""Allocated, non-preempting Slurm projections and opt-in additive worker scaling.

Only a separately registered new run may scale. Existing experiments are never
discovered as submission targets. A durable intent prevents duplicate submission
after ambiguous sbatch outcomes. Invoke at heartbeat cadence; sweeps are throttled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

GRID = [1, *range(10, 1001, 10)]


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def command(argv, *, env=None, timeout=20):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def starts_within(row, now, seconds=180):
    # Parse on the cluster, in the same timezone as the Slurm CLI.
    m = re.search(r"to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", row["stdout"] + row["stderr"])
    if row["returncode"] or not m:
        return False
    delay = dt.datetime.fromisoformat(m[1]).timestamp() - now
    return -60 <= delay <= seconds


def shape(cpus, exclude):
    args = [
        "--partition=cpu_preemptible",
        "--qos=preempt",
        "--time=02:00:00",
        "--mem-per-cpu=2G",
        "--requeue",
    ]
    # Same 5-process / 2-thread shape as the portable ten-CPU worker job.
    args += [f"--ntasks={max(1, cpus // 2)}", f"--cpus-per-task={min(cpus, 2)}"]
    if exclude:
        args += ["--exclude=" + exclude]
    return args


def additional_jobs(config, queue, ledger):
    """Count pending capacity too; never fill more than the fixed learner slots."""
    if not config.get("enabled") or "uncertain_submission" in ledger:
        return 0
    active = [j for j in queue if j["id"] in set(config["base_job_ids"]) | set(ledger["jobs"])]
    if any(j["state"] not in ("RUNNING", "PENDING", "CONFIGURING") for j in active):
        return 0
    cpus = sum(j["cpus"] for j in active)
    # Cloud processes reserve slots even while starting; caller updates this field.
    cpus_per_worker = config.get("cpus_per_worker", 2)
    if cpus_per_worker not in (1, 2, 5, 10):
        raise ValueError("Worker CPU shape must divide the ten-CPU job")
    free_workers = config["worker_slots"] - config.get("cloud_workers", 0) - cpus // cpus_per_worker
    return max(
        0,
        min(
            (config["max_chimera_cpus"] - cpus) // 10,
            free_workers // (10 // cpus_per_worker), config.get("max_jobs_per_cycle", 10)
        ),
    )


def cycle(directory):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside an allocated CPU step")
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "cycle.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    now = time.time()
    latest = directory / "latest.json"
    if latest.exists() and now - latest.stat().st_mtime < 300:
        return dict(action="keep_monitor", cached=True, receipt=str(latest))
    q = command(["squeue", "--noheader", "--format=%i|%T|%C|%N", "--user=" + os.environ["USER"]])
    if q["returncode"]:
        raise RuntimeError("Cannot obtain current owned jobs: " + q["stderr"])
    queue = []
    for line in q["stdout"].splitlines():
        job, state, cpus, hosts = line.strip().split("|", 3)
        queue.append(dict(id=job, state=state, cpus=int(cpus), nodes=hosts))
    # Conservative: exclude every host with ANY running job owned by this user,
    # including unrelated research, the learner, and the monitor allocation.
    exclude = ",".join(sorted({j["nodes"] for j in queue if j["state"] == "RUNNING" and j["nodes"]}))
    part = command(["scontrol", "show", "partition", "cpu_preemptible", "-o"])
    match = re.search(r"\bTotalNodes=(\d+)", part["stdout"])
    if part["returncode"] or not match:
        raise RuntimeError("Cannot determine partition node count")
    total_nodes = int(match[1])
    rows = []
    for cpus in GRID:
        row = command(["sbatch", "--test-only", *shape(cpus, exclude), "--wrap=true"])
        row.update(cpus=cpus, near=starts_within(row, time.time()))
        rows.append(row)
    # Also retain the literal requested node sweep. Impossible shapes are explicitly
    # classified without sending pointless requests to the scheduler.
    nodes = []
    for count in GRID:
        if count > total_nodes:
            nodes.append(dict(nodes=count, status="exceeds_partition_node_count"))
        else:
            args = [
                "sbatch",
                "--test-only",
                "--partition=cpu_preemptible",
                "--qos=preempt",
                "--time=02:00:00",
                f"--nodes={count}",
                "--ntasks-per-node=5",
                "--cpus-per-task=2",
                "--mem-per-cpu=2G",
                "--requeue",
            ]
            if exclude:
                args += ["--exclude=" + exclude]
            nodes.append(dict(nodes=count, **command([*args, "--wrap=true"])))
    record = dict(
        timestamp=now,
        allocation=os.environ["SLURM_JOB_ID"],
        excluded_hosts=exclude,
        physical_partition_nodes=total_nodes,
        cpu_projections=rows,
        node_projections=nodes,
        action="observe_only_no_registered_run",
        submitted=[],
    )
    cfgpath = directory / "active-run.json"
    ledgerpath = directory / "submissions.json"
    ledger = json.loads(ledgerpath.read_text()) if ledgerpath.exists() else dict(jobs=[])
    if cfgpath.exists():
        config = json.loads(cfgpath.read_text())
        count = additional_jobs(config, queue, ledger)
        record["action"] = "keep_monitor"
        if count:
            run = Path(config["run_dir"])
            # Bind activation to frozen source, the actual listener, and fresh SGD.
            script = Path(config["worker_script"])
            if hashlib.sha256(script.read_bytes()).hexdigest() != config["worker_script_sha256"]:
                raise RuntimeError("Worker script identity changed")
            ticket = json.loads((run / "server.json").read_text())
            if ticket["n_conns"] != config["worker_slots"]:
                raise RuntimeError("Wrong learner ticket")
            log = run / "log.jsonl"
            if now - log.stat().st_mtime > config.get("progress_fresh_s", 180):
                raise RuntimeError("Learner has no fresh progress; refuse scale-up")
            records = log.read_text().splitlines()
            if not records or json.loads(records[-1]).get("it", 0) < 1:
                raise RuntimeError("Learner has no committed PPO update; refuse scale-up")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SBATCH_"))}
            env.update(
                SO101_ROOT=config["root"],
                SO101_RUN=str(run),
                SO101_THREADS="2",
                SO101_PYTHON=config["python"],
            )
            # Aggregate projection is advisory. Check the EXACT single-node job
            # again immediately before EACH submission and refresh exclusion hosts.
            near = [r["cpus"] for r in rows if r["near"] and r["cpus"] >= 10]
            count = min(count, max(near, default=0) // 10)
            for _ in range(count):
                fresh = command(
                    [
                        "squeue",
                        "--user=" + os.environ["USER"],
                        "--states=RUNNING",
                        "--noheader",
                        "--format=%N",
                    ]
                )
                if fresh["returncode"]:
                    raise RuntimeError("Cannot refresh protected hosts")
                protected = ",".join(sorted(set(fresh["stdout"].split())))
                args = [
                    *shape(10, protected),
                    "--nodes=1",
                    "--signal=B:USR1@120",
                    "--job-name=assembly-main-cpu",
                    "--output=" + str(run / "workers-%j.log"),
                ]
                probe = command(["sbatch", "--test-only", *args, str(script)], env=env)
                if not starts_within(probe, time.time()):
                    break
                # Persist BEFORE sending: timeout/crash never blindly resubmits.
                ledger["uncertain_submission"] = dict(timestamp=time.time(), argv=args)
                atomic(ledgerpath, ledger)
                result = command(["sbatch", "--parsable", *args, str(script)], env=env)
                if result["returncode"]:
                    record["submission_error"] = result
                    break
                job = result["stdout"].strip().split(";")[0]
                if not job.isdigit():
                    raise RuntimeError("Unrecognized sbatch result; reconcile before retry")
                ledger["jobs"].append(job)
                ledger.pop("uncertain_submission")
                atomic(ledgerpath, ledger)
                record["submitted"].append(job)
            if record["submitted"]:
                record["action"] = "scaled_up"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic(directory / (stamp + ".json"), record)
    atomic(latest, record)
    return dict(
        action=record["action"],
        submitted=record["submitted"],
        receipt=str(latest),
        near_cpu_shapes=[r["cpus"] for r in rows if r["near"]],
        excluded_hosts=exclude,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(cycle(args.state_dir)), flush=True)


if __name__ == "__main__":
    main()
