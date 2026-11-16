"""Supervise multiple provider pools against one unchanged learner ticket."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def validate_qualification(config_path, config):
    """Require current evidence before a main-run template can spend compute."""
    name = config.get("qualification_manifest")
    if name is None:
        return
    path = (Path(config_path).resolve().parent / name).resolve()
    receipt = json.loads(path.read_text())
    checks = (
        "physical_reset_admission", "current_cad_worker_parity",
        "full_ppo_resume", "throughput_budget", "trainer_backend",
    )
    if receipt.get("status") != "qualified" or any(
        receipt.get("checks", {}).get(key) is not True for key in checks
    ):
        raise ValueError(f"Main run is not qualified: {path}")
    artifacts = receipt.get("artifacts", [])
    if not artifacts:
        raise ValueError("Qualification requires frozen input artifacts")
    for artifact in artifacts:
        source = path.parent / artifact["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Qualified artifact changed: {source}")


def plan(config, ticket, ticket_path):
    if not ticket.get("tls_ca"):
        raise ValueError("A mixed cloud fleet requires a TLS learner ticket")
    pools = config["pools"]
    total = sum(p["hosts"] * p["workers"] for p in pools)
    if total > ticket["n_conns"]:
        raise ValueError("Pool workers exceed fixed learner connection slots")
    result = []
    provider_budgets = {"modal": 0.0, "sail": 0.0}
    for pool in pools:
        name = pool["provider"]
        if name not in ("modal", "sail", "chimera"):
            raise ValueError("Unknown provider")
        for key in ("hosts", "workers", "threads", "seconds", "memory_mib"):
            if not math.isfinite(pool[key]) or pool[key] <= 0:
                raise ValueError("Positive resource and time limits required")
        if name in ("modal", "sail"):
            provider_budgets[name] += pool["max_cost_usd"]
            if provider_budgets[name] > config["credits_usd"][name]:
                raise ValueError("Pool budget exceeds configured provider credits")
            if pool["estimated_hourly_usd"] * pool["seconds"] / 3600 > pool["max_cost_usd"]:
                raise ValueError("Pool duration exceeds its declared cost budget")
        cmd = [sys.executable, "-m", "assembly_policy.compute", name, "--ticket", str(ticket_path)]
        for key in ("hosts", "workers", "threads", "seconds", "memory_mib"):
            cmd += ["--" + key.replace("_", "-"), str(pool[key])]
        for key in ("region", "cpus", "sail_size", "run_dir", "partition", "qos"):
            if key in pool:
                cmd += ["--" + key.replace("_", "-"), str(pool[key])]
        result.append(dict(provider=name, command=cmd, workers=pool["hosts"] * pool["workers"]))
    return result


def launch(config_path, ticket_path, dry_run=False):
    config = json.loads(Path(config_path).read_text())
    ticket = json.loads(Path(ticket_path).read_text())
    rows = plan(config, ticket, ticket_path)
    print(
        json.dumps(
            dict(pools=rows, scope="Cost declarations are estimates; provider billing is authoritative")
        ),
        flush=True,
    )
    if dry_run:
        return
    if not config.get("enabled"):
        raise ValueError("Register and enable the qualified new run before launching")
    validate_qualification(config_path, config)
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for row in rows:
            children.append(subprocess.Popen(row["command"], start_new_session=True))
        while children and not stopping:
            for child in children:
                code = child.poll()
                if code not in (None, 0):
                    raise RuntimeError(f"Provider pool failed with exit code {code}")
            if all(child.poll() is not None for child in children):
                break
            time.sleep(0.5)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Cloud max_lifetime/function timeout still bounds resources.
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)
