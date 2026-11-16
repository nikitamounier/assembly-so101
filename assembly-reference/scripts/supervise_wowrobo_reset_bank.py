"""Durably finish an authorized reset bank, retry gaps, then publish review assets.

Does not alter training, relax admission, change noise or launch cluster jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from assembly_policy.so101.wowrobo_bank import write_json

ROOT = Path(__file__).resolve().parents[1]


def active(plan):
    with (plan / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def progress(plan):
    manifest = json.loads((plan / "plan.json").read_text())
    total = 0
    stages = {}
    for stage in manifest["stage_ids"]:
        accepted, attempted, reasons = {"train": 0, "eval": 0}, 0, {}
        # Receipts are atomic; count actual accepted rows even before a worker's
        # next progress export. Never infer progress from a running process.
        for path in (plan / "shards" / stage / "attempts").glob("*.json"):
            row = json.loads(path.read_text())
            attempted += 1
            if row["training_eligible"]:
                accepted[row["split"]] += 1
            for reason in row["rejection_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        stages[stage] = dict(accepted, attempts=attempted, rejection_reasons=reasons)
        total += sum(accepted.values())
    result = {
        "accepted": total,
        "target": manifest["target_total"],
        "by_stage": stages,
        "updated_unix": time.time(),
        "status": "generating",
        "coordinator_active": active(plan),
    }
    write_json(plan / "live-progress.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--port", type=int, default=8878)
    a = p.parse_args()
    plan, review = a.plan.resolve(), a.review.resolve()
    with (plan / ".supervisor.lock").open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            plan / "supervisor.json",
            {
                "pid": os.getpid(),
                "status": "running",
                "workers": a.workers,
                "review": str(review),
                "port": a.port,
            },
        )
        # An initial foreground coordinator may already own the plan. Adopt its
        # durable outputs; never race it with a second writer.
        while active(plan):
            progress(plan)
            time.sleep(30)
        for limit in [20, 40, 80, 160]:
            current = progress(plan)
            if (
                current["accepted"] == current["target"]
                and (plan / "bank/manifest.json").exists()
            ):
                break
            runs = list((plan / "shards").glob("*/run.json"))
            if any(
                json.loads(path.read_text())["status"] == "source_changed"
                for path in runs
            ):
                raise RuntimeError(
                    "Frozen generation source changed; repair requires a new versioned plan"
                )
            command = [
                sys.executable,
                str(ROOT / "scripts/build_wowrobo_reset_framework.py"),
                "run",
                "--output",
                str(plan),
                "--workers",
                str(a.workers),
                "--max-attempts",
                str(limit),
            ]
            with (plan / f"retry-{limit}.log").open("a") as log:
                child = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                while child.poll() is None:
                    progress(plan)
                    time.sleep(30)
        current = progress(plan)
        if (
            current["accepted"] != current["target"]
            or not (plan / "bank/manifest.json").exists()
        ):
            write_json(
                plan / "supervisor.json",
                {
                    "status": "incomplete_after_bounded_retries",
                    "progress": current,
                    "review_published": False,
                },
            )
            return 2
        final = json.loads((plan / "bank/manifest.json").read_text())
        if not final["complete"]:
            raise RuntimeError("Bank is not sealed")
        manifest = json.loads((plan / "plan.json").read_text())
        config = json.loads(Path(manifest["config"]).read_text())
        command = [
            sys.executable,
            str(ROOT / "scripts/publish_wowrobo_variation_preview.py"),
            "--source",
            config["nominal_bank"],
            "--shards",
            str(plan / "shards"),
            "--output",
            str(review),
            "--stages",
            *manifest["stage_ids"],
        ]
        with (plan / "publication.log").open("a") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        # Adapt labels after the common publisher has exported admitted rows.
        index = json.loads((review / "stages/manifest.json").read_text())
        index["display_name"] = "WowRobo · 1,800 validated variations"
        index["progress"].update(
            status="complete",
            complete=True,
            preview_only=False,
            requested_component="All five assembly tasks",
        )
        write_json(review / "stages/manifest.json", index)
        write_json(review / "progress.json", index["progress"])
        environment = dict(os.environ, PYTHONPATH=str(ROOT.parent / "assembly-so101"))
        command = [
            sys.executable,
            "-m",
            "assembly_policy.viewer",
            "--scene",
            "wowrobo-layout",
            "--layout",
            str(review),
            "--port",
            str(a.port),
        ]
        with (plan / "viewer.log").open("a") as log:
            viewer = subprocess.Popen(
                command,
                cwd=ROOT.parent,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(
            plan / "supervisor.json",
            {
                "status": "complete",
                "accepted": current["accepted"],
                "review": str(review),
                "viewer_pid": viewer.pid,
                "port": a.port,
                "viewer_url": f"http://localhost:{a.port}/",
                "training_bank": str(plan / "bank"),
                "review_published": True,
            },
        )
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--plan" in sys.argv:
            folder = Path(sys.argv[sys.argv.index("--plan") + 1])
            write_json(
                folder / "supervisor.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "pid": os.getpid(),
                    "updated_unix": time.time(),
                    "review_published": False,
                },
            )
        raise
