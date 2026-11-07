"""One no-argument, allocated cube observation with durable independent receipts."""

import datetime
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[2]
OUT = LOCAL_ROOT / "artifacts_local/cube_trial/chimera/monitor"
REMOTE = "/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6"
PINS = Path(__file__).with_name("monitor-pins.json")


def main():
    if sys.argv[1:]:
        raise SystemExit("This fixed observer takes no arguments")
    pins = json.loads(PINS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "cycle.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / "series.json"
    series = json.loads(path.read_text()) if path.exists() else dict(failures=0, conditions={})
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    script = REMOTE + "/operational/cube-trial-20261107/cluster.py"
    guarded = (
        f"cube_observer_hash=$(sha256sum {shlex.quote(script)}); "
        f"[ \"${{cube_observer_hash%% *}}\" = {shlex.quote(pins['observer_sha256'])} ] || exit 42; "
        f"exec {REMOTE}/.venv/bin/python {shlex.quote(script)} observe"
    )
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "chimera-login",
               shlex.join(["srun", "--jobid=3077705", "--overlap", "--exact", "--ntasks=1",
                           "--cpus-per-task=1", "bash", "-lc", guarded])]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        raw = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        raw = dict(command=command, returncode=None, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or ""),
                   error="Observation timed out; original local invocation terminated; no retry this cycle")
    now = time.time()
    summary = None
    conditions = []
    try:
        if raw["returncode"] == 42:
            conditions.append("cube_observer_source_integrity")
        if raw["returncode"] != 0:
            raise ValueError("Allocated cube observation unavailable")
        full = json.loads(raw["stdout"])
        assert full["allocated_monitor_job"] == "3077705"
        series["failures"] = 0
        if series.get("run") != full["run"]:
            # A fresh scientific run starts new progress timers, preserving alert/delivery history.
            series["run"] = full["run"]
            series["startup_since"] = now
            series["stale_count"] = 0
            series.pop("waiting_since", None)
        if full["source_manifest_sha256"] != pins["source_manifest_sha256"] or not full["source_valid"]:
            conditions.append("cube_source_integrity")
        conditions.extend("cube_" + error for error in full.get("errors", []))
        state = full.get("controller", {})
        phase = state.get("phase")
        latest = full.get("last_update")
        for job in state.get("terminal_jobs", []):
            conditions.append("cube_allocation_terminal:" + job)
        if phase == "complete" and latest and latest.get("it", 0) >= 10000 and full["checkpoint_bytes"] > 0:
            conditions.append("cube_training_complete")
        if phase in ("running", "learner_starting"):
            series.setdefault("startup_since", now)
            if latest and full.get("progress_age_s", 0) > 1200:
                series["stale_count"] = series.get("stale_count", 0) + 1
                if series["stale_count"] >= 3:
                    conditions.append("cube_progress_stall")
            else:
                series["stale_count"] = 0
            if latest is None and now - series["startup_since"] > 1800:
                conditions.append("cube_first_update_missing_30min")
        else:
            series.pop("startup_since", None)
            series["stale_count"] = 0
        if phase == "waiting_allocations":
            series.setdefault("waiting_since", now)
            if now - series["waiting_since"] > 1800:
                conditions.append("cube_capacity_wait_30min")
        else:
            series.pop("waiting_since", None)
        running = [row for row in state.get("jobs", {}).values() if row.get("JobState") == "RUNNING"]
        summary = dict(
            run=full["run"], phase=phase, source_valid=full["source_valid"], source_errors=full["source_errors"],
            last_update={k: latest.get(k) for k in ("it", "transitions", "valid_transitions", "target_transitions",
                                                    "fps", "success", "fam_success", "t_sim", "t_update")} if latest else None,
            progress_age_s=full.get("progress_age_s"), checkpoint_bytes=full["checkpoint_bytes"],
            running_allocations=len(running), target_cpu_jobs=len(state.get("cpu_jobs", [])),
            running_cpu_jobs=sum(state.get("jobs", {}).get(j, {}).get("JobState") == "RUNNING" for j in state.get("cpu_jobs", [])),
            worker_steps_running=sum(v.get("running", False) for v in state.get("worker_steps", {}).values()),
            physics=full["physics"], errors=full.get("errors", []),
        )
    except (ValueError, KeyError, TypeError, AssertionError) as exc:
        series["failures"] = series.get("failures", 0) + 1
        if series["failures"] >= 3:
            conditions.append("cube_observation_failure_3cycles")
        raw["classification_error"] = str(exc)
    receipt = OUT / (stamp + ".json")
    receipt.write_text(json.dumps(raw, indent=2) + "\n")
    sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    active = {}
    events = []
    for condition in sorted(set(conditions)):
        key = series.get("conditions", {}).get(condition, condition + ":" + stamp)
        active[condition] = key
        events.append(dict(key=key, condition=condition, receipt=str(receipt), receipt_sha256=sha))
    series.update(conditions=active, last_receipt=str(receipt))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(series, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(dict(receipt=str(receipt), receipt_sha256=sha, summary=summary,
                          observation_failure_streak=series["failures"], events=events), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
