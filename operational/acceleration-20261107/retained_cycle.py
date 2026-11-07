"""One allocated benchmark observation, with immutable local transport receipts."""

import argparse
import datetime
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out-dir", type=Path, required=True)
a = p.parse_args()
out = a.out_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
lock = open(out / "cycle.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
series_path = out / "series.json"
series = json.loads(series_path.read_text()) if series_path.exists() else dict(failures=0, conditions={})
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
command = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "chimera-login",
    "srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 /large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107/.venv/bin/python /large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107/operational/acceleration-20261107/observe_retained.py --manifest-sha dd95e97d3046193e2bb0e120bdf67ac4b97ab7d17d114b1de54724114835fc85",
]
try:
    r = subprocess.run(command, capture_output=True, text=True, timeout=90)
    raw = dict(command=command, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
except subprocess.TimeoutExpired as e:
    raw = dict(
        command=command,
        returncode=None,
        stdout=e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout,
        stderr=e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr,
        error="timeout; original request terminated; no retry this cycle",
    )
receipt = out / (stamp + ".json")
receipt.write_text(json.dumps(raw, indent=2) + "\n")
sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
summary = None
conditions = []
try:
    assert raw["returncode"] == 0, "observation command failed"
    summary = json.loads(raw["stdout"])
    assert summary["control_job"] == "3602602", "unexpected benchmark scope"
    assert all(state != "UNKNOWN" for state in summary["jobs"].values()), (
        "scheduler/accounting state unavailable"
    )
    series["failures"] = 0
    if summary.get("condition"):
        conditions.append(summary["condition"])
except (AssertionError, ValueError, KeyError, TypeError) as e:
    series["failures"] += 1
    if series["failures"] >= 3:
        conditions.append("benchmark_observation_failure_3cycles")
    series["classification_error"] = str(e)
active = {}
events = []
for condition in sorted(set(conditions)):
    key = series["conditions"].get(condition, "benchmark:" + condition + ":" + stamp)
    active[condition] = key
    events.append(dict(key=key, condition=condition, receipt=str(receipt), receipt_sha256=sha))
series.update(conditions=active, last_receipt=str(receipt), last_observed=time.time())
tmp = series_path.with_suffix(".tmp")
tmp.write_text(json.dumps(series, indent=2) + "\n")
tmp.replace(series_path)
print(
    json.dumps(
        dict(
            receipt=str(receipt),
            receipt_sha256=sha,
            summary=summary,
            events=events,
            observation_failure_streak=series["failures"],
        )
    ),
    flush=True,
)
