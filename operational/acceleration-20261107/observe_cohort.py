"""Allocated read-only observer for the fixed-cohort three-phase PPO benchmark."""

import argparse
import collections
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--manifest-sha", required=True)
a = p.parse_args()
assert os.environ.get("SLURM_JOB_ID") == "3077705"
root = Path("/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107")
out = root / "artifacts_local/available-cohort"
manifest = root / "artifacts_local/cohort-source-hashes.json"
errors = {}
if hashlib.sha256(manifest.read_bytes()).hexdigest() != a.manifest_sha:
    errors["manifest"] = "hash mismatch"
else:
    for name, expected in json.loads(manifest.read_text()).items():
        path = Path(name) if Path(name).is_absolute() else root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        if actual != expected:
            errors[name] = actual
journal = json.loads((out / "allocations.json").read_text()) if (out / "allocations.json").exists() else []
ids = ["3605855", *[row["job"] for row in journal]]
assert len(ids) == len(set(ids)) and all(x.isdigit() for x in ids)
assert not set(ids) & {"3077705", "3598717", *map(str, range(3602571, 3602611))}
active = subprocess.run(
    ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"], capture_output=True, text=True, check=True
)
assert not active.stderr.strip(), active.stderr
states = dict(line.split("|", 1) for line in active.stdout.splitlines())
missing = set(ids) - states.keys()
if missing:
    accounting = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(sorted(missing)),
            "--format=JobIDRaw,State,ExitCode,Elapsed",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not accounting.stderr.strip(), accounting.stderr
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if fields[0] in missing:
            states[fields[0]] = fields[1].split()[0].rstrip("+")
for jid in ids:
    states.setdefault(jid, "UNKNOWN")
phases = {}
for phase in ["flat-dense", "tree-dense", "tree-sparse"]:
    path = out / phase / "validated.json"
    log = out / phase / "log.jsonl"
    logs = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    valid = False
    if path.exists():
        data = json.loads(path.read_text())
        rows = data["log"]
        valid = (
            len(rows) == 2
            and rows[-1]["transitions"] == 524288
            and rows[0]["transitions"] == 262144
            and all(math.isfinite(float(row[k])) for row in rows for k in ["pi", "vf", "ent", "kl", "std"])
        )
        times_path = out / phase / "iteration-times.json"
        times = json.loads(times_path.read_text()) if times_path.exists() else []
        resources_equal = all(
            data["jobs_before"][job].get(field) == data["jobs_after"][job].get(field)
            for job in data["jobs_before"]
            for field in ("Restarts", "NodeList", "NumCPUs", "AllocTRES")
        )
        valid = valid and len(times) == 2 and resources_equal
    phases[phase] = dict(validated=valid, last_update=logs[-1] if logs else None)
terminal = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
}
closed = bool(journal) and all(states[row["job"]] in terminal for row in journal)
complete = (out / "complete.json").exists() and all(x["validated"] for x in phases.values())
control = states["3605855"]
condition = None
if errors:
    condition = "benchmark_source_integrity"
elif not journal and control == "RUNNING" and time.time() - out.stat().st_mtime > 900:
    condition = "benchmark_setup_stalled"
elif complete and closed and control == "COMPLETED":
    condition = "benchmark_complete"
elif control in terminal and not (complete and closed and control == "COMPLETED"):
    condition = "benchmark_control_terminal"
print(
    json.dumps(
        dict(
            observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_valid=not errors,
            source_errors=errors,
            control_job="3605855",
            control_state=control,
            jobs=states,
            counts=dict(collections.Counter(states[row["job"]] for row in journal)),
            phases=phases,
            complete_marker=(out / "complete.json").exists(),
            owned_resources_closed=closed,
            condition=condition,
        )
    ),
    flush=True,
)
