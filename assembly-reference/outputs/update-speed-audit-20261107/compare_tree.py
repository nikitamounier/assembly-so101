"""Exact replay parity and timing comparison; fail on any differing array."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--output", type=Path)
a = p.parse_args()
left, right = (
    np.load(a.baseline.with_suffix(".npz")),
    np.load(a.candidate.with_suffix(".npz")),
)
assert left.files == right.files, "Capture keys differ"
failures = []
for key in left.files:
    if (
        left[key].shape != right[key].shape
        or left[key].dtype != right[key].dtype
        or left[key].tobytes() != right[key].tobytes()
    ):
        failures.append(
            {
                "key": key,
                "baseline_shape": left[key].shape,
                "candidate_shape": right[key].shape,
            }
        )
lb = json.loads(a.baseline.with_suffix(".json").read_text())
rc = json.loads(a.candidate.with_suffix(".json").read_text())
assert lb["model_signature"] == rc["model_signature"]
rows = []
for x, y in zip(lb["rows"], rc["rows"], strict=True):
    assert (x["task"], x["stage"], x["variation"], x["step"]) == (
        y["task"],
        y["stage"],
        y["variation"],
        y["step"],
    )
    rows.append(
        {
            "task": x["task"],
            "stage": x["stage"],
            "variation": x["variation"],
            "step": x["step"],
            "baseline_s": x["wall_s"],
            "candidate_s": y["wall_s"],
            "speedup": x["wall_s"] / y["wall_s"],
        }
    )
result = {
    "exact_parity": not failures,
    "arrays_compared": len(left.files),
    "failures": failures,
    "rows": rows,
    "total_speedup": sum(r["baseline_s"] for r in rows)
    / sum(r["candidate_s"] for r in rows),
    "model_signature": lb["model_signature"],
}
if a.output:
    a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert not failures, f"{len(failures)} array mismatches"
