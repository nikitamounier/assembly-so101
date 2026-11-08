"""Compare speed, detailed-point parity and contact/safety discrepancies."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).resolve().parent
baseline = json.loads((folder / "baseline-benchmark.json").read_text())
candidate = json.loads((folder / "candidate-benchmark.json").read_text())
a = np.load(folder / "baseline-replay.npz")
b = np.load(folder / "candidate-replay.npz")
assert baseline["hostname"] == candidate["hostname"], (
    "Benchmark allocation moved; rerun both on the same host"
)
initial = [k for k in a.files if "_initial_" in k]
assert baseline["point_signature"] == candidate["point_signature"]
assert baseline["point_array_sha256"] == candidate["point_array_sha256"]
for key in initial:
    assert a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes(), key
rows = []
for old in baseline["rows"]:
    new = next(
        r
        for r in candidate["rows"]
        if r["case"] == old["case"] and r["step"] == old["step"]
    )
    assert (old["task"], old["stage"]) == (new["task"], new["stage"])
    old_pairs = {tuple(sorted(c["bodies"])) for c in old["initial_contacts"]}
    new_pairs = {tuple(sorted(c["bodies"])) for c in new["initial_contacts"]}
    rows.append(
        {
            "case": old["case"],
            "task": old["task"],
            "stage": old["stage"],
            "baseline_s": old["wall_s"],
            "candidate_s": new["wall_s"],
            "speedup": old["wall_s"] / new["wall_s"],
            "old_safety": old["info"]["safety"],
            "new_safety": new["info"]["safety"],
            "initial_body_pairs_lost": sorted(old_pairs - new_pairs),
            "initial_body_pairs_added": sorted(new_pairs - old_pairs),
            "baseline_initial_deepest_m": min(
                [c["dist"] for c in old["initial_contacts"]] + [0]
            ),
            "candidate_initial_deepest_m": min(
                [c["dist"] for c in new["initial_contacts"]] + [0]
            ),
            "candidate_penetration_m": max(new["info"]["penetration"]),
            "candidate_unstable": any(new["info"]["unstable"]),
        }
    )
valid = [
    r
    for r in rows
    if not any(r["old_safety"])
    and not any(r["new_safety"])
    and not r["candidate_unstable"]
]
result = {
    "point_signature": baseline["point_signature"],
    "point_array_sha256": baseline["point_array_sha256"],
    "initial_arrays_byte_identical": len(initial),
    "cases": len(rows),
    "valid_matched_cases": len(valid),
    "valid_matched_speedup": sum(r["baseline_s"] for r in valid)
    / sum(r["candidate_s"] for r in valid)
    if valid
    else None,
    "baseline_total_s": sum(r["baseline_s"] for r in rows),
    "candidate_total_s": sum(r["candidate_s"] for r in rows),
    "paired_speedup": sum(r["baseline_s"] for r in rows)
    / sum(r["candidate_s"] for r in rows),
    "new_safety_cases": [
        r["stage"] for r in rows if any(r["new_safety"]) and not any(r["old_safety"])
    ],
    "rows": rows,
}
(folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
