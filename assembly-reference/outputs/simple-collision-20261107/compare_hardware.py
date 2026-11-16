"""Compare the conservative fallback on its own paired host and reset panel."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).resolve().parent
old = json.loads((folder / "baseline-hardware-benchmark.json").read_text())
new = json.loads((folder / "candidate-hardware-benchmark.json").read_text())
a = np.load(folder / "baseline-hardware-replay.npz")
b = np.load(folder / "candidate-hardware-replay.npz")
assert old["hostname"] == new["hostname"]
assert old["cpu_affinity"] == new["cpu_affinity"]
assert old["point_signature"] == new["point_signature"]
assert old["point_array_sha256"] == new["point_array_sha256"]
assert set(a.files) == set(b.files)
arrays = []
for key in a.files:
    arrays.append(
        {
            "name": key,
            "byte_identical": a[key].shape == b[key].shape
            and a[key].tobytes() == b[key].tobytes(),
            "max_abs_error": float(np.max(np.abs(a[key] - b[key])))
            if a[key].size
            else 0.0,
        }
    )
rows = []
assert len(old["rows"]) == len(new["rows"])
for x, y in zip(old["rows"], new["rows"], strict=True):
    assert (x["case"], x["step"], x["stage"]) == (y["case"], y["step"], y["stage"])
    rows.append(
        {
            "case": x["case"],
            "stage": x["stage"],
            "baseline_s": x["wall_s"],
            "candidate_s": y["wall_s"],
            "speedup": x["wall_s"] / y["wall_s"],
            "initial_contacts_identical": x["initial_contacts"]
            == y["initial_contacts"],
            "final_contacts_identical": x["contacts"] == y["contacts"],
            "episode_results_identical": all(
                x[k] == y[k] for k in ("reward", "done", "info")
            ),
            "safety": any(y["info"]["safety"]),
            "unstable": any(y["info"]["unstable"]),
        }
    )
base_s = sum(r["baseline_s"] for r in rows)
new_s = sum(r["candidate_s"] for r in rows)
body_position_error = max(
    float(np.linalg.norm(a[k] - b[k], axis=-1).max())
    for k in a.files
    if k.endswith("_bodypos")
)
quat_dot = [
    (
        np.abs(np.sum(a[k] * b[k], axis=-1))
        / (np.linalg.norm(a[k], axis=-1) * np.linalg.norm(b[k], axis=-1))
    ).clip(0, 1)
    for k in a.files
    if k.endswith("_bodyquat")
]
body_rotation_error_degrees = max(
    float(np.rad2deg(2 * np.arccos(x)).max()) for x in quat_dot
)
result = {
    "maximum_body_position_error_m": body_position_error,
    "maximum_body_rotation_error_degrees": body_rotation_error_degrees,
    "hostname": old["hostname"],
    "cpu_affinity": old["cpu_affinity"],
    "cases": len(rows),
    "arrays": len(arrays),
    "byte_identical_arrays": sum(r["byte_identical"] for r in arrays),
    "initial_contacts_identical": all(r["initial_contacts_identical"] for r in rows),
    "final_contacts_identical": all(r["final_contacts_identical"] for r in rows),
    "episode_results_identical": all(r["episode_results_identical"] for r in rows),
    "safety_cases": sum(r["safety"] for r in rows),
    "unstable_cases": sum(r["unstable"] for r in rows),
    "baseline_total_s": base_s,
    "candidate_total_s": new_s,
    "paired_speedup": base_s / new_s,
    "time_reduction_fraction": 1 - new_s / base_s,
    "scope": "Single-world, one 40ms zero-action interval per task-conditioned prototype. No worker scaling, whole-PPO or long-episode qualification.",
    "rows": rows,
    "array_comparisons": arrays,
}
(folder / "hardware-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("rows", "array_comparisons")},
        indent=2,
    )
)
