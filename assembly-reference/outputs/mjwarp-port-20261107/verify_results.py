"""Validate saved final-source H100 evidence without a GPU or model load."""

import hashlib
import json
from pathlib import Path

import numpy as np

p = Path(__file__).resolve().parent
records = json.loads((p / "results.json").read_text())
backend = hashlib.sha256(
    (p.parents[1] / "assembly_policy/so101/wowrobo_warp.py").read_bytes()
).hexdigest()
runner = hashlib.sha256((p / "runner.py").read_bytes()).hexdigest()
signature = "8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879"
assert len(records) == 12
peak = pos = rot = friction = 0.0
matched = 0
for name, r in records.items():
    assert r["complete"] and not r.get("error"), name
    assert r["backend_sha256"] == backend and r["runner_sha256"] == runner, name
    assert r["native_signature"] == r["expected_signature"] == signature
    assert r["bank_version"] == 6
    if r["mode"] == "query":
        q = r["query"]
        assert q["count"] == 3726 and q["finite"]
        assert q["max_distance_error_m"] == q["max_gradient_error"] == 0
        continue
    assert r["friction_max_error"] <= 1e-6
    assert abs(r["sensor_min"] - r["contact_min"]) <= 1e-12
    for s in r["supplement_panel"]:
        assert s["native"] == s["gpu"] == s["matched_emitted"], (name, s)
        assert s["max_dist_error"] == 0
        assert s["max_position_error"] <= 1e-6
        assert s["max_normal_error"] <= 1e-5
    a = r["assembly"]
    assert a["finite"] and not any(a["native_warnings"])
    peak = max(peak, *a["gpu_peak_penetration_m"], a["native_peak_penetration_m"])
    pos = max(pos, r["free_translation_error_m"])
    rot = max(rot, r["free_rotation_error_degrees"])
    assert peak < 0.0005 and pos < 0.0002 and rot < 0.1
    with np.load(p / "results" / name / "initial-contacts.npz") as z:
        for pair, coefficients in zip(z["gpu_geom"], z["gpu_friction"]):
            rows = np.flatnonzero(np.all(z["native_geom"] == pair, axis=1))
            if rows.size:
                friction = max(
                    friction,
                    float(
                        np.min(
                            np.max(
                                np.abs(z["native_friction"][rows] - coefficients),
                                axis=1,
                            )
                        )
                    ),
                )
                matched += 1
assert friction <= 1e-6 and matched > 0
summary = {
    "passed": True,
    "records": len(records),
    "backend_sha256": backend,
    "runner_sha256": runner,
    "model_signature": signature,
    "peak_penetration_mm": peak * 1000,
    "max_free_translation_error_um": pos * 1e6,
    "max_free_rotation_error_degrees": rot,
    "matched_native_friction_rows": matched,
    "max_native_friction_coefficient_error": friction,
}
(p / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
