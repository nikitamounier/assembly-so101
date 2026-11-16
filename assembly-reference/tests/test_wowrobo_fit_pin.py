"""The selected shaft must reach every exported native/print mesh dimension."""

import json
from pathlib import Path

import numpy as np

from scripts.build_wowrobo_fit_pin import pin_mesh


def test_fit_pin_uses_selected_diameter_and_preserves_user_grip_lengths():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/wowrobo_components.json"
        ).read_text()
    )
    p = config["pin_mm"]
    mesh = pin_mesh(p)
    assert mesh.is_watertight
    assert p["shaft_diameter"] == 2.3
    assert (
        p["grip_diameter"] == 12
        and p["shaft_length"] == 14.5
        and p["total_length"] == 42.5
    )
    np.testing.assert_allclose(mesh.bounds, [[-6, -6, 0], [6, 6, 42.5]], atol=1e-10)
    shaft = mesh.vertices[mesh.vertices[:, 2] > p["grip_length"] + 0.001]
    radius = np.linalg.norm(shaft[:, :2], axis=1)
    np.testing.assert_allclose(radius[radius > 0], 1.15, atol=1e-10)
    # Catch the previous bug where STEP honored config but OBJ hardcoded1.35.
    other = dict(p, shaft_diameter=2.1)
    changed = pin_mesh(other)
    cap = changed.vertices[changed.vertices[:, 2] > 28.001]
    radii = np.linalg.norm(cap[:, :2], axis=1)
    np.testing.assert_allclose(radii[radii > 0], 1.05, atol=1e-10)


def test_registered_pin_has_real_head_stop_and_positive_blind_end_clearance():
    import mujoco
    import pytest

    from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
    from assembly_policy.so101.wowrobo_pin_registration import (
        apply_pin_registration,
        measure_pin_registration,
    )

    root = Path(__file__).resolve().parents[1]
    scene = root / "assets_local/wowrobo/fit-pin-v2/registration-review.xml"
    nominal = (
        root / "assets_local/wowrobo/photo-resets-v3/stages/17_left_support_train.npz"
    )
    if not scene.exists() or not nominal.exists():
        pytest.skip("Requires local pin/CAD review assets")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, np.load(nominal)["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    dims = json.loads((root / "configs/wowrobo_components.json").read_text())["pin_mm"]
    registration = measure_pin_registration(model, data, dims)
    assert registration["annulus_hits"] > 1000
    assert registration["sampled_min_radial_clearance_m"] > 0.00009
    assert 0.00028 < registration["seated_axial_clearance_m"] < 0.00032
    assert registration["shaft_tip_below_bore_reference_m"] > 0
    assert registration["case_mesh_watertight"] is False
    apply_pin_registration(model, data, registration, "17_left_support")
    address = model.joint("assembly_pin_free").qposadr[0]
    indices = np.ones(model.nq, dtype=bool)
    indices[address : address + 7] = False
    np.testing.assert_array_equal(data.qpos[indices], before[indices])
    axis = -data.xmat[model.body("assembly_pin").id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(axis, registration["bore_axis_world"], atol=1e-10)
    receipt = NativeBoundaryAudit(model).audit(data)
    pin_pairs = [
        p
        for p in receipt["target_target"]
        if p["a"]["geom_name"] == "pin" or p["b"]["geom_name"] == "pin"
    ]
    assert pin_pairs and max(p["penetration_m"] for p in pin_pairs) < 1e-6
