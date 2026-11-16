"""Geometric contract of the measured workcell and grasp-based randomization."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_layout import InitialLayout

ASSETS = Path(__file__).resolve().parents[1] / "assets_local/wowrobo/components-v2"
pytestmark = pytest.mark.skipif(
    not (ASSETS / "manifest.json").is_file(), reason="Build component CAD first"
)


@pytest.fixture(scope="module")
def layout():
    return InitialLayout(ASSETS)


def test_rear_edge_measurements_and_fixed_target_direction(layout):
    l, r = layout.helper_bounds
    assert r[1, 0] - l[0, 0] == pytest.approx(0.72, abs=1e-8)
    assert layout.base_bounds[0, 0] - l[0, 0] == pytest.approx(0.30, abs=1e-8)
    np.testing.assert_allclose(layout.base_rotation @ [0, -1, 0], [1, 0, 0], atol=1e-12)
    assert layout.model.ncam == 2 and layout.model.nu == 12
    left_tip = layout.template.site_xpos[layout.model.site("left_gripperframe").id]
    right_tip = layout.template.site_xpos[layout.model.site("right_gripperframe").id]
    assert np.linalg.norm(left_tip - right_tip) > 0.12

    assert "base_servo_free" not in [
        layout.model.joint(i).name for i in range(layout.model.njnt)
    ]


def test_seeded_randomization_constrains_grip_center_not_component_center(layout):
    expected = [
        ("base_side_cover", "left"),
        ("shoulder_connector", "right"),
        ("assembly_pin", "left"),
        ("upper_arm_module", "right"),
        ("forearm_module", "left"),
    ]
    assert [(s["piece"], s["arm"]) for s in layout.cfg["pickup_order"]] == expected
    centers = []
    for seed in range(128):
        data, parts = layout.sample(seed)
        occupied = [layout.base_bounds, *layout.helper_bounds]
        assert np.isfinite(data.qpos).all()
        for name, item in parts.items():
            radial = item["grip_center"][:2] - layout.axes[item["arm"]][:2]
            assert abs(np.dot(radial, item["tangent"][:2])) < 1e-12
            assert abs(item["tangent"][2]) < 1e-12
            assert item["grasp_end_faces_arm"]
            assert item["nearest_grasp_end_gap_m"] < 1e-8
            assert item["nearest_grasp_distance_gap_m"] < 1e-8
            assert item["bounds"][0, 2] == pytest.approx(0, abs=1e-12)
            assert layout.clear(item, occupied)
            occupied.append(item["bounds"])
            root = layout.model.body(name).id
            grip_world = (
                data.xmat[root].reshape(3, 3) @ layout.grips[name]["center"]
                + data.xpos[root]
            )
            np.testing.assert_allclose(grip_world, item["grip_center"], atol=1e-10)
        centers.append(parts["upper_arm_module"]["grip_center"])
    # Actual position diversity and reproducibility, not merely random yaw labels.
    assert np.ptp(np.asarray(centers)[:, 1]) > 0.25
    a, _ = layout.sample(121)
    b, _ = layout.sample(121)
    np.testing.assert_array_equal(a.qpos, b.qpos)


def test_exported_xml_default_pose_matches_nominal_bank(layout, tmp_path):
    import json

    sig, bank = layout.export(tmp_path, count=3)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    states = np.load(bank / "initial_scene_train.npz")["states"]
    restore = mujoco.MjData(m)
    mujoco.mj_setState(m, restore, states[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for step in layout.cfg["pickup_order"]:
        j = m.joint(step["piece"] + "_free").qposadr[0]
        np.testing.assert_allclose(
            d.qpos[j : j + 7], restore.qpos[j : j + 7], atol=1e-11
        )
    assert json.loads((bank / "manifest.json").read_text())["model_signature"] == sig


def test_forearm_uses_opposite_servo_end(layout):
    step = next(s for s in layout.cfg["pickup_order"] if s["piece"] == "forearm_module")
    assert step["servo"] == 3
    assert step["grip_axis_sign"] == -1
    expected = np.array(
        layout.manifest["constituents"]["servo_3_case"]["bounds_m"]
    ).mean(0)
    np.testing.assert_allclose(layout.grips["forearm_module"]["center"], expected)


def test_side_cover_reversed_and_shoulder_restored(layout):
    steps = {s["piece"]: s for s in layout.cfg["pickup_order"]}
    assert steps["base_side_cover"]["grip_axis_sign"] == -1
    assert steps["shoulder_connector"].get("grip_axis_sign", 1) == 1
    for name, end, offset in [
        ("base_side_cover", "max", -0.018),
    ]:
        points = layout.vertices[name][:, 1]
        expected = (points.max() if end == "max" else points.min()) + offset
        assert layout.grips[name]["center"][1] == pytest.approx(expected)
