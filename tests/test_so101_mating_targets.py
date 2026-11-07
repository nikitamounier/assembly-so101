"""Regression checks for physical receiving frames and spline-indexed goals."""

import copy

import mujoco
import numpy as np
import pytest

from assembly_policy.so101 import scene, workcell
from assembly_policy.so101.fastener import AXIS, bearing_pose, origins, target_rotations
from assembly_policy.so101.goals import placement_errors
from assembly_policy.so101.task_scenes import geometric_scene


@pytest.fixture(scope="module")
def physical_model():
    return scene.model()


def test_opposite_within_base_tolerance_offsets_do_not_fake_a_mated_horn(physical_model):
    d = geometric_scene(physical_model, "horn")
    R = d.body("base").xmat.reshape(3, 3).copy()
    for name, offset in [("motor", 0.00075), ("horn", -0.00075)]:
        q = int(physical_model.joint(name + "_free").qposadr[0])
        d.qpos[q : q + 3] += R @ np.array([offset, 0, 0])
    mujoco.mj_kinematics(physical_model, d)
    meta = scene.metadata()
    errors = placement_errors(d, "horn", meta, origins(meta))
    assert errors["motor"] == pytest.approx(0.00075, abs=1e-9)
    assert errors["horn"] == pytest.approx(0.0015, abs=1e-9)


def test_joint_motion_preserves_the_actual_horn_mating_transform(physical_model):
    d = geometric_scene(physical_model, "horn")
    R = d.body("base").xmat.reshape(3, 3).copy()
    for name in ("motor", "horn"):
        q = int(physical_model.joint(name + "_free").qposadr[0])
        d.qpos[q : q + 3] += R @ np.array([0.00075, 0, 0])
    mujoco.mj_kinematics(physical_model, d)
    meta = scene.metadata()
    errors = placement_errors(d, "horn", meta, origins(meta))
    assert errors["motor"] == pytest.approx(0.00075, abs=1e-9)
    assert errors["horn"] < 1e-9


def test_indexed_targets_compose_with_the_sideways_base(monkeypatch):
    cfg = workcell.configuration()
    cfg["mating_targets"] = {
        name: dict(lift_m=lift, shaft_yaw_deg=-3.0) for name, lift in [("horn", 0.0007), ("yoke", 0.0005)]
    }
    monkeypatch.setattr(workcell, "configuration", lambda: copy.deepcopy(cfg))
    m = scene.model(assembled=True)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    rotation = d.body("base").xmat.reshape(3, 3)
    meta = scene.metadata()
    goals = target_rotations(meta)
    for name in ("horn", "yoke"):
        np.testing.assert_allclose(rotation.T @ d.body(name).xmat.reshape(3, 3), goals[name], atol=1e-9)
    assert max(placement_errors(d, "yoke", meta, origins(meta)).values()) < 1e-8


@pytest.mark.parametrize("turns", [0.0, 0.25])
def test_bearing_pose_matches_actual_shaft_and_thread_phase(physical_model, turns):
    d = geometric_scene(physical_model, "horn")
    meta = scene.metadata()
    position, quaternion = bearing_pose(physical_model, d, meta, preload_turns=turns)
    motor, horn = d.body("motor"), d.body("horn")
    R = motor.xmat.reshape(3, 3)
    R[:, 2]
    origin = motor.xpos + R @ (AXIS - np.asarray(meta["motor"]["origin"]))
    local = R.T @ (position - origin)
    assert np.linalg.norm(local[:2]) < 1e-12
    normal = horn.xmat.reshape(3, 3)[:, 2]
    bearing = horn.xpos + normal * meta["horn"]["bounds"][1][2]
    assert abs(normal @ (position - bearing)) < 1e-12
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    relative = R.T @ rotation.reshape(3, 3)
    np.testing.assert_allclose(relative[:, 2], [0, 0, 1], atol=1e-12)
    yaw = np.arctan2(relative[1, 0], relative[0, 0])
    instance = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_PLUGIN, "female")
    values = bytes(physical_model.plugin_attr[int(physical_model.plugin_attradr[instance]) :]).split(b"\0")
    pitch, phase = float(values[0]), float(values[8])
    mismatch = (local[2] / pitch - yaw / (2 * np.pi) - phase - turns + 0.5) % 1 - 0.5
    assert abs(mismatch) < 1e-12


def test_indexed_yoke_approach_has_clearance_in_original_meshes(physical_model):
    from assembly_policy.so101.contact_audit import sampled_assembly_overlaps

    d = geometric_scene(physical_model, "yoke")
    q = int(physical_model.joint("yoke_free").qposadr[0])
    seated = d.qpos[q : q + 3].copy()
    direction = np.asarray(workcell.configuration()["assembly_plan"]["extractions"]["yoke"])
    direction = d.body("base").xmat.reshape(3, 3) @ direction / np.linalg.norm(direction)
    for distance in np.linspace(0, 0.022, 12):
        d.qpos[q : q + 3] = seated + distance * direction
        mujoco.mj_kinematics(physical_model, d)
        overlaps = sampled_assembly_overlaps(d, ["base", "motor", "horn", "retainer", "yoke"], scene.ASSETS)
        assert max(overlaps.values()) < 0.00035, (distance, overlaps)
