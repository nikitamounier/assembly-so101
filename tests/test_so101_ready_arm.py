"""Ready-pose checks include disabled visual geometry and never repair states."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101 import scene
from assembly_policy.so101.ready_arm import (
    _segment_intersects_box,
    prepare_ready_arms,
    validate_ready_arms,
)
from assembly_policy.so101.robot import arm_indices, home


@pytest.fixture(scope="module")
def model():
    return scene.model(contact_monitor=True)


def fresh(model):
    data = mujoco.MjData(model)
    home(model, data)
    return data


def test_both_nearby_open_and_validation_is_read_only(model):
    data = fresh(model)
    receipt = prepare_ready_arms(model, data, np.random.default_rng(3))
    assert receipt["ok"] and receipt["sides"] == ["left", "right"]
    assert receipt["geometry_checks"] > 100
    assert not receipt["dynamic_retention_verified"]
    before = [a.copy() for a in (data.qpos, data.qvel, data.ctrl, data.qacc, data.qfrc_applied)]
    assert validate_ready_arms(model, data, receipt["sides"])["ok"]
    for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.qacc, data.qfrc_applied), before):
        np.testing.assert_array_equal(actual, expected)
    for side in receipt["sides"]:
        qi, _ = arm_indices(model, side)
        assert data.qpos[qi[5]] == 0.65
        assert data.ctrl[model.actuator(side + "_gripper").id] == 0.65


def test_active_arm_controls_are_preserved(model):
    data = fresh(model)
    qi, vi = arm_indices(model, "left")
    oldq, oldv, oldctrl = data.qpos[qi].copy(), data.qvel[vi].copy(), data.ctrl[:6].copy()
    receipt = prepare_ready_arms(model, data, np.random.default_rng(4), active_side="left")
    assert receipt["sides"] == ["right"]
    np.testing.assert_array_equal(data.qpos[qi], oldq)
    np.testing.assert_array_equal(data.qvel[vi], oldv)
    np.testing.assert_array_equal(data.ctrl[:6], oldctrl)


def test_corridor_and_closed_gripper_are_rejected(model):
    data = fresh(model)
    prepare_ready_arms(model, data, np.random.default_rng(3))
    point = data.site("right_gripperframe").xpos.copy()
    receipt = validate_ready_arms(model, data, ["right"], corridor_start=point, corridor_end=point)
    assert not receipt["ok"]
    assert any("corridor" in error for error in receipt["errors"])
    qi, _ = arm_indices(model, "right")
    data.qpos[qi[5]] = 0
    receipt = validate_ready_arms(model, data, ["right"])
    assert not receipt["ok"]
    assert any("not open" in error for error in receipt["errors"])


def test_failed_preparation_is_atomic(model):
    data = fresh(model)
    before = [a.copy() for a in (data.qpos, data.qvel, data.ctrl)]
    with pytest.raises(ValueError, match="No clear nearby"):
        prepare_ready_arms(
            model,
            data,
            np.random.default_rng(3),
            attempts=1,
            corridor_start=[-0.4, 0.15, 0.17],
            corridor_end=[0.4, 0.15, 0.17],
        )
    for actual, expected in zip((data.qpos, data.qvel, data.ctrl), before):
        np.testing.assert_array_equal(actual, expected)


def test_home_is_not_a_nearby_pose(model):
    receipt = validate_ready_arms(model, fresh(model), ["left", "right"])
    assert not receipt["ok"]
    assert any("outside nearby" in error for error in receipt["errors"])


def test_segment_box_checks_interior_and_parallel_cases():
    lower, upper = np.zeros(3), np.ones(3)
    assert _segment_intersects_box(np.array([-1.0, 0.5, 0.5]), np.array([2.0, 0.5, 0.5]), lower, upper)
    assert not _segment_intersects_box(np.array([-1.0, 2.0, 0.5]), np.array([2.0, 2.0, 0.5]), lower, upper)


def test_idle_geometry_rejects_part_overlap(model):
    data = fresh(model)
    prepare_ready_arms(model, data, np.random.default_rng(3))
    q = int(model.joint("motor_free").qposadr[0])
    data.qpos[q : q + 3] = data.site("right_gripperframe").xpos
    receipt = validate_ready_arms(model, data, ["right"])
    assert not receipt["ok"]
    assert any("motor" in error for error in receipt["errors"])


def test_nonadjacent_self_collision_is_rejected(model):
    data = fresh(model)
    qi, _ = arm_indices(model, "left")
    data.qpos[qi] = [1.0807519618, 1.2000536451, 0.1862866182, 1.4644718365, -2.6414966134, 0.65]
    receipt = validate_ready_arms(model, data, ["left"])
    assert any("self-link clearance failure" in error for error in receipt["errors"])


@pytest.mark.parametrize("kwargs", [{"clearance_m": float("nan")}, {"corridor_radius_m": float("inf")}])
def test_nonfinite_clearance_is_rejected(model, kwargs):
    with pytest.raises(ValueError, match="nonnegative"):
        validate_ready_arms(model, fresh(model), ["left"], **kwargs)
