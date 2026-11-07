"""A stationary picture cannot substitute for an unchanged-control hold."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_equilibrium import rotation_degrees, settle_and_hold
from assembly_policy.so101.wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS, PARTS


def toy(*, gravity, height):
    bodies = []
    for i, name in enumerate(PARTS):
        children = ""
        shafts = {
            "base_servo": [1],
            "upper_arm_module": [2],
            "forearm_module": [3, 4],
        }.get(name, [])
        for shaft in shafts:
            children += f'<body name="target_servo_{shaft}_pivot"><joint name="target_servo_{shaft}_shaft" type="hinge" damping=".02"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0"/></body>'
        bodies.append(
            f'<body name="{name}" pos="{i * 0.1} 0 {height}"><freejoint/><geom type="sphere" size=".01" mass=".1"/>{children}</body>'
        )
    equalities = "".join(
        f'<weld name="{name}" body1="{a}" body2="{b}" active="false"/>'
        for name, a, b, _, _ in ASSEMBLY_ATTACHMENTS
    )
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><option timestep=".001" gravity="0 0 {gravity}" integrator="implicitfast"/><worldbody>{"".join(bodies)}</worldbody><equality>{equalities}</equality></mujoco>'
    )
    d = mujoco.MjData(m)
    validator = SimpleNamespace(
        model=m,
        _grip_pairs=lambda _: [],
        _loaded_pads=lambda _d, _pairs: {},
        _native_audit=lambda _d: {"candidate_rejected": False},
        definitions={"00_loose": {"task": "initial_scene"}},
    )
    return validator, d


def test_unchanged_stationary_configuration_passes_without_manufactured_velocity():
    v, d = toy(gravity=0, height=0.1)
    state, ctrl, receipt, end = settle_and_hold(
        v, "00_loose", d, settle_seconds=0.01, hold_seconds=0.02
    )
    assert receipt["equilibrium_pass"]
    assert receipt["controls_unchanged_during_hold"]
    assert receipt["velocity_cleared_after_settling"] is False
    assert receipt["prescribed_bracket_angle"] is False
    np.testing.assert_array_equal(ctrl, d.ctrl)
    assert state.shape == end.shape


def test_zero_velocity_start_in_freefall_is_rejected_and_actual_velocity_is_saved():
    v, d = toy(gravity=-9.81, height=1.0)
    state, _, receipt, _ = settle_and_hold(
        v, "00_loose", d, settle_seconds=0.02, hold_seconds=0.05
    )
    assert not receipt["equilibrium_pass"]
    assert "part_moves_during_hold" in receipt["rejection_reasons"]
    assert "nonzero_hold_velocity" in receipt["rejection_reasons"]
    restored = mujoco.MjData(v.model)
    mujoco.mj_setState(v.model, restored, state, mujoco.mjtState(receipt["state_spec"]))
    assert np.max(np.abs(restored.qvel)) > 0.1


def test_rotation_difference_and_invalid_windows():
    assert rotation_degrees(np.eye(3), np.diag([-1.0, -1.0, 1.0])) == pytest.approx(
        180.0
    )
    v, d = toy(gravity=0, height=0.1)
    with pytest.raises(ValueError, match="positive"):
        settle_and_hold(v, "00_loose", d, hold_seconds=0)
