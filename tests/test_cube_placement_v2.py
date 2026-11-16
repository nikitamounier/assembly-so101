"""Native placement physics plus force and touchdown-window boundary controls."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from assembly_policy.cube_trial import core
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig


@pytest.fixture
def released():
    m = core.model()
    d, _ = core.generate("placed_held", 0, m)
    for side in ("left", "right"):
        qi, _ = core.arm_indices(m, side)
        d.qpos[qi] = [0, -0.7, -0.5, 0.3, 0, 0.8]
    d.qpos[core.arm_indices(m, "left")[0][5]] = 0  # left may stay closed
    d.qpos[core.arm_indices(m, "right")[0][5]] = 0.275
    q = int(m.joint("cube_free").qposadr[0])
    d.qpos[q : q + 3] = [0.1, 0, core.CUBE_SIZE / 2 - 1e-5]
    d.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
    d.mocap_pos[m.body("target").mocapid[0], :2] = d.qpos[q : q + 2]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    assert core.success(m, d)
    return m, d


@pytest.mark.parametrize("overlap,expected", [(-1e-6, True), (1e-6, False)])
def test_entire_cube_boundary_includes_two_mm_tolerance(released, overlap, expected):
    m, d = released
    half = core.CUBE_SIZE / 2
    offset = np.sqrt((core.TARGET_RADIUS + 0.002) ** 2 - half**2) - half + overlap
    d.mocap_pos[m.body("target").mocapid[0], 0] = d.body("cube").xpos[0] - offset
    mujoco.mj_forward(m, d)
    assert core.success(m, d) == expected


def test_tilted_table_contact_allowed_but_no_contact_rejected(released):
    m, d = released
    q = int(m.joint("cube_free").qposadr[0])
    theta = 0.3
    d.qpos[q + 3 : q + 7] = [np.cos(theta / 2), np.sin(theta / 2), 0, 0]
    d.qpos[q + 2] = core.CUBE_SIZE / 2 * (np.cos(theta) + np.sin(theta)) - 1e-5
    mujoco.mj_forward(m, d)
    assert abs(d.body("cube").xpos[2] - core.CUBE_SIZE / 2) > 0.001
    assert core.success(m, d)
    d.qpos[q + 2] += 0.002
    mujoco.mj_forward(m, d)
    assert not core.placement_contacts(m, d)["table_contact"]
    assert not core.success(m, d)


@pytest.mark.parametrize(
    "linear,angular,expected",
    [
        (0.01999, 0, True),
        (0.02, 0, False),
        (0.02001, 0, False),
        (0, 0.49999, True),
        (0, 0.5, False),
        (0, 0.50001, False),
    ],
)
def test_separate_strict_speed_limits(released, linear, angular, expected):
    m, d = released
    v = int(m.joint("cube_free").dofadr[0])
    d.qvel[v] = linear
    d.qvel[v + 3] = angular
    mujoco.mj_forward(m, d)
    assert core.success(m, d) == expected


@pytest.mark.parametrize(
    "loads,support,held",
    [
        ([0.01] * 4, False, False),
        ([0.01001, 0, 0, 0], True, False),
        ([0, 0.01001, 0, 0], True, False),
        ([0, 0, 0.01001, 0], True, False),
        ([0, 0, 0, 0.01001], True, False),
        ([0, 0, 0.01001, 0.01001], True, True),
    ],
)
def test_all_four_pads_loaded_contact_and_brushing(monkeypatch, loads, support, held):
    m = core.model()
    cube = m.geom("cube_geom").id
    pads = [m.geom(s + "_" + p + "_pad").id for s in ("left", "right") for p in ("fixed", "moving")]
    contacts = [SimpleNamespace(geom1=cube, geom2=p, dist=0) for p in pads]
    contacts.append(SimpleNamespace(geom1=cube, geom2=m.geom("table").id, dist=0))

    def force(m, d, index, result):
        result[:] = 0
        result[0] = [*loads, 0.2][index]

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    result = core.placement_contacts(m, SimpleNamespace(contact=contacts))
    np.testing.assert_allclose(result["pad_loads_n"], loads)
    assert result["table_contact"] and result["pad_support"] == support and result["right_held"] == held


@pytest.mark.parametrize("elapsed,expected", [(0.0999, True), (0.1, True), (0.1001, False)])
def test_touchdown_uses_grasp_in_preceding_100_ms(elapsed, expected):
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        env._touchdown_sample(0, 0, False, True)
        env._touchdown_sample(0, elapsed / 2, False, False)
        env._touchdown_sample(0, elapsed, True, False)
        assert env.right_touchdown[0] == expected
        # An episode reset must not inherit grasp history from a previous cube.
        d, _ = core.generate("loose", 0, env.m)
        state = np.empty(mujoco.mj_stateSize(env.m, core.STATE_SPEC))
        mujoco.mj_getState(env.m, d, state, core.STATE_SPEC)
        env.set_states([0], state[None])
        assert not env.right_touchdown[0] and np.isneginf(env.last_right_grasp_time[0])
    finally:
        env.close()


def test_grasp_history_is_sampled_above_the_previous_height_shortcut():
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        d, _ = core.generate("dual_grasp", 0, env.m)
        assert d.body("cube").xpos[2] > 0.035
        state = np.empty(mujoco.mj_stateSize(env.m, core.STATE_SPEC))
        mujoco.mj_getState(env.m, d, state, core.STATE_SPEC)
        env.set_states([0], state[None])
        env.step(np.zeros((1, 12)))
        assert env.last_right_grasp_time[0] == pytest.approx(0.04)
    finally:
        env.close()
