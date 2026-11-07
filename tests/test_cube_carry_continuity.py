"""Free-flight gaps are allowed; off-target table contact after pickup is permanent."""

import mujoco
import numpy as np
import pytest

from assembly_policy.cube_trial import core
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig


def test_left_pickup_arms_history_and_air_gap_is_allowed():
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        env._continuity_sample(0, True, False, False, False)
        env._continuity_sample(0, True, True, True, False)
        assert not env.carry_started[0] and not env.continuity_broken[0]
        env._continuity_sample(0, False, True, True, False)  # actual left pickup
        assert env.carry_started[0]
        env._continuity_sample(0, False, True, True, False)  # overlapping handoff
        env._continuity_sample(0, False, False, True, False)  # right only
        assert not env.continuity_broken[0]
        env._continuity_sample(0, False, False, False, False)  # one native-step gap
        env._continuity_sample(0, False, False, True, False)  # caught again
        assert not env.continuity_broken[0]
    finally:
        env.close()


def test_release_at_target_table_contact_is_allowed_but_not_outside():
    env = BatchedCubeEnv(["cube_handoff"], [0, 0], EnvConfig(nthread=1))
    try:
        env.carry_started[:] = True
        env.right_touchdown[:] = True
        env._continuity_sample(0, True, False, False, True)
        env._continuity_sample(1, True, False, False, False)
        assert env.delivery_touchdown[0] and not env.continuity_broken[0]
        assert not env.continuity_broken[1] and not env.delivery_touchdown[1]
        assert env.off_target_table_contact[1]
        env._continuity_sample(0, True, False, False, True)
        assert not env.continuity_broken[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    "family", ["pre_handoff", "dual_grasp", "post_handoff", "right_transport", "near_target", "placed_held"]
)
def test_midtask_reset_has_valid_prefix_and_reset_clears_history(family):
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        d, _ = core.generate(family, 0, env.m)
        state = np.empty(env.nstate)
        mujoco.mj_getState(env.m, d, state, core.STATE_SPEC)
        env.set_states([0], state[None])
        assert env.carry_started[0] and not env.continuity_broken[0]
        env.continuity_broken[0] = True
        env.off_target_table_contact[0] = True
        env.set_states([0], state[None])
        assert not env.continuity_broken[0] and not env.off_target_table_contact[0]
        assert env.delivery_touchdown[0] == (family == "placed_held")
    finally:
        env.close()


@pytest.mark.parametrize("gap,expected", [(False, 1), (True, 1)])
def test_gap_allows_actual_native_placement_with_recent_right_grasp(gap, expected):
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1, cost_coef=0))
    m = env.m
    try:
        # A real settled, released cube in the circle, with arms clear.
        d, _ = core.generate("placed_held", 0, m)
        for side in ("left", "right"):
            qi, _ = core.arm_indices(m, side)
            d.qpos[qi] = [0, -0.7, -0.5, 0.3, 0, 0.8]
        d.ctrl[:] = d.qpos[env.qi]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        assert core.success(m, d)
        mujoco.mj_getState(m, d, env.state[0], core.STATE_SPEC)
        mujoco.mj_setState(m, env.data[0], env.state[0], core.STATE_SPEC)
        env.last_ctrl[0] = d.ctrl
        env._refresh()
        env.carry_started[0] = True
        env.grounded[0] = False
        env.last_right_grasp_time[0] = 0
        if gap:
            env._continuity_sample(0, False, False, False, False)
        for _ in range(4):
            reward, done, info = env.step(np.zeros((1, 12)))
            if done[0]:
                break
        assert env.right_touchdown[0]  # still satisfies the100ms lookback
        assert int(reward[0]) == expected and bool(info["success"][0]) == bool(expected)
        assert not env.continuity_broken[0]
    finally:
        env.close()


def test_initial_drag_allowed_but_any_later_outside_table_contact_is_permanent():
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        for _ in range(5):
            env._continuity_sample(0, True, True, True, False)  # initial slide/drag
        assert not env.carry_started[0] and not env.off_target_table_contact[0]
        env._continuity_sample(0, False, True, True, False)  # lifted by left
        assert env.carry_started[0]
        env._continuity_sample(0, True, True, True, False)  # still held, touches outside
        assert env.off_target_table_contact[0] and not env.continuity_broken[0]
        env.right_touchdown[0] = True
        env._continuity_sample(0, True, False, True, True)  # later right delivery
        assert env.off_target_table_contact[0]
    finally:
        env.close()


def test_outside_contact_after_initial_target_landing_also_disqualifies():
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        env.carry_started[0] = True
        env.right_touchdown[0] = True
        env._continuity_sample(0, True, False, False, True)
        assert env.delivery_touchdown[0] and not env.off_target_table_contact[0]
        env._continuity_sample(0, True, False, False, False)  # rolled out
        assert env.off_target_table_contact[0]
        env._continuity_sample(0, True, False, False, True)  # rolled back in
        assert env.off_target_table_contact[0]
    finally:
        env.close()


def test_continuity_uses_actual_touch_even_without_supporting_load(monkeypatch):
    from types import SimpleNamespace

    m = core.model()
    cube = m.geom("cube_geom").id
    pads = [m.geom(s + "_" + p + "_pad").id for s in ("left", "right") for p in ("fixed", "moving")]
    d = SimpleNamespace(
        contact=[
            SimpleNamespace(geom1=cube, geom2=p, dist=dist) for p, dist in zip(pads, [-0.0001, 0, 0.0001, 0])
        ]
    )

    def force(m, d, index, result):
        result[:] = 0

    d.contact.append(SimpleNamespace(geom1=cube,geom2=m.geom("table").id,dist=0))
    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    c = core.placement_contacts(m, d)
    assert c["table_touching"] and not c["table_contact"]
    assert not c["pad_support"]
    np.testing.assert_array_equal(c["pad_touching"], [True, True, False, True])
