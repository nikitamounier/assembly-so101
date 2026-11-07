"""Physics and sparse-reward controls for the native cube-handoff trainer."""

import mujoco
import numpy as np
import pytest

from assembly_policy.cube_trial.core import STATE_SPEC, generate, model, success
from assembly_policy.cube_trial.families.dual_grasp import validate_four_pad_contacts


@pytest.fixture(scope="module")
def cube_model():
    return model()


def integration_state(m, d):
    state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
    mujoco.mj_getState(m, d, state, STATE_SPEC)
    return state


def test_physical_right_release_is_required_for_placement(cube_model):
    m = cube_model
    d, _ = generate("placed_held", 0, m)
    assert not success(m, d), "A cube resting in the circle while grasped is not success"
    start = d.body("cube").xpos.copy()
    for side in ("left", "right"):
        d.ctrl[m.actuator(side + "_gripper").id] = 0.8
    observed = []
    for _ in range(10):
        mujoco.mj_step(m, d, nstep=25)
        mujoco.mj_forward(m, d)
        observed.append(int(success(m, d)))
    assert set(observed) <= {0, 1}
    assert observed[-1] == 1, "Opening the actual jaws must allow placement success"
    assert np.linalg.norm(d.body("cube").xpos - start) < 0.003
    # A released, settled cube outside the circle must still receive no reward.
    d.mocap_pos[m.body("target").mocapid[0], 0] += 0.1
    mujoco.mj_forward(m, d)
    assert not success(m, d)


def test_native_handoff_holds_without_freefall_or_plugins(cube_model):
    m = cube_model
    assert m.nplugin == 0 and m.neq == 0
    assert not np.any(m.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
    assert mujoco.get_mjcb_contactfilter() is None
    d, _ = generate("dual_grasp", 0, m)
    start = d.body("cube").xpos.copy()
    pads = {s: [m.geom(s + "_" + p + "_pad").id for p in ("fixed", "moving")] for s in ("left", "right")}
    expected = {"left": {(1, -1), (1, 1)}, "right": {(2, -1), (2, 1)}}
    for _ in range(10):
        mujoco.mj_step(m, d, nstep=25)
        mujoco.mj_forward(m, d)
        validate_four_pad_contacts(m, d, m.geom("cube_geom").id, pads, expected_faces=expected)
    assert np.linalg.norm(d.body("cube").xpos - start) < 0.003
    assert not success(m, d)


def test_integration_replay_preserves_individual_target_pose(cube_model):
    m = cube_model
    originals = [generate("placed_held", seed, m)[0] for seed in (0, 1)]
    assert not np.array_equal(originals[0].mocap_pos, originals[1].mocap_pos)
    replay = mujoco.MjData(m)
    for original in (*originals, originals[0]):
        state = integration_state(m, original)
        mujoco.mj_setState(m, replay, state, STATE_SPEC)
        mujoco.mj_forward(m, replay)
        np.testing.assert_array_equal(replay.mocap_pos, original.mocap_pos)
        np.testing.assert_array_equal(replay.site("target_center").xpos, original.site("target_center").xpos)
        np.testing.assert_allclose(integration_state(m, replay), state, rtol=0, atol=1e-12)


def test_batched_native_rollout_matches_two_direct_envs_for_ten_steps(cube_model):
    from assembly_policy.cube_trial.env import PHYSICS_SPEC, BatchedCubeEnv, EnvConfig

    m = cube_model
    originals = [generate(family, seed, m)[0] for family, seed in (("dual_grasp", 0), ("placed_held", 1))]
    for i, d in enumerate(originals):
        d.xfrc_applied[m.body("cube").id, 0] = 0.05 * (1 if i == 0 else -1)
        angle = 0.3 * (1 if i == 0 else -1)
        d.mocap_quat[m.body("target").mocapid[0]] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    states = np.array([integration_state(m, d) for d in originals])
    env = BatchedCubeEnv(["cube_handoff"], np.zeros(2, int), EnvConfig(nthread=2, cost_coef=0))
    direct = [mujoco.MjData(m) for _ in originals]
    rng = np.random.default_rng(512)
    try:
        env.set_states(np.arange(2), states)
        for i, d in enumerate(direct):
            mujoco.mj_setState(m, d, env.state[i], STATE_SPEC)
            mujoco.mj_forward(m, d)
        for _ in range(10):
            actions = rng.uniform(-0.08, 0.08, (2, 12))
            scales = np.tile(np.r_[env.ecfg.action_scale_joints, env.ecfg.gripper_scale], 2)
            for i, d in enumerate(direct):
                d.ctrl[:] = np.clip(
                    d.ctrl + scales * actions[i],
                    m.actuator_ctrlrange[:, 0],
                    m.actuator_ctrlrange[:, 1],
                )
                mujoco.mj_step(m, d, nstep=env.substeps)
                mujoco.mj_forward(m, d)
            reward, done, info = env.step(actions)
            assert np.isin(reward, [0, 1]).all()
            assert not done.any() and not info["success"].any()
            for i, d in enumerate(direct):
                actual = np.empty(mujoco.mj_stateSize(m, PHYSICS_SPEC))
                expected = np.empty_like(actual)
                mujoco.mj_getState(m, env.data[i], actual, PHYSICS_SPEC)
                mujoco.mj_getState(m, d, expected, PHYSICS_SPEC)
                # Stock rollout does not return endpoint solver warmstarts.
                # This10-step forced probe measured maximum state error6.68e-6
                # and maximum pose component error4.00e-7.
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
                np.testing.assert_allclose(actual[1 : 1 + m.nq], expected[1 : 1 + m.nq], rtol=0, atol=1e-6)
                np.testing.assert_array_equal(env.data[i].xfrc_applied, originals[i].xfrc_applied)
                np.testing.assert_array_equal(env.data[i].mocap_pos, originals[i].mocap_pos)
                np.testing.assert_array_equal(env.data[i].mocap_quat, originals[i].mocap_quat)
    finally:
        env.close()


def test_observations_and_subset_reset_keep_targets_isolated(cube_model):
    from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig

    m = cube_model
    states = np.array([integration_state(m, generate("placed_held", seed, m)[0]) for seed in (0, 1)])
    env = BatchedCubeEnv(["cube_handoff"], np.zeros(2, int), EnvConfig(nthread=2, cost_coef=0))
    try:
        env.set_states(np.arange(2), states)
        observation = env.observe()
        shapes = {
            "proprio": (2, 250),
            "task": (2,),
            "tcp_p": (2, 4, 3),
            "tcp_R": (2, 4, 3, 3),
            "part_p": (2, 2, 3),
            "part_q": (2, 2, 4),
            "finger": (2, 2),
        }
        assert observation.keys() == shapes.keys()
        for key, shape in shapes.items():
            assert observation[key].shape == shape
            assert np.isfinite(observation[key]).all()
        assert not np.array_equal(observation["part_p"][0, 1], observation["part_p"][1, 1])
        env.step(np.zeros((2, 12)))
        untouched = env.state[1].copy()
        env.set_states([0], states[:1])
        np.testing.assert_allclose(env.state[1], untouched, rtol=0, atol=1e-12)
        np.testing.assert_allclose(env.observe()["part_p"][0], observation["part_p"][0], rtol=0, atol=1e-7)
        assert not env.prev_action[0].any() and env.t[0] == 0
    finally:
        env.close()


def test_environment_reward_has_only_zero_or_one_and_requires_real_release(cube_model):
    from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig

    m = cube_model
    env = BatchedCubeEnv(["cube_handoff"], np.zeros(1, int), EnvConfig(nthread=1, cost_coef=0))
    state = integration_state(m, generate("placed_held", 0, m)[0])
    try:
        env.set_states([0], state[None])
        reward, done, _ = env.step(np.zeros((1, 12)))
        assert reward[0] == 0 and not done[0]
        actions = np.zeros((1, 12))
        actions[:, [5, 11]] = 4.0
        for _ in range(30):
            reward, done, info = env.step(actions)
            assert reward[0] in (0, 1) and info["cost"][0] == 0
            if done[0]:
                break
        assert reward[0] == 1 and done[0] and info["success"][0]
        assert success(m, env.data[0])
    finally:
        env.close()


@pytest.mark.parametrize("family", ["dual_grasp", "right_transport"])
def test_neutral_policy_preserves_preloaded_native_grasp(cube_model, family):
    from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig
    from assembly_policy.cube_trial.families.dual_grasp import validate_pad_contacts

    m = cube_model
    d, meta = generate(family, 0, m)
    start = d.body("cube").xpos.copy()
    holders = meta["holder"]
    pads = {s: [m.geom(s + "_" + p + "_pad").id for p in ("fixed", "moving")] for s in holders}
    expected = {s: {(1, -1), (1, 1)} if s == "left" else {(2, -1), (2, 1)} for s in holders}
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1, cost_coef=0))
    try:
        env.set_states([0], integration_state(m, d)[None])
        for _ in range(13):
            reward, done, _ = env.step(np.zeros((1, 12)))
            assert reward[0] == 0 and not done[0]
            validate_pad_contacts(m, env.data[0], m.geom("cube_geom").id, pads, expected_faces=expected)
        assert np.linalg.norm(env.data[0].body("cube").xpos - start) < 0.003
    finally:
        env.close()


def test_unheld_touchdown_cannot_earn_placement(cube_model):
    from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig

    m = cube_model
    d, _ = generate("placed_held", 0, m)
    for side in ("left", "right"):
        d.ctrl[m.actuator(side + "_gripper").id] = 0.8
    mujoco.mj_step(m, d, nstep=500)
    mujoco.mj_forward(m, d)
    assert success(m, d)
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1, cost_coef=0))
    try:
        env.set_states([0], integration_state(m, d)[None])
        assert not env.right_touchdown[0]
        for _ in range(5):
            reward, done, info = env.step(np.zeros((1, 12)))
            assert reward[0] == 0 and not info["success"][0]
    finally:
        env.close()


def test_assembly_control_cost_formula(cube_model):
    from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig

    m = cube_model
    env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
    try:
        env.set_states([0], integration_state(m, generate("dual_grasp", 0, m)[0])[None])
        action = np.full((1, 12), 0.02)
        reward, done, info = env.step(action)
        d = env.data[0]
        velocity = d.qvel[m.jnt_dofadr[m.actuator_trnid[:, 0]]]
        expected = 2e-5 * (
            (action**2).sum()
            + (velocity**2 / 25).sum()
            + ((d.actuator_force / m.actuator_forcerange[:, 1]) ** 2).sum()
        )
        np.testing.assert_allclose(info["cost"][0], expected, rtol=1e-12)
        np.testing.assert_allclose(reward[0], -expected, rtol=1e-6)
        assert not done[0]
    finally:
        env.close()


@pytest.mark.parametrize("angle,expected", [(0.2749, False), (0.275, True), (0.2751, True)])
def test_release_angle_threshold_boundary(cube_model, angle, expected):
    from assembly_policy.cube_trial.core import arm_indices

    m = cube_model
    d, _ = generate("placed_held", 0, m)
    for side in ("left", "right"):
        d.ctrl[m.actuator(side + "_gripper").id] = 0.8
    mujoco.mj_step(m, d, nstep=500)
    # Move both hands clear, keeping the settled cube in the target.
    for side in ("left", "right"):
        qi, _ = arm_indices(m, side)
        d.qpos[qi] = [0, -0.7, -0.5, 0.3, 0, 0.8]
    d.qpos[arm_indices(m, "right")[0][5]] = angle
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    assert success(m, d) == expected
