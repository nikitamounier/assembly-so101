"""Native failure terminates; initial table drag remains allowed."""
import mujoco
import numpy as np
import pytest
from assembly_policy.cube_trial import core
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig


def loose_env():
    env = BatchedCubeEnv(['cube_handoff'], [0], EnvConfig(nthread=1))
    d, _ = core.generate('loose', 0, env.m)
    state = np.empty(env.nstate)
    mujoco.mj_getState(env.m, d, state, core.STATE_SPEC)
    env.set_states([0], state[None])
    return env


@pytest.mark.parametrize('at_timeout', [False, True])
def test_outside_table_contact_after_pickup_is_terminal_failure(at_timeout):
    env = loose_env()
    try:
        env.carry_started[0] = True
        if at_timeout:
            env.t[0] = env.horizon - 1
        reward, done, info = env.step(np.zeros((1, 12)))
        assert env.off_target_table_contact[0]
        assert done[0] and not info['success'][0] and not info['timeout'][0]
        assert reward[0] <= 0
        env.set_states([0], env.state.copy())
        assert not env.off_target_table_contact[0]
    finally:
        env.close()


def test_initial_table_contact_allowed_and_ten_second_timeout():
    env = loose_env()
    try:
        assert env.horizon == 250 and env.ecfg.control_dt == .04
        _, done, _ = env.step(np.zeros((1, 12)))
        assert not done[0] and not env.off_target_table_contact[0]
        env.t[0] = env.horizon - 1
        _, done, info = env.step(np.zeros((1, 12)))
        assert done[0] and info['timeout'][0] and not info['success'][0]
    finally:
        env.close()
