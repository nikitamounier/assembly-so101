import hashlib
import json
import mujoco
import numpy as np
import pytest

from assembly_policy.cube_trial.core import generate, STATE_SPEC
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig
from assembly_policy.cube_trial.remote import _pack


def test_dynamic_reset_restores_history_action_grasp_clock_and_continuation(tmp_path):
    original = BatchedCubeEnv(['cube_handoff'], [0], EnvConfig(nthread=1))
    restored = None
    try:
        d, _ = generate('post_handoff', 0, original.m)
        state = np.empty(original.nstate)
        mujoco.mj_getState(original.m, d, state, STATE_SPEC)
        original.set_states([0], state[None])
        for _ in range(3):
            original.step(np.full((1, 12), .03))
        state = original.state.copy()
        history = original.hist.copy()
        previous = original.prev_action.copy()
        age = original.data[0].time-original.last_right_grasp_time[0]
        path = tmp_path/'reset-context.npz'
        np.savez_compressed(path, state_sha256=[hashlib.sha256(state[0].tobytes()).hexdigest()],
                            histories=history, previous_actions=previous, right_grasp_age_s=[age], carry_started=original.carry_started.copy())
        (tmp_path/'manifest.json').write_text(json.dumps({'reset_context_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
        restored = BatchedCubeEnv(['cube_handoff'], [0], EnvConfig(nthread=1, reset_context_path=str(path)))
        restored.set_states([0], state)
        np.testing.assert_array_equal(restored.hist, history)
        np.testing.assert_array_equal(restored.prev_action, previous)
        assert restored.last_right_grasp_time[0] == -age
        assert restored.carry_started[0] == original.carry_started[0]
        packed = _pack(restored.observe(), 1)
        np.testing.assert_array_equal(packed['proprio'], history.reshape(1, -1))
        action = np.full((1,12), .02)
        original.step(action)
        restored.step(action)
        np.testing.assert_allclose(restored.data[0].qpos, original.data[0].qpos, atol=1e-9)
        np.testing.assert_allclose(restored.data[0].qvel, original.data[0].qvel, atol=1e-8)
        np.testing.assert_allclose(restored.hist, original.hist, atol=1e-6)
        # An unrelated reset clears the mined prefix and uses normal repeated history.
        restored.set_states([0], original.state.copy())
        np.testing.assert_array_equal(restored.prev_action, 0)
        np.testing.assert_array_equal(restored.hist[:,0],restored.hist[:,-1])
    finally:
        original.close()
        if restored is not None:
            restored.close()


def test_reset_context_checksum_required(tmp_path):
    path = tmp_path/'reset-context.npz'
    path.write_bytes(b'invalid')
    (tmp_path/'manifest.json').write_text(json.dumps({'reset_context_sha256':'wrong'}))
    with pytest.raises(ValueError, match='checksum'):
        BatchedCubeEnv(['cube_handoff'], [0], EnvConfig(nthread=1, reset_context_path=str(path)))
