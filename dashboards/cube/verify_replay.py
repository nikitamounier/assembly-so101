"""Verify replay with a real MuJoCo placement; never writes a training success."""

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from server import Dashboard

from assembly_policy.cube_trial.core import STATE_SPEC, generate, model
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig
from assembly_policy.cube_trial.telemetry import EpisodeRecorder

m = model()
d, _ = generate("placed_held", 0, m)
state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
mujoco.mj_getState(m, d, state, STATE_SPEC)
env = BatchedCubeEnv(["cube_handoff"], [0], EnvConfig(nthread=1))
env.set_states([0], state[None])
with tempfile.TemporaryDirectory() as folder:
    recorder = EpisodeRecorder(folder, 1, ["placed_held"])
    recorder.complete[:] = True
    for _ in range(40):
        recorder.begin(env.observe(), np.array([0]))
        action = np.zeros((1, 12))
        action[:, [5, 11]] = 4
        reward, done, info = env.step(action)
        info["valid"] = np.array([True])
        recorder.step(reward, done, info, env.observe() if done[0] else None, np.array([0]), 1)
        if done[0]:
            break
    assert info["success"][0] and env.right_touchdown[0]
    clip = json.loads((Path(folder) / "latest-success.json").read_text())
    dash = Dashboard()
    dash.episode = clip
    replay = dash.render_episode()
    assert len(replay["frames"]) == len(clip["frames"]) > 2
    assert all(len(f) == len(dash.scene["geoms"]) for f in replay["frames"])
    assert np.isfinite(replay["frames"]).all()
    dash.episodes[clip["family"]] = clip
    assert dash.render_episode(clip["family"])["id"] == clip["id"]
    assert dash.render_episode("loose") is None  # never substitute another family's success
    print(
        json.dumps(
            dict(
                actual_native_placement_succeeded=True,
                qualified_right_touchdown=True,
                frames=len(replay["frames"]),
                rendered_geoms=len(dash.scene["geoms"]),
                episode_return=clip["episode_return"],
                training_artifacts_modified=False,
            )
        )
    )
env.close()
