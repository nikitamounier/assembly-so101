"""Training and Quest compile the same canonical factory, without adapter overrides."""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import sys
from pathlib import Path
import mujoco
import numpy as np
from teleop.backend import ROOT,load_scene
from assembly_policy.so101.env import BatchedAssemblyEnv,EnvConfig
from assembly_policy.so101.robot import home
from assembly_policy.so101.workcell import environment_identity


def test_training_and_quest_have_identical_compiled_models_and_initialization():
    quest,description=load_scene()
    env=BatchedAssemblyEnv(['motor'],[0],EnvConfig(nthread=1,max_contact_depth=PENETRATION_LIMIT_M))
    try:
        assert (ROOT/'configs/so101_scene.json').is_file()
        def binary(m):
            data=np.empty(mujoco.mj_sizeModel(m),np.uint8)
            mujoco.mj_saveModel(m,buffer=data)
            return data
        np.testing.assert_array_equal(binary(quest),binary(env.m))
        d=mujoco.MjData(env.m);home(env.m,d)
        np.testing.assert_array_equal(d.qpos,description['home_qpos'])
        np.testing.assert_array_equal(d.ctrl,description['home_ctrl'])
        assert description['environment_identity']==environment_identity()
    finally:env.close()


def test_native_learner_excludes_unused_link_and_preserves_network_dimensions():
    from assembly_policy.so101.env import STATE_SPEC
    env=BatchedAssemblyEnv(['yoke'],[0],EnvConfig(nthread=1))
    try:
        d=mujoco.MjData(env.m);home(env.m,d)
        state=np.empty(mujoco.mj_stateSize(env.m,STATE_SPEC));mujoco.mj_getState(env.m,d,state,STATE_SPEC)
        env.set_states([0],state[None]);observation=env.observe()
        assert observation['proprio'].shape==(1,265)
        assert 'arm_link' not in env.names and env.m.nu==12
        assert observation['part_p'].shape[1]==11
        reward,done,info=env.step(np.zeros((1,12)))
        assert np.isfinite(reward).all() and not info['success'].any()
    finally:env.close()
