"""Explicit scene changes preserve canonical native physics and revoke old grabs."""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import time
import mujoco
import numpy as np
import pytest
from teleop.backend import AssemblyRuntime, load_scene
from assembly_policy.so101.task_scenes import TASKS


@pytest.fixture(scope='module')
def plant():
    m,description=load_scene();r=AssemblyRuntime(m,description);r.prepare_tasks()
    return m,description,r.prepared_scenes


@pytest.mark.parametrize('task',TASKS)
def test_completed_states_earned_native_reward_without_attachments(plant,task):
    m,description,prepared=plant;start=prepared[task]
    assert np.isfinite(start['qpos']).all() and np.isfinite(start['qvel']).all()
    assert start['preparation']['predecessors']==list(TASKS[:TASKS.index(task)])
    assert start['preparation']['completed_tasks']==list(TASKS[:TASKS.index(task)+1])
    assert start['preparation']['native_reward']['passed']
    assert start['preparation']['native_reward']['final']['success']
    assert start['preparation']['native_reward']['physics_seconds']>=.44
    assert max(start['preparation']['placement_error_m'].values())<.001
    assert start['preparation']['max_contact_depth_m']<PENETRATION_LIMIT_M
    assert start['preparation']['sampled_motor_base_overlap_m']<PENETRATION_LIMIT_M
    assert m.neq==0
    for name in (*TASKS,'base','driver'):
        assert m.joint(name+'_free').type[0]==mujoco.mjtJoint.mjJNT_FREE


def runtime(plant):
    m,desc,prepared=plant;now=[5_000_000_000];r=AssemblyRuntime(m,desc,clock=lambda:now[0])
    r.prepared_scenes=prepared;r.initialize_owner();auth=r.create_session(0.);sequence=[0]
    def packet(**actions):
        now[0]+=20_000_000;sequence[0]+=1
        raw=dict(session=auth['session'],sync_receive_ms=0.,sequence=sequence[0],source_ms=sequence[0]*20.,
                 focus=True,actions=actions,controllers={})
        for side in ('left','right'):
            pose=r.controller.tool(side)
            raw['controllers'][side]=dict(tracked=True,grip=False,trigger=0.,position=pose.position.tolist(),rotation=pose.rotation.ravel().tolist())
        return raw
    r.cache_frames({k:b'camera fixture' for k in r.cameras},r.measured_snapshot())
    return r,now,packet


def test_A_completes_current_task_once_then_wraps_to_loose_reset(plant):
    r,now,packet=runtime(plant)
    for stage in (*range(1,len(TASKS)+1),0):
        assert r.receive_input(packet(advance_task=True))['accepted'];r.cycle()
        assert r.stage_index==stage and r.task_index==max(0,stage-1)
        assert not r.controller.enabled and not r.controller.stopped
        assert not r.controller.mapper.grabs and not r.controller.samples
        generation=r.scene_generation
        assert r.receive_input(packet(advance_task=True))['accepted'];r.cycle()
        assert r.scene_generation==generation
        assert r.receive_input(packet())['accepted'];r.cycle()
    assert r.receive_input(packet(advance_task=True))['accepted'];r.cycle()
    r.controller.faults.append('fixture_fault');r.controller.stop()
    assert r.receive_input(packet(environment_reset=True))['accepted'];r.cycle()
    assert r.stage_index==0 and r.task_index==0 and not r.controller.faults and not r.controller.stopped
    assert not r.controller.enabled and not r.controller.mapper.grabs
    # The loose reset includes clearance above the table. Compare its subsequent
    # gravity/contact motion with an independent native controller, not a pause.
    from teleop.backend import AssemblyController
    initial=plant[2]['initial']
    expected=AssemblyController(r.model,dict(plant[1],home_qpos=initial['qpos'],initial_qvel=initial['qvel']))
    for _ in range(r.physics_steps_per_cycle):expected.step(now[0])
    np.testing.assert_array_equal(r.controller.data.qpos,expected.data.qpos)
    np.testing.assert_array_equal(r.controller.data.qvel,expected.data.qvel)
    r.apply_scene_action('environment_reset')
    np.testing.assert_array_equal(r.controller.data.qpos,plant[2]['initial']['qpos'])
    np.testing.assert_array_equal(r.controller.data.qvel,plant[2]['initial']['qvel'])


def test_stop_and_expired_scene_actions_cannot_advance(plant):
    r,now,packet=runtime(plant)
    assert r.receive_input(packet(advance_task=True,stop=True))['accepted'];r.cycle()
    assert r.task_index==0 and r.scene_generation==0 and r.controller.stopped
    assert r.receive_input(packet())['accepted'];r.cycle()
    assert r.receive_input(packet(advance_task=True))['accepted'];now[0]+=250_000_000;r.cycle()
    assert r.task_index==0 and r.scene_generation==0


def test_stereo_controls_survive_old_hidden_JPEGs_but_still_expire_missing_inputs(plant):
    r,now,packet=runtime(plant)
    for tick in range(50):
        raw=packet(enable=tick==0)
        for c in raw['controllers'].values():c['grip']=tick>=2
        assert r.receive_input(raw)['accepted'];r.cycle()
        assert r.controller.enabled
        if tick>=2:assert set(r.controller.mapper.grabs)=={'left','right'}
    assert now[0]-r.frame_acquired_ns>500_000_000
    assert r._fresh(now[0])
    now[0]+=250_000_000;r.cycle()
    assert not r.controller.mapper.grabs and not r.controller.samples


def test_new_Stop_is_admitted_inside_a_native_physics_cycle(plant):
    r,now,packet=runtime(plant)
    assert r.receive_input(packet(enable=True))['accepted'];r.cycle()
    original=r.controller.step;steps=[]
    def step(stamp):
        steps.append(len(steps)+1)
        if len(steps)==1:
            assert r.receive_input(packet(stop=True))['accepted']
            assert not r.controller.stopped
        if len(steps)==3:
            assert r.controller.stopped
        return original(stamp)
    r.controller.step=step
    r.cycle()
    assert len(steps)==10 and r.controller.stopped and not r.pending_stop


@pytest.mark.parametrize('task',TASKS[1:])
def test_post_motor_presets_face_the_screw_downward_from_above(plant,task):
    axis=np.asarray(plant[2][task]['preparation']['screw_insertion_axis_world'])
    assert np.dot(axis,[0.,0.,-1.])>.999


def test_unused_link_is_absent_from_shared_scene_and_tasks(plant):
    from assembly_policy.so101.env import TASK_NAMES
    from assembly_policy.so101.scene import metadata
    m,desc,prepared=plant
    assert TASKS==TASK_NAMES and TASKS[-1]=='yoke'
    assert 'arm_link' not in prepared
    assert 'arm_link' not in metadata()
    assert mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'arm_link')==-1
    assert m.nbody==27 and m.nq==65 and m.nv==58


def test_motor_stays_sideways_and_round_horn_is_flat_after_upright_rotation(plant):
    prepared=plant[2];d=mujoco.MjData(plant[0])
    d.qpos[:]=prepared['motor']['qpos'];mujoco.mj_forward(plant[0],d)
    assert np.dot(d.body('base').xmat.reshape(3,3)[:,2],[0,1,0])>.999
    for task in TASKS[1:]:
        d.qpos[:]=prepared[task]['qpos'];mujoco.mj_forward(plant[0],d)
        assert np.dot(d.body('base').xmat.reshape(3,3)[:,2],[0,0,1])>.999
        assert prepared[task]['preparation']['horn_flatness_error_rad']<np.deg2rad(1.)


@pytest.mark.parametrize('task',TASKS)
def test_loaded_completion_states_independently_pass_the_native_reward_again(plant,task):
    from assembly_policy.so101.task_scenes import audit_reward
    report=audit_reward(plant[0],plant[2][task],task,ticks=30)
    assert report['passed'],report
