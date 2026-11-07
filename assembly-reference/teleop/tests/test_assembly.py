"""Native assembly integration checks; no headset, listening port or hardware."""
import sys
import json
import time
import gzip
import base64
from pathlib import Path

import mujoco
import numpy as np
import pytest

from teleop.backend import AssemblyController, AssemblyFrames, load_scene
from teleop.runtime import Capture


@pytest.fixture(scope='module')
def plant():
    return load_scene()


def settle(model, desc):
    c = AssemblyController(model, desc)
    for i in range(500):
        c.step(1_000_000_000+i*2_000_000)
    return c


def test_native_parts_settle_without_spurious_free_joint_faults(plant):
    model, desc = plant
    c = AssemblyController(model, desc)
    initial = c.data.qpos.copy()
    for i in range(500):
        c.step(1_000_000_000+i*2_000_000)
    assert model.nu == 12 and model.nq == 65
    assert model.opt.timestep == .0005
    assert np.isclose(c.data.time, 1.)
    assert not c.enabled and not c.stopped and not c.faults
    assert c.data.ncon > 0  # actual loose-part/table contact is present
    assert not c.collisions() and c.enable()
    z = int(model.joint('motor_free').qposadr[0])+2
    assert initial[z]-c.data.qpos[z] > .001  # gravity moved the free motor
    for name in ('base','motor','horn','screw','retainer','yoke','driver'):
        assert model.joint(name+'_free').type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.neq == 0  # no grasp weld, attachment or fastener shortcut


def test_both_arms_and_jaws_move_through_servo_physics_and_expire(plant):
    model, desc = plant
    c = settle(model, desc)
    assert c.enable()
    start = {s:c.tool(s) for s in ('left','right')}
    jaw_start = {s:float(c.data.qpos[desc['arms'][s]['gripper_qpos_index']]) for s in start}
    now = 3_000_000_000
    for side, pose in start.items():
        assert c.grab(side, pose.position, pose.rotation, now, now)
    for tick in range(100):
        now = 3_000_000_000+tick*20_000_000
        for side, pose in start.items():
            delta = np.array([.012, 0., -.012])*min(1.,tick/50)
            assert c.submit(side, pose.position+delta, pose.rotation, now, now)
            assert c.gripper(side, .45, now)
        for sub in range(10):
            c.step(now+sub*2_000_000)
        assert c.enabled and not c.faults, (c.reason,c.faults,c.collisions())
    for side, pose in start.items():
        assert np.linalg.norm(c.tool(side).position-pose.position) > .01
        assert abs(c.data.qpos[desc['arms'][side]['gripper_qpos_index']]-jaw_start[side]) > .1
    c.release('left')
    assert 'left' not in c.mapper.grabs
    c.step(now+220_000_000)
    assert not c.mapper.grabs and c.reason == 'original_input_expired'
    c.stop()
    assert c.stopped and not c.enabled and not c.enable()
    for i in range(500):
        c.step(now+240_000_000+i*2_000_000)
    assert c.reset_stop() and not c.enabled and not c.stopped


def test_three_native_cameras(plant):
    model, desc = plant
    c = settle(model, desc)
    frames = AssemblyFrames(model, desc['cameras'], width=480, height=270)
    try:
        images, capture = frames.render(Capture(tuple(c.data.qpos), 1, b'{}'))
        assert set(images) == {'main','left','right'}
        assert len(set(images.values())) == 3
        assert all(v.startswith(b'\xff\xd8') for v in images.values())
        assert frames.scene_option.geomgroup[3] == 0
    finally:
        frames.close()


def test_free_mode_keeps_native_contact_without_software_enable_block(plant):
    model, desc = plant
    c = settle(model, desc)
    base = model.body('left_base').id
    initial = model.body_pos[base].copy()
    try:
        # Deliberately invalid initialization fixture; never a live control command.
        model.body_pos[base,2] -= .10
        mujoco.mj_forward(model,c.data)
        hazards = c.collisions()
        assert c.data.ncon>0
        assert hazards==[]
        assert c.enable()
    finally:
        model.body_pos[base] = initial


def test_free_part_motion_does_not_prevent_robot_stop_reset(plant):
    model, desc = plant
    c = settle(model,desc)
    c.stop()
    v = int(model.joint('screw_free').dofadr[0])
    c.data.qvel[v:v+6] = .1
    assert c.reset_stop() and not c.enabled


def test_quest_protocol_reaches_native_owner_and_expires_without_new_receipts(plant):
    from teleop.backend import AssemblyRuntime
    model, desc = plant
    now = [5_000_000_000]
    runtime = AssemblyRuntime(model, desc, clock=lambda:now[0])
    runtime.initialize_owner()
    auth = runtime.create_session(0.)
    sequence = 0

    def packet(grip=False, enable=False, stop=False, offset=0.):
        nonlocal sequence
        sequence += 1
        controllers = {}
        for side in ('left','right'):
            tool = runtime.controller.tool(side)
            controllers[side] = dict(tracked=True,grip=grip,trigger=.5,
                position=(tool.position+np.array([offset,0.,0.])).tolist(),
                rotation=tool.rotation.ravel().tolist())
        return dict(session=auth['session'],sequence=sequence,sync_receive_ms=0.,
                    source_ms=(now[0]-5_000_000_000)/1e6,focus=True,
                    controllers=controllers,actions=dict(enable=enable,stop=stop))

    for tick in range(1,12):
        now[0] += 20_000_000
        runtime.cache_frames({k:b'camera transport fixture' for k in runtime.cameras},runtime.measured_snapshot())
        raw = packet(grip=tick>=3,enable=tick==1,offset=.001 if tick>=4 else 0.)
        assert runtime.receive_input(raw)['accepted']
        runtime.cycle()
    assert runtime.controller.enabled and set(runtime.controller.mapper.grabs)=={'left','right'}
    assert not runtime.controller.faults and runtime.get_config()['backend']=='assembly'
    assert np.isclose(runtime.controller.data.time,11*runtime.physics_steps_per_cycle*.002)
    now[0] += 250_000_000
    runtime.cycle()
    assert not runtime.controller.mapper.grabs
    raw = packet(stop=True)
    assert runtime.receive_input(raw)['accepted']
    runtime.cycle()
    assert runtime.controller.stopped and not runtime.controller.enabled


def test_new_owner_fences_old_session_and_identifies_every_camera_capture(plant):
    from teleop.backend import AssemblyRuntime
    model,desc=plant
    first=AssemblyRuntime(model,desc);first.initialize_owner()
    old_auth=first.create_session(0.)
    first.cache_frames({k:b'transport fixture' for k in first.cameras},first.measured_snapshot())
    owner=first.get_config()['owner_id']
    assert owner==old_auth['owner_id']==first.get_state()['owner_id']
    assert first.get_views()['owner_id']==first.get_views()['state']['owner_id']==owner
    second=AssemblyRuntime(model,desc);second.initialize_owner()
    assert second.owner_id!=owner
    result=second.receive_input(dict(session=old_auth['session'],actions=dict(enable=True)))
    assert not result['accepted'] and result['reason']=='unknown_or_superseded_session'
    assert result['owner_id']==second.owner_id
    auth=second.create_session(0.)
    assert auth['owner_id']==second.get_state()['owner_id']
    assert not second.controller.enabled and not second.controller.mapper.grabs


def test_free_mode_does_not_latch_a_measured_joint_speed_stop(plant):
    model,desc=plant
    c=settle(model,desc)
    assert not c.stop_on_joint_velocity and c.enable()
    c.data.qvel[c.dofs[4]]=2.
    c.step(3_000_000_000)
    assert max(abs(c.data.qvel[c.dofs]))>1.5
    assert not c.stopped and not c.faults and c.enabled
    c.stop()
    assert c.stopped and not c.enabled


def test_free_simulation_accepts_headset_clock_drift_without_changing_strict_default(plant):
    from teleop.backend import AssemblyRuntime
    from teleop.runtime import PreviewRuntime
    model,desc=plant;now=[10_000_000_000]
    for receipt in (True,False):
        runtime=AssemblyRuntime(model,desc,clock=lambda:now[0]) if receipt else PreviewRuntime(model,desc,clock=lambda:now[0])
        auth=runtime.create_session(0.)
        raw=dict(session=auth['session'],sequence=1,source_ms=110.,sync_receive_ms=0.,focus=True,actions=dict(enable=True),
                 controllers={s:dict(tracked=True,grip=False,trigger=0.,position=[0.,0.,0.],rotation=np.eye(3).ravel().tolist()) for s in ('left','right')})
        now[0]+=100_000_000
        result=runtime.receive_input(raw)
        assert result['accepted']==receipt
        if receipt:
            assert result['age_basis']=='server_receipt' and result['upper_age_ms']==0.
        else:assert result['reason']=='future_or_expired_original_source'


@pytest.mark.parametrize('side',['left','right'])
def test_direct_clutch_moves_from_any_hand_position_without_initial_tool_jump(plant,side):
    model,desc=plant;c=settle(model,desc);assert c.enable()
    initial=c.tool(side);hand=initial.position+np.array([.3,-.2,.1]);rotation=np.eye(3);now=3_000_000_000
    assert c.grab(side,hand,rotation,now,now)
    assert c.submit(side,hand,rotation,now,now)
    assert np.linalg.norm(c.goals[side].position-initial.position)<1e-12
    assert np.linalg.norm(c.goals[side].rotation-initial.rotation)<1e-12
    for tick in range(1,51):
        now=3_000_000_000+tick*20_000_000
        assert c.submit(side,hand+np.array([.012,0.,0.]),rotation,now,now)
        for sub in range(10):c.step(now+sub*2_000_000)
    assert np.linalg.norm(c.tool(side).position-initial.position)>.009
    assert c.enabled and not c.faults


def test_direct_mode_y_packet_enables_with_only_left_controller_tracked(plant):
    from teleop.backend import AssemblyRuntime
    model,desc=plant;now=[10_000_000_000];r=AssemblyRuntime(model,desc,clock=lambda:now[0]);r.initialize_owner();auth=r.create_session(0.)
    r.cache_frames({k:b'transport fixture' for k in r.cameras},r.measured_snapshot());now[0]+=20_000_000
    raw=dict(session=auth['session'],sequence=1,source_ms=20.,sync_receive_ms=0.,focus=True,actions=dict(enable=True),
        controllers=dict(left=dict(tracked=True,grip=False,trigger=0.,position=[0.,0.,0.],rotation=np.eye(3).ravel().tolist()),right=dict(tracked=False)))
    assert r.receive_input(raw)['accepted'];r.cycle();assert r.controller.enabled
    assert not r.controller.mapper.grabs and r.get_config()['acquire_from_anywhere']


def test_contact_loaded_failure_pose_remains_responsive_and_grabbed(plant):
    """Use real wall time: synthetic timestamps concealed the operator's failure."""
    from teleop.backend import AssemblyRuntime
    model,desc=plant
    fixture=json.loads(Path(__file__).with_name('contact_pose.json').read_text())
    runtime=AssemblyRuntime(model,desc);runtime.initialize_owner();c=runtime.controller
    # Initialization of an isolated fixture, never a live control operation.
    from teleop.tests.native_fixture import fixture_state
    c.data.qpos[:],c.data.qvel[:]=fixture_state(model,fixture)
    c.reference[:]=fixture['reference'];c.targets[:]=c.reference;c.data.ctrl[:]=c.reference
    mujoco.mj_forward(model,c.data);runtime.publish(time.monotonic_ns())
    initial=c.data.qpos[c.qids].copy()
    poses={side:c.tool(side) for side in ('left','right')}
    start=time.monotonic();auth=runtime.create_session(0.);durations=[]
    for tick in range(20):
        runtime.cache_frames({k:b'transport fixture' for k in runtime.cameras},runtime.measured_snapshot())
        raw=dict(session=auth['session'],sequence=tick+1,source_ms=(time.monotonic()-start)*1000,
                 sync_receive_ms=0.,focus=True,actions=dict(enable=tick==0),controllers={
                     s:dict(tracked=True,grip=tick>=2,trigger=.5,
                            position=(p.position+np.array([0.,0.,-.002 if tick>=4 else 0.])).tolist(),
                            rotation=p.rotation.ravel().tolist()) for s,p in poses.items()})
        assert runtime.receive_input(raw)['accepted']
        before=time.monotonic();runtime.cycle();durations.append(time.monotonic()-before)
        assert c.enabled and not c.faults, (c.reason,c.faults)
        if tick>=2:assert set(c.mapper.grabs)=={'left','right'},c.reason
        assert c.data.ncon>0 and np.isfinite(c.data.qpos).all()
    # The ten-step native cycle may take >150 ms on Mac. Input is polled and
    # measured state published every two controller steps, within that cycle.
    assert max(durations)<.2, durations
    assert np.max(abs(c.data.qpos[c.qids]-initial))>.001
    assert not c.data.warning.number.any()
    c.stop();assert c.stopped and not c.enabled


def test_stereo_geometry_matches_native_measured_body_and_mesh_frames(plant):
    from teleop.backend import AssemblyRuntime
    model,desc=plant;r=AssemblyRuntime(model,desc);r.initialize_owner()
    assets=json.loads(gzip.decompress(r.get_visual_scene()))
    state=r.get_state();poses=np.asarray(state['body_poses'])
    assert assets['owner_id']==state['owner_id']
    assert np.array_equal(poses,r.measured_snapshot().state['body_poses'])
    assert len(poses)==model.nbody and len(assets['geoms'])>60
    for geom in assets['geoms']:
        g=geom['id'];body=geom['body'];rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,poses[body,3:])
        assert np.allclose(poses[body,:3]+rotation.reshape(3,3)@geom['position'],r.controller.data.geom_xpos[g],atol=1e-12)
        assert model.geom_group[g]!=3
    for key,mesh in assets['meshes'].items():
        i=int(key);vertices=np.frombuffer(base64.b64decode(mesh['position']),dtype='<f4').reshape(-1,3)
        faces=np.frombuffer(base64.b64decode(mesh['index']),dtype='<u4').reshape(-1,3)
        v=int(model.mesh_vertadr[i]);f=int(model.mesh_faceadr[i])
        assert np.array_equal(vertices,model.mesh_vert[v:v+len(vertices)])
        assert np.array_equal(faces,model.mesh_face[f:f+len(faces)])
