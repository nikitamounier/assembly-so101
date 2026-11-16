"""Native response and lifecycle regressions for the wearer-requested faster profile."""
import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

from teleop.backend import AssemblyController,load_scene
from teleop.control import Kinematics,Pose


@pytest.fixture(scope='module')
def plant():
    return load_scene()


def ready(plant,side):
    model,desc=plant;c=AssemblyController(model,desc)
    for tick in range(100):c.step(1_000_000_000+tick*2_000_000)
    assert c.enable()
    pose=c.tool(side);now=2_000_000_000
    assert c.grab(side,pose.position,pose.rotation,now,now)
    assert c.submit(side,pose.position,pose.rotation,now,now)
    return c,pose


def advance(c,tick):
    now=2_000_000_000+tick*20_000_000
    for sub in range(10):
        c.step(now+sub*2_000_000)
        assert not c.faults
        assert np.max(abs(c.data.qvel[c.dofs]))<1.5
    return now


@pytest.mark.parametrize('side',['left','right'])
def test_tracks_attainable_rotation_during_translation(plant,side):
    model,desc=plant;c,start=ready(plant,side);fk=Kinematics(model,desc)
    seed=c.data.qpos.copy();errors=[]
    for tick in range(1,81):
        now=2_000_000_000+tick*20_000_000;q=seed.copy()
        q[desc['arms'][side]['qpos_indices']]+=np.array([.1,.2,.2,-.3,.55])*min(1.,tick*.02/.8)
        goal=fk.forward(q,side)
        assert c.submit(side,goal.position,goal.rotation,now,now)
        advance(c,tick);actual=c.tool(side)
        errors.append([np.linalg.norm(goal.position-actual.position),np.linalg.norm(fk.orientation_error(actual.rotation,goal.rotation))])
        assert c.enabled
    rms=np.sqrt(np.mean(np.array(errors[:75])**2,axis=0))
    assert rms[0]<.002
    assert np.degrees(rms[1])<3.
    assert c.submit_status[side]['orientation_error_rad']<.03


@pytest.mark.parametrize('side',['left','right'])
def test_wrist_rotation_reaches_within_three_degrees_in_point_seven_seconds(plant,side):
    c,start=ready(plant,side);a=.6
    rot=np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]])
    goal=Pose(start.position,start.rotation@rot)
    for tick in range(1,36):
        now=2_000_000_000+tick*20_000_000
        assert c.submit(side,goal.position,goal.rotation,now,now)
        advance(c,tick)
    actual=c.tool(side)
    assert np.linalg.norm(actual.position-start.position)<.004
    assert np.linalg.norm(Kinematics.orientation_error(actual.rotation,goal.rotation))<.05


@pytest.mark.parametrize('side',['left','right'])
def test_eight_centimetre_move_reaches_within_five_mm_in_point_eight_seconds(plant,side):
    c,start=ready(plant,side);goal=Pose(start.position+[0.,0.,-.08],start.rotation)
    for tick in range(1,71):
        now=2_000_000_000+tick*20_000_000
        assert c.submit(side,goal.position,goal.rotation,now,now)
        advance(c,tick)
        if tick>=40:
            assert np.linalg.norm(c.tool(side).position-goal.position)<.005


@pytest.mark.parametrize('action',['release','stop','expiry'])
def test_fast_rotation_still_relinquishes_on_release_stop_or_original_expiry(plant,action):
    c,start=ready(plant,'right');a=.6
    rot=np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]])
    for tick in range(1,14):
        now=2_000_000_000+tick*20_000_000
        assert c.submit('right',start.position,start.rotation@rot,now,now)
        advance(c,tick)
    assert max(abs(c.data.qvel[c.dofs]))>.8
    if action=='release':c.release('right')
    if action=='stop':c.stop()
    for tick in range(14,70):advance(c,tick)
    assert not c.mapper.grabs and not c.samples and not c.goals
    if action=='stop':assert c.stopped and not c.enabled and not c.enable()
    elif action=='expiry':assert c.reason=='original_input_expired'


def test_unattainable_and_half_turn_orientations_remain_finite_bounded_and_reported(plant):
    model,desc=plant;ik=Kinematics(model,desc);q=np.array(desc['home_qpos'])
    for side in ('left','right'):
        initial=ik.forward(q,side)
        for a in (.4,np.pi):
            rot=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
            result=ik.solve(q,side,Pose(initial.position,initial.rotation@rot))
            assert result['accepted'] and result['position_error_m']<.004
            assert result['orientation_error_rad']>.2
            assert np.isfinite(result['target']).all()
            assert np.all(result['target']>=ik.arms[side]['limits'][:,0])
            assert np.all(result['target']<=ik.arms[side]['limits'][:,1])
            assert result['iterations']<=55


def test_other_arm_and_gripper_do_not_slow_the_right_arm(plant):
    solo,start=ready(plant,'right');both,_=ready(plant,'right')
    left=both.tool('left');now=2_000_000_000
    assert both.grab('left',left.position,left.rotation,now,now)
    a=.3;rot=np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]])
    for tick in range(1,31):
        now=2_000_000_000+tick*20_000_000
        for ctl in (solo,both):
            assert ctl.submit('right',start.position,start.rotation@rot,now,now)
        assert both.submit('left',left.position,left.rotation@rot.T,now,now)
        assert both.gripper('left',.6,now)
        advance(solo,tick);advance(both,tick)
        assert np.linalg.norm(solo.tool('right').position-both.tool('right').position)<.0001
        assert np.linalg.norm(Kinematics.orientation_error(solo.tool('right').rotation,both.tool('right').rotation))<.001
