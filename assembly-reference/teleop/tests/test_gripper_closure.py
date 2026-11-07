"""The actual fingers stop by native contact and reopen, without CAD overlap."""
import mujoco
import numpy as np
import pytest
from teleop.backend import load_scene,AssemblyController
from assembly_policy.so101.contact_audit import finger_surface_clearance
from assembly_policy.so101.workcell import configuration


@pytest.fixture(scope='module')
def plant():return load_scene()


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('initial',[.65,.1])
def test_empty_fingers_touch_by_contact_before_joint_limit_and_reopen(plant,side,initial):
    m,description=plant;d=mujoco.MjData(m);d.qpos[:]=description['home_qpos'];d.ctrl[:]=description['home_ctrl']
    q=int(m.joint(side+'_gripper').qposadr[0]);a=m.actuator(side+'_gripper').id
    d.qpos[q]=initial;d.ctrl[a]=description['arms'][side]['gripper_closed']
    mujoco.mj_step(m,d,nstep=2400)
    clearance=finger_surface_clearance(m,d,side)
    assert clearance['sampled_surface_gap_m']<.00005,clearance
    assert clearance['max_sampled_material_overlap_m']<.0002,clearance
    assert d.qpos[q]>m.joint(side+'_gripper').range[0]+.02
    forces=[]
    for i,c in enumerate(d.contact):
        if any(m.geom(int(g)).name==side+'_distal_finger_contact' for g in c.geom):
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);forces.append(f[0])
    assert sum(forces)>1.
    assert not d.warning.number.any() and m.neq==0
    d.ctrl[a]=description['arms'][side]['gripper_open'];mujoco.mj_step(m,d,nstep=2400)
    assert finger_surface_clearance(m,d,side)['sampled_surface_gap_m']>.01


def test_quest_and_policy_have_same_gripper_command_endpoints(plant):
    m,description=plant;c=AssemblyController(m,description);assert c.enable()
    for side in ('left','right'):
        pose=c.tool(side);now=1_000_000_000
        assert c.grab(side,pose.position,pose.rotation,now,now)
        assert c.submit(side,pose.position,pose.rotation,now,now)
        for openness,key in ((0,'gripper_closed'),(1,'gripper_open')):
            assert c.gripper(side,openness,now)
            a=m.actuator(side+'_gripper').id
            assert c.targets[a]==pytest.approx(description['arms'][side][key])
        np.testing.assert_equal(m.actuator(side+'_gripper').ctrlrange,
                                [configuration()['gripper'][k] for k in ('closed_command_rad','open_command_rad')])
