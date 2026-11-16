"""Reject the reported overlapping poses instead of stabilizing them in a wall."""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest
from teleop.backend import AssemblyController,ROOT,load_scene
from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from teleop.serve import resume_description

@pytest.fixture(scope='module')
def plant():return load_scene()

@pytest.mark.parametrize('filename',['motor_seated_pose.json','motor_side_pose.json'])
def test_reported_interpenetrating_poses_cannot_be_resumed(plant,filename,tmp_path):
    m,desc=plant;path=Path(__file__).with_name(filename);saved=json.loads(path.read_text())
    from teleop.tests.native_fixture import fixture_state
    qpos,qvel=fixture_state(m,saved);saved=dict(saved,qpos=qpos.tolist(),qvel=qvel.tolist())
    path=tmp_path/filename;path.write_text(json.dumps(saved))
    assert sampled_motor_base_overlap(m,saved['qpos'],ROOT/'assets_local')>.0005
    with pytest.raises(ValueError,match='material overlap'):
        resume_description(m,desc,path)

def test_free_motor_responds_to_force_without_sleep_or_attachment(plant):
    m,desc=plant;d=mujoco.MjData(m);d.qpos[:]=desc['home_qpos'];d.ctrl[:]=desc['home_ctrl']
    mujoco.mj_step(m,d,nstep=4000);before=d.body('motor').xpos.copy()
    d.xfrc_applied[m.body('motor').id,2]=.8;mujoco.mj_step(m,d,nstep=200)
    assert np.linalg.norm(d.body('motor').xpos-before)>.005
    assert not m.opt.enableflags&int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    assert m.neq==0 and d.body_awake[m.body('motor').id]==1 and np.isfinite(d.qpos).all()

def test_valid_resume_preserves_measured_physics_and_starts_disabled(plant,tmp_path):
    m,desc=plant;q=desc['home_qpos'];v=np.zeros(m.nv).tolist();path=tmp_path/'valid.json'
    path.write_text(json.dumps(dict(qpos=q,qvel=v)))
    resumed=resume_description(m,desc,path);c=AssemblyController(m,resumed)
    assert np.array_equal(c.data.qpos,q) and np.array_equal(c.data.qvel,v)
    assert not c.enabled and not c.stopped and not c.mapper.grabs
    path.write_text(json.dumps(dict(qpos=[0.],qvel=[0.])))
    with pytest.raises(ValueError):resume_description(m,desc,path)
