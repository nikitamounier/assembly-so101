"""Interactive drop regressions, not insertion-success or hardware qualification."""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import ctypes
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from teleop.backend import load_scene
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.scene import metadata


@pytest.fixture(scope='module')
def plant():
    result=load_scene()
    lib=ctypes.CDLL(None)
    assert hasattr(lib,'assembly_contact_capacity_guard'), 'The canonical model factory must load its native contact adapter'
    assert lib.assembly_contact_capacity_guard()==4
    return result


def test_contact_capacity_covers_sdf_plane_and_mesh_returns(plant):
    m,_=plant;lib=ctypes.CDLL(None);capacity=lib.mj_maxContact
    capacity.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int];capacity.restype=ctypes.c_int
    for name in ('base_contact','motor_contact'):
        assert capacity(m._address,m.geom('table').id,m.geom(name).id,0)==mujoco.mjMAXCONPAIR
    assert m.opt.sdf_initpoints==50  # capacity is separate from actual search work
    assert m.numeric('so101_sdf_search').data[0]==4
    assert capacity(m._address,m.geom('left_clamp_a').id,m.geom('left_clamp_b').id,0)==8


@pytest.mark.parametrize('offset',[(.003,0,.06),(-.003,.002,.06),(0,-.025,.06)])
def test_clear_motor_drop_is_finite_and_settles_without_explosive_speed(plant,offset):
    m,desc=plant;d=mujoco.MjData(m);d.qpos[:]=desc['home_qpos'];d.ctrl[:]=desc['home_ctrl']
    target=origins(metadata());b=int(m.joint('base_free').qposadr[0]);q=int(m.joint('motor_free').qposadr[0]);v=int(m.joint('motor_free').dofadr[0])
    mujoco.mj_forward(m,d);rotation=d.body('base').xmat.reshape(3,3)
    d.qpos[q:q+3]=d.qpos[b:b+3]+rotation@(target['motor']-target['base'])+offset
    d.qpos[q+3:q+7]=d.qpos[b+3:b+7]
    from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
    from teleop.backend import ROOT
    assert sampled_motor_base_overlap(m,d.qpos,ROOT/'assets_local')<1e-6
    peak_speed=0.;settling_speeds=[]
    for tick in range(500):
        mujoco.mj_step(m,d,nstep=20)
        peak_speed=max(peak_speed,float(np.linalg.norm(d.qvel[v:v+3])))
        if tick>=460:settling_speeds.append(float(np.linalg.norm(d.qvel[v:v+3])))
    assert np.isfinite(d.qpos).all() and not d.warning.number.any()
    assert peak_speed<2.,peak_speed  # free fall from this height is about1.2m/s
    assert np.linalg.norm(d.qvel[v:v+3])<.01
    # A single low-speed sample can hide recurring support/contact chatter.
    assert max(settling_speeds)<.01
    assert sampled_motor_base_overlap(m,d.qpos,ROOT/'assets_local')<PENETRATION_LIMIT_M


def test_misaligned_horn_drop_settles_throughout_release_tail(plant):
    """The lightweight horn must not keep rocking on a single support contact."""
    from assembly_policy.so101 import scene
    from assembly_policy.so101.contact_audit import sampled_assembly_overlaps
    from assembly_policy.so101.task_scenes import task_scene

    m,_=plant;prepared=task_scene(m,'horn');d=mujoco.MjData(m)
    for name in ('qpos','qvel','ctrl'):getattr(d,name)[:]=prepared[name]
    q=int(m.joint('horn_free').qposadr[0]);v=int(m.joint('horn_free').dofadr[0])
    d.qpos[q:q+3]+=[.001,0.,.01]
    delta=np.empty(4);mujoco.mju_axisAngle2Quat(delta,np.array([1.,0.,0.]),np.deg2rad(2.))
    quat=d.qpos[q+3:q+7].copy();mujoco.mju_mulQuat(d.qpos[q+3:q+7],delta,quat)
    d.qvel[v:v+6]=0.;mujoco.mj_forward(m,d)
    initial=sampled_assembly_overlaps(d,scene.PARTS,scene.ASSETS)
    assert max(value for pair,value in initial.items() if 'horn' in pair.split('/'))<1e-6
    peak=tail=depth=0.
    for step in range(round(5./m.opt.timestep)):
        mujoco.mj_step(m,d)
        speed=float(np.linalg.norm(d.qvel[v:v+3]));peak=max(peak,speed)
        depth=max(depth,-min(d.contact.dist,default=0.))
        if (step+1)*m.opt.timestep>=4.6:tail=max(tail,speed)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
    assert peak<2. and depth<PENETRATION_LIMIT_M,(peak,depth)
    assert tail<.01,tail
    mujoco.mj_kinematics(m,d)
    assert max(sampled_assembly_overlaps(d,scene.PARTS,scene.ASSETS).values())<PENETRATION_LIMIT_M
