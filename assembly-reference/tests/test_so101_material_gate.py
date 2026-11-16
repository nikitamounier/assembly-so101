"""A near-goal retainer inside the base must not receive completion reward."""
import mujoco
import numpy as np
import pytest
from assembly_policy.so101 import scene
from assembly_policy.so101.env import BatchedAssemblyEnv,EnvConfig
from assembly_policy.so101.task_scenes import geometric_scene
from assembly_policy.so101.contact_audit import sampled_assembly_overlaps

@pytest.fixture(scope='module')
def physical_model():return scene.model(contact_monitor=True)


def test_intersecting_retainer_cannot_complete_inside_pose_tolerance(physical_model):
    env=BatchedAssemblyEnv(['retainer'],[0],EnvConfig(nthread=1),model_factory=lambda **_:physical_model)
    try:
        d=geometric_scene(physical_model,'retainer')
        q=int(physical_model.joint('retainer_free').qposadr[0])
        d.qpos[q:q+3]-=d.body('base').xmat.reshape(3,3)@np.array([0,0,.0007])
        mujoco.mj_forward(physical_model,d);env.data[0]=d
        assert env.assembly_distance()[0]<env.ecfg.success_threshold
        env.stable[:]=11;success=np.array([True]);safety=np.array([False])
        overlap=env._check_material_success(success,safety)
        assert overlap[0]>env.ecfg.max_material_overlap
        assert not success[0] and safety[0] and env.stable[0]==0
    finally:env.close()


def test_within_budget_motor_candidate_survives_material_guard(physical_model):
    env=BatchedAssemblyEnv(['motor'],[0],EnvConfig(nthread=1),model_factory=lambda **_:physical_model)
    try:
        env.data[0]=geometric_scene(physical_model,'motor')
        success=np.array([True]);safety=np.array([False]);env.stable[:]=11
        overlap=env._check_material_success(success,safety)
        assert 0<overlap[0]<env.ecfg.max_material_overlap
        assert success[0] and not safety[0] and env.stable[0]==11
    finally:env.close()


def test_cached_mesh_query_matches_independent_mesh_distance(physical_model):
    d=geometric_scene(physical_model,'retainer');q=int(physical_model.joint('retainer_free').qposadr[0])
    d.qpos[q:q+3]-=d.body('base').xmat.reshape(3,3)@np.array([0,0,.00055]);mujoco.mj_forward(physical_model,d)
    import igl
    with np.load(scene.ASSETS/'parts/base.npz') as mesh:v,f=mesh['vertices'],mesh['faces']
    with np.load(scene.ASSETS/'parts/retainer.npz') as mesh:
        vertices,faces=mesh['vertices'],mesh['faces']
        samples=np.vstack((vertices,vertices[faces].mean(axis=1)))
    a,b=d.body('retainer'),d.body('base')
    points=(samples@a.xmat.reshape(3,3).T+a.xpos-b.xpos)@b.xmat.reshape(3,3)
    inside=igl.fast_winding_number(v,f,points)>.5
    independent=float(np.sqrt(igl.point_mesh_squared_distance(points[inside],v,f)[0]).max())
    pairs=sampled_assembly_overlaps(d,['base','retainer','screw'],scene.ASSETS)
    assert set(pairs)=={'base/retainer','base/table','retainer/table'}
    assert pairs['base/retainer']>=independent-1e-12
    assert pairs['base/retainer']<.0008


def test_rigidly_translated_assembly_below_table_cannot_complete(physical_model):
    env=BatchedAssemblyEnv(['motor'],[0],EnvConfig(nthread=1),model_factory=lambda **_:physical_model)
    try:
        d=geometric_scene(physical_model,'motor')
        for name in ('base','motor'):
            q=int(physical_model.joint(name+'_free').qposadr[0]);d.qpos[q+2]-=.0015
        mujoco.mj_forward(physical_model,d);env.data[0]=d
        assert env.assembly_distance()[0]<env.ecfg.success_threshold
        pairs=sampled_assembly_overlaps(d,['base','motor'],scene.ASSETS)
        assert pairs['base/motor']<env.ecfg.max_material_overlap
        assert pairs['base/table']>env.ecfg.max_material_overlap
        success=np.array([True]);safety=np.array([False]);env.stable[:]=11
        overlap=env._check_material_success(success,safety)
        assert overlap[0]>env.ecfg.max_material_overlap
        assert not success[0] and safety[0] and env.stable[0]==0
    finally:env.close()


@pytest.mark.parametrize('depth,accepted',[(.00045,True),(.00055,False)])
def test_user_selected_half_mm_limit_on_actual_original_mesh(physical_model,depth,accepted):
    """Rigid translation preserves mating, so only the changed floor budget decides."""
    env=BatchedAssemblyEnv(['motor'],[0],EnvConfig(nthread=1),model_factory=lambda **_:physical_model)
    try:
        d=geometric_scene(physical_model,'motor')
        with np.load(scene.ASSETS/'parts/base.npz') as mesh:vertices=mesh['vertices']
        body=d.body('base');plane=d.geom('table')
        height=float((vertices@body.xmat.reshape(3,3).T+body.xpos)[:,2].min()-plane.xpos[2])
        for name in ('base','motor'):
            q=int(physical_model.joint(name+'_free').qposadr[0]);d.qpos[q+2]-=height+depth
        mujoco.mj_forward(physical_model,d);env.data[0]=d
        assert env.assembly_distance()[0]<env.ecfg.success_threshold
        measured=sampled_assembly_overlaps(d,['base','motor'],scene.ASSETS)['base/table']
        assert abs(measured-depth)<1e-10
        success=np.array([True]);safety=np.array([False]);env.stable[:]=11
        env._check_material_success(success,safety)
        assert bool(success[0])==accepted and bool(safety[0])!=accepted
    finally:env.close()
