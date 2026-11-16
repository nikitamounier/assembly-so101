"""Native counterexamples for curriculum contact/material and action gates."""
import numpy as np
import mujoco
import pytest

from assembly_policy.so101.curriculum import Builder, DEPTH, holding_action, material_overlaps
from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.task_scenes import geometric_scene


@pytest.fixture(scope='module')
def builder():
    result=Builder()
    yield result
    result.close()


def test_material_gate_rejects_a_near_goal_retainer_with_hidden_overlap(builder):
    """A near-goal pose can hide material overlap behind valid native contacts."""
    m=builder.m;d=geometric_scene(m,'retainer')
    q=int(m.joint('retainer_free').qposadr[0])
    d.qpos[q:q+3]-=d.body('base').xmat.reshape(3,3)@np.array([0,0,.0007])
    mujoco.mj_forward(m,d)
    assert -min(d.contact.dist,default=0.) < DEPTH
    reason,values=builder.inspect(d,dict(task='yoke',predecessors=['motor','horn','screw','retainer']))
    assert reason=='sampled_material_overlap'
    assert values['base/retainer']>DEPTH


def test_qualified_retainer_prefix_stays_valid_without_attachments(builder):
    """Completed prerequisites persist under free native dynamics."""
    m=builder.m;source=builder.receiver('yoke');d=mujoco.MjData(m)
    assert source['predecessors']==['motor','horn','screw','retainer']
    mujoco.mj_setState(m,d,source['state'],STATE_SPEC);d.ctrl[:]=source['controls']
    mujoco.mj_step(m,d,nstep=1040);mujoco.mj_forward(m,d)
    assert m.neq==0
    reason,values=builder.inspect(d,dict(task='yoke',predecessors=source['predecessors']))
    assert reason is None,values


def test_qualification_controller_uses_bounded_incremental_actions(builder):
    env=builder.env;q=env.state[:,1:13].copy();target=q.copy()
    target[:,0]+=10.;target[:,6]-=10.;target[:,[5,11]]=-.23
    a=holding_action(env,target)
    assert a.shape==(1,12)
    assert np.max(abs(a))<=1
    assert a[0,0]==1 and a[0,6]==-1
    assert np.all(a[:,[5,11]]<=0)


def test_sampling_is_repeatable_and_noise_changes_the_state(builder):
    a=builder.candidate('motor','pickup',2030101,0)
    repeat=builder.candidate('motor','pickup',2030101,0)
    b=builder.candidate('motor','pickup',2030102,0)
    np.testing.assert_array_equal(a['state'],repeat['state'])
    assert not np.array_equal(a['state'],b['state'])
    assert max(abs(np.array(a['noise']['xy_m'])))<=.003
    assert abs(a['noise']['yaw_rad'])<=np.deg2rad(2)
    assert max(abs(np.array(a['noise']['grasp_translation_local_m'])))<=.00015


def test_material_gate_detects_table_intersection(builder):
    m=builder.m;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,builder.receiver('motor')['state'],STATE_SPEC)
    q=int(m.joint('motor_free').qposadr[0]);d.qpos[q+2]-=.05
    mujoco.mj_kinematics(m,d)
    values=material_overlaps(d,('motor',))
    assert values['motor/table']>.01


def test_thread_geometry_distinguishes_engagement_from_nearby_screw(builder):
    from assembly_policy.so101.curriculum import set_pose, thread_engagement
    from assembly_policy.so101.fastener import bearing_pose
    from scipy.spatial.transform import Rotation
    m=builder.m;d=geometric_scene(m,'screw')
    p,q=bearing_pose(m,d,builder.meta,preload_turns=0.)
    R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    set_pose(m,d,'screw',p,R);mujoco.mj_forward(m,d)
    assert thread_engagement(m,d)['engaged']
    set_pose(m,d,'screw',p+.01*R[:,2],R);mujoco.mj_forward(m,d)
    assert not thread_engagement(m,d)['engaged']
    set_pose(m,d,'screw',p+.0004*R[:,0],R);mujoco.mj_forward(m,d)
    assert not thread_engagement(m,d)['engaged']


def test_fork_envelope_distinguishes_entry_from_separated_parts(builder):
    from assembly_policy.so101.curriculum import set_pose, yoke_envelope_overlap
    from assembly_policy.so101.workcell import configuration
    m=builder.m;d=geometric_scene(m,'yoke');mujoco.mj_kinematics(m,d)
    p=d.body('yoke').xpos.copy();R=d.body('yoke').xmat.reshape(3,3).copy()
    axis=d.body('base').xmat.reshape(3,3)@np.asarray(configuration()['assembly_plan']['extractions']['yoke'])
    axis/=np.linalg.norm(axis)
    assert yoke_envelope_overlap(d)>.001
    set_pose(m,d,'yoke',p+.1*axis,R);mujoco.mj_kinematics(m,d)
    assert yoke_envelope_overlap(d)==0.


def test_prepared_bank_rejects_changed_validator_but_legacy_bank_loads(tmp_path,monkeypatch):
    import json
    from assembly_policy.so101 import resets
    monkeypatch.setattr(resets,'signature',lambda:'model')
    monkeypatch.setattr(resets,'validation_identity',lambda:{'env.py':'current'})
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps(dict(model_signature='model')))
    resets.check_bank(tmp_path)

    p.write_text(json.dumps(dict(model_signature='model',validator_identity={'env.py':'old'})))
    with pytest.raises(ValueError,match='Stale prepared-bank validation'):
        resets.check_bank(tmp_path)
    p.write_text(json.dumps(dict(model_signature='model',validator_identity={'env.py':'current'})))
    resets.check_bank(tmp_path)


def test_merge_refuses_stale_and_incomplete_curricula(tmp_path,monkeypatch):
    import hashlib
    import json
    from scripts import merge_so101_curriculum as merger
    source=tmp_path/'source';source.mkdir();parent=tmp_path/'parent'
    monkeypatch.setattr(merger,'check_bank',lambda _:None)
    monkeypatch.setattr(merger,'directory_signature',lambda _:'parent')
    monkeypatch.setattr(merger,'environment_identity',lambda:{'model':'current'})
    monkeypatch.setattr(merger,'validation_identity',lambda:{'env.py':'current'})
    report=dict(environment={'model':'current'},parent_directory_signature='parent',
                validator_identity={'env.py':'old'},diagnostic_timestep_s=.0005)
    receipt=source/'qualification.json';receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='fresh replay'):
        merger.merge([source],parent,tmp_path/'output')
    report.update(validator_identity={'env.py':'current'},generator_sha256=hashlib.sha256(b'generator').hexdigest())
    receipt.write_text(json.dumps(report));(source/'generator_source.py').write_bytes(b'generator')
    for task in merger.ASSEMBLY_TASKS:(source/f'{task}_provenance.json').write_text('[]')
    with pytest.raises(ValueError,match='Missing qualified coverage'):
        merger.merge([source],parent,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_measured_grasp_ik_cannot_move_receiver_to_solve_reach(builder,monkeypatch):
    """Reject an unreachable grasp rather than adding receiver pose variables."""
    from types import SimpleNamespace
    from assembly_policy.so101 import curriculum
    dimensions=[]
    def trial(fun,x,bounds,max_nfev):
        dimensions.append(len(x))
        return SimpleNamespace(x=x)
    monkeypatch.setattr(curriculum,'least_squares',trial)
    monkeypatch.setattr(builder,'qualify',lambda item:(item,{}))
    item=builder.candidate('motor','pickup',2030101,0)
    result,_=builder.place_grasp(item,'near',2030102)
    assert dimensions==[5]*len(dimensions) and dimensions
    if result is not None:
        assert result['preparation_base_yaw_rad']==0.
        assert result['preparation_base_xy_m']==[0.,0.]


def test_ready_arm_proposal_survives_saved_state_qualification(builder):
    item=builder.loose_candidate('motor',81000000)
    item,preparation=builder.prepare_ready(item)
    assert item is not None,preparation
    assert item['ready_arm']['sides']==['left','right']
    saved=item['state'].copy()
    result,receipt=builder.qualify(item,warmup=False)
    assert result is not None,receipt
    np.testing.assert_array_equal(result['state'],saved)
    checks=receipt['ready_arm_checks']
    assert len(checks)==26 and all(check['ok'] for check in checks)


def test_grasp_side_override_preserves_default_variant(builder,monkeypatch):
    default=builder.candidate('motor','pickup',2030101,0)
    explicit=builder.candidate('motor','pickup',2030101,0,side_override=default['side'])
    np.testing.assert_array_equal(default['state'],explicit['state'])
    assert explicit['variant']==default['variant']==0
    opposite='left' if default['side']=='right' else 'right'
    class CapturedSide(Exception):pass
    def capture(data,side,point,rotation):
        assert side==opposite
        raise CapturedSide
    monkeypatch.setattr(builder,'grasp_ik',capture)
    with pytest.raises(CapturedSide):
        builder.candidate('motor','pickup',2030101,0,side_override=opposite)
    with pytest.raises(ValueError,match='side override'):
        builder.candidate('motor','pickup',2030101,0,side_override='invalid')


def test_candidate_axis_gap_changes_mating_depth_without_moving_receiver(builder,monkeypatch):
    captured=[]
    class CapturedPose(Exception):pass
    def capture(data,side,point,rotation):
        captured.append((data.body('motor').xpos.copy(),data.body('base').xpos.copy(),
                         data.body('base').xmat.reshape(3,3).copy()))
        raise CapturedPose
    monkeypatch.setattr(builder,'grasp_ik',capture)
    for gap in (None,.02):
        with pytest.raises(CapturedPose):
            builder.candidate('motor','near',2030101,0,axis_gap=gap)
    from assembly_policy.so101.workcell import configuration
    axis=captured[0][2]@np.asarray(configuration()['assembly_plan']['extractions']['motor'],float)
    axis/=np.linalg.norm(axis)
    np.testing.assert_allclose(captured[0][0]-captured[1][0],.02*axis,atol=1e-12)
    np.testing.assert_array_equal(captured[0][1],captured[1][1])
    for gap in (-1.,float('nan')):
        with pytest.raises(ValueError,match='Axis gap'):
            builder.candidate('motor','near',2030101,0,axis_gap=gap)
    with pytest.raises(ValueError,match='Axis gap'):
        builder.candidate('motor','pickup',2030101,0,axis_gap=.02)


def test_fixed_part_pose_survives_approximate_grasp_ik(builder,monkeypatch):
    captured={}
    def approximate(data,side,point,rotation):
        captured['position']=data.body('motor').xpos.copy()
        captured['rotation']=data.body('motor').xmat.reshape(3,3).copy()
        start=0 if side=='left' else 6
        return data.qpos[start:start+5].copy()
    monkeypatch.setattr(builder,'grasp_ik',approximate)
    item=builder.candidate('motor','partial',2030101,0,preserve_part_pose=True)
    data=mujoco.MjData(builder.m);mujoco.mj_setState(builder.m,data,item['state'],STATE_SPEC)
    mujoco.mj_kinematics(builder.m,data)
    np.testing.assert_allclose(data.body('motor').xpos,captured['position'],atol=1e-12)
    np.testing.assert_allclose(data.body('motor').xmat.reshape(3,3),captured['rotation'],atol=1e-12)
    assert item['noise']['kinematic_initialization_shift_m']==[0.,0.,0.]
    with pytest.raises(ValueError,match='Fixed part pose'):
        builder.candidate('motor','pickup',2030101,0,preserve_part_pose=True)
