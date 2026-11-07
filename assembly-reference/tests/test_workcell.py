"""Shared scene, sideways resets, contact capacity and versioned training starts."""
import json
import ctypes
import mujoco
import numpy as np
import pytest
from assembly_policy.so101.scene import model,ROOT
from assembly_policy.so101.env import TASK_NAMES,STATE_SPEC
from assembly_policy.so101.resets import check_bank,reset
from assembly_policy.so101.workcell import configuration,default_bank_directory,assert_matches_reference
from assembly_policy.so101.contact_audit import sampled_motor_base_overlap

@pytest.fixture(scope='module')
def scene():return model(contact_monitor=True)

def test_compiled_physics_matches_frozen_shared_reference(scene):
    assert assert_matches_reference(scene)['passed']
    assert scene.nu==12 and scene.neq==0 and scene.ncam==3
    for part in (*TASK_NAMES,'base','driver'):
        assert scene.joint(part+'_free').type[0]==mujoco.mjtJoint.mjJNT_FREE

def test_search_capacity_and_contact_physics_are_shared(scene):
    cfg=configuration()
    assert ctypes.CDLL(None).assembly_contact_capacity_guard()==4
    assert scene.opt.sdf_initpoints==mujoco.mjMAXCONPAIR
    np.testing.assert_equal(scene.numeric('so101_sdf_search').data,[4,10])
    np.testing.assert_equal(scene.numeric('so101_motor_base_search').data,[40,40])
    np.testing.assert_equal(scene.numeric('so101_grasp_search').data,[16,20])
    for side in ('left','right'):
        np.testing.assert_allclose(scene.body(side+'_base').pos,cfg['mounts'][side]['position'])
        np.testing.assert_allclose(scene.actuator(side+'_gripper').forcerange,[-1,1])
        np.testing.assert_allclose(scene.actuator(side+'_gripper').ctrlrange,[-.23,.75])
        np.testing.assert_allclose(scene.actuator(side+'_shoulder_pan').forcerange,[-2.94,2.94])
    for name in ('motor_base_volume','base_rail_1_0_motor','base_rail_1_1_motor','base_rail_-1_0_motor','base_rail_-1_1_motor'):
        pair=scene.pair(name).id
        np.testing.assert_allclose(scene.pair_solref[pair],[.001,1])
        assert scene.pair_dim[pair]==3
    assert not scene.opt.enableflags&int(mujoco.mjtEnableBit.mjENBL_SLEEP)

@pytest.mark.parametrize('seed',[0,17,1000000])
def test_randomized_resets_retain_side_orientation_and_clear_geometry(scene,seed):
    state=reset(scene,'motor',seed);d=mujoco.MjData(scene)
    mujoco.mj_setState(scene,d,state,STATE_SPEC);mujoco.mj_forward(scene,d)
    axis=d.body('base').xmat.reshape(3,3)@np.array([0,1,0])
    assert axis[2]<-.999
    assert sampled_motor_base_overlap(scene,d.qpos,ROOT/'assets_local')<1e-6
    assert not d.warning.number.any()

def test_versioned_bank_has_independent_all_loose_starts():
    bank=default_bank_directory();check_bank(bank)
    manifest=json.loads((bank/'manifest.json').read_text())
    assert manifest['reset_distribution']=='all_loose'
    for task in TASK_NAMES:
        with np.load(bank/f'{task}_train.npz') as tr,np.load(bank/f'{task}_eval.npz') as ev:
            assert len(tr['states'])==len(ev['states'])==32
            assert not set(tr['seed'])&set(ev['seed'])
            assert set(tr['family'])==set(ev['family'])=={0}
            assert np.isfinite(tr['states']).all() and np.isfinite(ev['states']).all()

def test_old_bank_is_rejected_when_present():
    bank=ROOT/'data/so101/banks_quest_v2'
    if not bank.exists():pytest.skip('Previous local runtime is optional')
    with pytest.raises(ValueError,match='stale'):check_bank(bank)
