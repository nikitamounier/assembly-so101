"""Workcell rotations must not misalign the motor-local thread fixture."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from scripts import so101_thread_check as check
from assembly_policy.so101.fastener import AXIS
from assembly_policy.so101.scene import metadata


@pytest.mark.parametrize('quaternion',['.7071067811865476 -.7071067811865476 0 0',
                                     '.7071067811865476 0 0 .7071067811865476'])
def test_thread_fixture_stays_aligned_after_a_workcell_rotation(monkeypatch,quaternion):
    build=check.build_xml
    def rotated(*args,**kwargs):
        root=ET.fromstring(build(*args,**kwargs))
        for part in ('motor','screw'):
            root.find(f"worldbody/body[@name='{part}']").set('quat',quaternion)
        return ET.tostring(root,encoding='unicode')
    monkeypatch.setattr(check,'build_xml',rotated)
    m,d=check.thread_model();mujoco.mj_kinematics(m,d)
    motor=d.body('motor');screw=d.body('screw')
    local=motor.xmat.reshape(3,3).T@(screw.xpos-motor.xpos)
    expected=AXIS-np.asarray(metadata()['motor']['origin'])+[0,0,.0045]
    np.testing.assert_allclose(local,expected,atol=1e-12)
    np.testing.assert_allclose(motor.xmat,screw.xmat,atol=1e-12)
    np.testing.assert_allclose(motor.xmat.reshape(3,3)[:,2],[0,0,1],atol=1e-12)
    np.testing.assert_equal(m.numeric('so101_sdf_search').data,m.numeric('so101_assembly_search').data)
