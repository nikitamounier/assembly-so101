"""A recorded clear-drop failure must produce material-blocking native contacts."""
import mujoco
import numpy as np

from assembly_policy.so101 import scene
from assembly_policy.so101.robot import home
from assembly_policy.so101.contact_audit import sampled_motor_base_overlap

# Synthetic poses at the previously missed lip contact. Other bodies start at
# home; no generated run or private CAD is embedded.
POSES = {'base': [-0.005579671150251014, 0.11448659772149006, 0.04349783109457273, 0.7071030126453266, -0.7071073431014212, 0.001505603305139009, -0.0015059875962287056], 'motor': [-0.014418073959220632, 0.12232676192786572, 0.053659250506380096, 0.6339960747697921, -0.6646492544859391, -0.3409962453770131, -0.2000297636017201]}


def test_missed_base_lip_is_detected_in_both_geom_orders_without_pose_changes():
    m = scene.model(contact_monitor=True)
    d = mujoco.MjData(m)
    home(m, d)
    for name, pose in POSES.items():
        q = int(m.joint(name+'_free').qposadr[0])
        d.qpos[q:q+7] = pose
    initial = d.qpos.copy()
    base, motor = m.geom('base_contact').id, m.geom('motor_contact').id
    flag = m.numeric('so101_material_contact_guard').data
    flag[0] = 0
    mujoco.mj_forward(m, d)
    old_depth = max([0., *[-float(c.dist) for c in d.contact if set(c.geom) == {base,motor}]])
    material = sampled_motor_base_overlap(m, d.qpos, scene.ASSETS)
    assert material > .0008 and old_depth < .0001
    flag[0] = 1
    mujoco.mj_forward(m, d)
    contacts = [c for c in d.contact if set(c.geom) == {base,motor}]
    assert -min(c.dist for c in contacts) > .0006
    assert len(contacts) <= mujoco.mjMAXCONPAIR
    assert all(np.isfinite(c.frame).all() for c in contacts)
    assert all(abs(np.linalg.norm(c.frame[:3])-1) < 1e-10 for c in contacts)
    for first, second in ((base,motor),(motor,base)):
        assert mujoco.mj_geomDistance(m,d,first,second,.01,None) < -.0006
    np.testing.assert_array_equal(d.qpos, initial)
    assert not d.warning.number.any()
