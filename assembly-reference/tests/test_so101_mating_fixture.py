"""Force feedback must use the integrated state, not cached pre-step poses."""
import mujoco
import numpy as np
from scripts.check_so101_sliding import free_body_state


def test_free_body_feedback_matches_fresh_native_kinematics_after_integration():
    m=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".0005"/><worldbody><body name="part" pos="0 0 1"><freejoint/><geom type="box" size=".01 .02 .03" mass=".05"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);d.qpos[3:7]=[.7071067811865476,0,0,.7071067811865476]
    d.qvel[:]=[.1,.2,.3,1,2,3];mujoco.mj_forward(m,d);mujoco.mj_step(m,d)
    position,rotation,velocity=free_body_state(m,d,m.body('part').id)
    assert np.linalg.norm(position-d.body('part').xpos)>1e-6
    check=mujoco.MjData(m);check.qpos[:]=d.qpos;check.qvel[:]=d.qvel;mujoco.mj_forward(m,check)
    expected=np.zeros(6);mujoco.mj_objectVelocity(m,check,mujoco.mjtObj.mjOBJ_BODY,m.body('part').id,expected,0)
    np.testing.assert_allclose(position,check.body('part').xpos,atol=1e-12)
    np.testing.assert_allclose(rotation,check.body('part').xmat.reshape(3,3),atol=1e-12)
    np.testing.assert_allclose(velocity,expected,atol=1e-12)
