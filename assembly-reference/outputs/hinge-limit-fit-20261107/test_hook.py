from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_hinge_limits import install,JOINTS
xml='''<mujoco><option timestep=".0005" integrator="implicitfast" gravity="0 0 0"/><size nuserdata="4"/><worldbody><body><joint name="target_servo_2_shaft" type="hinge" limited="true" range="-.1 .1" damping=".02"/><geom type="box" size=".01 .02 .03" mass="1" contype="0"/><body pos="0 0 .1"><joint name="target_servo_3_shaft" type="hinge" limited="true" range="-.1 .1" damping=".02"/><geom type="box" size=".01 .02 .03" mass="1" contype="0"/></body></body></worldbody></mujoco>'''
# XML joint ranges are degrees by default; use explicit radians.
xml=xml.replace('<option','<compiler angle="radian"/><option')
m=mujoco.MjModel.from_xml_string(xml);lib=install(m,Path(__file__).parent/'hook-build',offset=0)
original=m.jnt_range.copy()
def check(sign):
 reference=mujoco.MjModel.from_xml_string(xml);a=mujoco.MjData(m);b=mujoco.MjData(reference)
 delta=np.array([sign*.0005,-sign*.0003,-sign*.0004,sign*.0006]);a.userdata[:]=delta
 reference.jnt_range[:]+=delta.reshape(2,2)
 for d in [a,b]:d.qpos[:]=[.0999,-.0999];d.qvel[:]=[1,-1]
 for _ in range(300):
  for model,data in [(m,a),(reference,b)]:data.qfrc_applied[:]=[.002,-.003];mujoco.mj_step(model,data)
  np.testing.assert_array_equal(a.qpos,b.qpos);np.testing.assert_array_equal(a.qvel,b.qvel)
 return True
try:
 with ThreadPoolExecutor(max_workers=2) as pool:assert all(pool.map(check,[-1,1]))
 np.testing.assert_array_equal(m.jnt_range,original)
 print('PASS:600steps,two simultaneous noisy worlds match separate native range models exactly; shared range unchanged.')
finally:lib.wowrobo_remove_hinge_limits()
