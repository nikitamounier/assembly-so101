"""Extract exact stored hold endpoints; no simulation or velocity clearing."""
import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_equilibrium import rotation_degrees
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'))
rows={}
for label in ['baseline-v4','baseline-gravity-v4']:
 z=np.load(OUT/(label+'.npz'));before=mujoco.MjData(m);after=mujoco.MjData(m)
 # INTEGRATION begins with unchanged FULLPHYSICS fields in MuJoCo's bit order.
 n=mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_FULLPHYSICS)
 mujoco.mj_setState(m,before,z['hold_state'][:n],mujoco.mjtState.mjSTATE_FULLPHYSICS)
 after.qpos[:]=z['qpos'];after.qvel[:]=z['qvel'];before.ctrl[:]=z['ctrl'];after.ctrl[:]=z['ctrl']
 mujoco.mj_kinematics(m,before);mujoco.mj_kinematics(m,after)
 names=['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module','target_servo_2_pivot','target_servo_3_pivot','target_servo_4_pivot','left_gripper','right_gripper']
 poses={}
 for name in names:
  b=m.body(name).id;poses[name]={'endpoint_translation_m':float(np.linalg.norm(after.xpos[b]-before.xpos[b])),'endpoint_rotation_degrees':rotation_degrees(after.xmat[b].reshape(3,3),before.xmat[b].reshape(3,3))}
 dominant=[{'dof':int(i),'joint':m.joint(int(m.dof_jntid[i])).name,'value':float(after.qvel[i])}for i in np.argsort(abs(after.qvel))[-10:][::-1]]
 # Compare actual cover pose in actual gripper frame, not just world hand motion.
 def rel(data,side,name):
  site=m.site(side+'_gripperframe').id;b=m.body(name).id;r=data.site_xmat[site].reshape(3,3);return r.T@(data.xpos[b]-data.site_xpos[site]),r.T@data.xmat[b].reshape(3,3)
 p0,r0=rel(before,'left','base_side_cover');p1,r1=rel(after,'left','base_side_cover')
 rows[label]={'hold_start_time':float(before.time),'endpoint_motion_only_not_maximum':True,'poses':poses,'left_cover_relative_endpoint_translation_m':float(np.linalg.norm(p1-p0)),'left_cover_relative_endpoint_rotation_degrees':rotation_degrees(r1,r0),'dominant_final_dofs':dominant}
(OUT/'hold-endpoints.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
