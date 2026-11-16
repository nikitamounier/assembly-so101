import json,mujoco,numpy as np
from pathlib import Path
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'hold22-farbracket';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);z=np.load(ROOT/'outputs/equilibrium-reset-20261107/minimum-v6-source-fingers/22_upper_half.npz');d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION);q0=d.qpos.copy();c0=d.ctrl.copy();mujoco.mj_kinematics(m,d);b=m.body('target_servo_2_pivot').id;upper=m.body('upper_arm_module').id;cp=d.xpos[b].copy();cr=d.xmat[b].reshape(3,3).copy();h=m.joint('target_servo_2_shaft').qposadr[0];qa=m.joint('upper_arm_module_free').qposadr[0];h0=float(d.qpos[h]);rows=[]
for delta in [-.03,-.06,-.09,.03]:
 d.qpos[:]=q0;d.qpos[h]=h0+delta;mujoco.mj_kinematics(m,d);rp=d.xpos[upper].copy();rr=d.xmat[upper].reshape(3,3).copy();change=cr@d.xmat[b].reshape(3,3).T;d.qpos[qa:qa+3]=cp+change@(rp-d.xpos[b]);mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(change@rr).ravel());d.qvel[:]=0;d.ctrl[:]=c0;mujoco.mj_forward(m,d);contacts=[]
 for con in d.contact:
  ga,gb=map(int,con.geom);an,bn=m.geom(ga).name or '',m.geom(gb).name or ''
  if not any(n.startswith('upper_arm_module_printed')for n in [an,bn]):continue
  # Skip own bearing contacts; source surfaces retain original filtering.
  if all(n.startswith('upper_arm_module')or n.startswith('servo_2')for n in [an,bn]):continue
  if con.dist<.001:contacts.append({'pair':[an,bn],'dist':float(con.dist),'pos':con.pos.tolist()})
 row={'stage':'21_upper_far_side','registered_model':True,'qpos':d.qpos.tolist(),'ctrl':d.ctrl.tolist(),'bracket_delta_rad':delta,'motor_pose_error_m':float(np.linalg.norm(d.xpos[b]-cp)),'motor_rotation_error':float(np.linalg.norm(d.xmat[b].reshape(3,3)-cr)),'contacts':sorted(contacts,key=lambda x:x['dist'])[:12]};rows.append(row);(OUT/(str(delta)+'.json')).write_text(json.dumps(row,indent=2));print(delta,row['motor_pose_error_m'],row['contacts'][:3],flush=True)
(OUT/'candidates.json').write_text(json.dumps(rows,indent=2))
