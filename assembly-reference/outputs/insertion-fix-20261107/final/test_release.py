from pathlib import Path
import json,numpy as np,mujoco,xml.etree.ElementTree as ET
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_final_fit import write_final_fit_scene
from scripts.diagnose_wowrobo_insertion import configure
ROOT=Path.cwd();out=ROOT/'outputs/insertion-fix-20261107/final';load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');set_surface_mode(1);a=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/final/alignment.json').read_text());axis=np.array(a['axis_world']);drive=np.load(out/'width-0.0004-drive.npz');width=.0004;scene=out/'candidate-guide.xml';write_final_fit_scene(ROOT/'outputs/legacy-fit-port-20261107/final/legacy_mapped_20N.xml',scene);guide=mujoco.MjModel.from_xml_path(str(scene));gd=mujoco.MjData(guide);gd.qpos[:]=drive['qpos'];gd.qvel[:]=drive['qvel'];mujoco.mj_forward(guide,gd);startp=gd.geom('servo_3_case').xpos.copy();startr=gd.geom('servo_3_case').xmat.copy();results=[]
for mode in ['free_root_locked_shape','free_root_internal_free']:
 if mode=='free_root_locked_shape':
  root=ET.parse(scene).getroot();body=root.find(".//body[@name='forearm_module']");bid=guide.body('forearm_module').id;body.set('pos',' '.join(map(str,gd.xpos[bid])));body.set('quat',' '.join(map(str,gd.xquat[bid])));
  for x in list(body):
   if x.tag in ['joint','freejoint']:body.remove(x)
  ET.SubElement(body,'freejoint',name='release_free');root.find('option').set('gravity','0 0 -9.81');path=out/(mode+'.xml');ET.ElementTree(root).write(path,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);d.qvel[:3]=float(drive['qvel'][0])*axis;frozen_q=np.zeros(m.nq,dtype=bool);frozen_v=np.zeros(m.nv,dtype=bool)
 else:
  path=out/(mode+'.xml');write_final_fit_scene(ROOT/'outputs/linear-insertion-updated-20261107/scene.xml',path);m=mujoco.MjModel.from_xml_path(str(path));m.opt.gravity[:]=[0,0,-9.81];d=mujoco.MjData(m);d.qpos[:]=a['start_qpos'];qa=int(m.joint('forearm_module_free').qposadr[0]);va=int(m.joint('forearm_module_free').dofadr[0]);d.qpos[qa:qa+3]+=float(drive['qpos'][0])*axis;d.qvel[va:va+3]=float(drive['qvel'][0])*axis;activeids={m.body(x).id for x in ['forearm_module','target_servo_3_pivot','target_servo_4_pivot']};frozen_q=np.ones(m.nq,dtype=bool);frozen_v=np.ones(m.nv,dtype=bool)
  for j in range(m.njnt):
   if int(m.jnt_bodyid[j]) in activeids:
    qa2=int(m.jnt_qposadr[j]);va2=int(m.jnt_dofadr[j]);free=m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE;frozen_q[qa2:qa2+(7 if free else 1)]=False;frozen_v[va2:va2+(6 if free else 1)]=False
  # Preexisting freebody review damping isnot a gripper or active externalfield; remove for free-root release.
  for j in range(m.njnt):
   if int(m.jnt_bodyid[j]) in activeids:
    va2=int(m.jnt_dofadr[j]);free=m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE;m.dof_damping[va2:va2+(6 if free else 1)]=0
  m.actuator_gainprm[:]=0;m.actuator_biasprm[:]=0
 active=configure(m,'forearm_module',['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module'],True)
 for g in range(m.ngeom):
  if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:
   m.geom_contype[g]=2;m.geom_conaffinity[g]=1;b=int(m.geom_bodyid[g]);m.body_contype[b]|=2;m.body_conaffinity[b]|=1
 mujoco.mj_forward(m,d);q0=d.qpos.copy();p0=d.body('forearm_module').xpos.copy();error=float(np.max(np.abs(d.geom('servo_3_case').xpos-startp)));peak=0;trace=[];reason='elapsed';initiald=d.geom('servo_3_case').xpos.copy()
 full_t=[0.0];full_q=[d.qpos.copy()];full_v=[d.qvel.copy()]
 for i in range(1000):
  d.qpos[frozen_q]=q0[frozen_q];d.qvel[frozen_v]=0;mujoco.mj_step(m,d);d.qpos[frozen_q]=q0[frozen_q];d.qvel[frozen_v]=0;mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth)
  full_t.append(float(d.time));full_q.append(d.qpos.copy());full_v.append(d.qvel.copy())
  if i%10==0:trace.append([float(d.time),*d.geom('servo_3_case').xpos.tolist(),depth,float(np.linalg.norm(d.qvel[~frozen_v]))])
  if depth>.0005:reason='penetrationguard';break
 np.savez_compressed(out/(mode+'-trajectory.npz'),time_s=np.array(full_t),qpos=np.array(full_q),qvel=np.array(full_v))
 r={'mode':mode,'actual_initialization_error_m':error,'drive_q_mm':float(drive['qpos'][0])*1000,'seconds':float(d.time),'case_displacement_mm':float(np.linalg.norm(d.geom('servo_3_case').xpos-initiald))*1000,'root_displacement_mm':float(np.linalg.norm(d.body('forearm_module').xpos-p0))*1000,'case_orientation_change_deg':float(np.rad2deg(np.arccos(np.clip((np.trace(d.geom('servo_3_case').xmat.reshape(3,3).T@startr.reshape(3,3))-1)/2,-1,1)))),'peak_native_mm':peak*1000,'last_vel_norm':float(np.linalg.norm(d.qvel[~frozen_v])),'stop':reason,'warning_count':int(sum(w.number for w in d.warning)),'trace':trace};results.append(r);np.savez_compressed(out/(mode+'-state.npz'),qpos=d.qpos,qvel=d.qvel);(out/(mode+'.json')).write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='trace'},flush=True)
(out/'release-validation.json').write_text(json.dumps(results,indent=2)+'\n')
