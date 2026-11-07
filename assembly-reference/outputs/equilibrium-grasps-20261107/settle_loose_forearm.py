"""Isolated free loose-module equilibrium library; no fixed module DOFs."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_equilibrium import rotation_degrees
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'loose-forearm-v1';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
full=mujoco.MjModel.from_xml_path(p['model_path']);z=np.load(Path(__file__).parent/'baseline-v4.npz')
r=ET.parse(p['model_path']).getroot();r.set('model','loose_forearm_equilibrium_diagnostic');w=r.find('worldbody')
for node in list(w):
 if node.tag=='body' and node.get('name')!='forearm_module':w.remove(node)
keep={g.get('name')for g in w.iter('geom')};bodies={b.get('name')for b in w.iter('body')}
for tag in ['actuator','equality','keyframe','sensor','tendon']:
 for node in r.findall(tag):r.remove(node)
contact=r.find('contact')
for node in list(contact):
 if node.tag=='pair' and not{node.get('geom1'),node.get('geom2')}<=keep:contact.remove(node)
 elif node.tag=='exclude' and not{node.get('body1'),node.get('body2')}<=bodies:contact.remove(node)
scene=OUT/'scene.xml';ET.ElementTree(r).write(scene,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
assert m.nu==0 and m.neq==0 and m.nq==9 and m.nv==8
names=['forearm_module_free','target_servo_3_shaft','target_servo_4_shaft']
for name in names:
 j=full.joint(name);k=m.joint(name);nq=7 if name.endswith('_free') else 1;nv=6 if name.endswith('_free') else 1
 d.qpos[int(k.qposadr[0]):int(k.qposadr[0])+nq]=z['qpos'][int(j.qposadr[0]):int(j.qposadr[0])+nq]
 d.qvel[int(k.dofadr[0]):int(k.dofadr[0])+nv]=z['qvel'][int(j.dofadr[0]):int(j.dofadr[0])+nv]
mujoco.mj_forward(m,d);initialq=d.qpos.copy();initialv=d.qvel.copy();trace=[];window_start=None;maxtrans={};maxrot={};maxvel=0.;start=time.monotonic();passed=False
parts=['forearm_module','target_servo_3_pivot','target_servo_4_pivot'];friction=dict(finger_part=1.,part_table=1.,part_part=1.)
def pose():return{name:(d.xpos[m.body(name).id].copy(),d.xmat[m.body(name).id].reshape(3,3).copy())for name in parts}
def startwindow():
 global window_start,maxtrans,maxrot,maxvel
 window_start=pose();maxtrans={name:0. for name in parts};maxrot={name:0. for name in parts};maxvel=0.
startwindow()
for i in range(round(5./m.opt.timestep)):
 physical_split_step(m,d,friction)
 maxvel=max(maxvel,float(max(abs(d.qvel))))
 for name,(pos,rot)in pose().items():
  maxtrans[name]=max(maxtrans[name],float(np.linalg.norm(pos-window_start[name][0])));maxrot[name]=max(maxrot[name],rotation_degrees(rot,window_start[name][1]))
 if(i+1)%round(.5/m.opt.timestep)==0:
  row={'isolated_elapsed_time':float(d.time),'full_scene_equivalent_time':.8+float(d.time),'max_window_translation_m':maxtrans.copy(),'max_window_rotation_degrees':maxrot.copy(),'max_window_qvel':maxvel,'end_qvel':d.qvel.tolist(),'hinge_degrees':[float(np.rad2deg(d.joint(n).qpos[0]))for n in names[1:]],'min_contact_m':float(min(d.contact.dist[:d.ncon]))if d.ncon else None,'ncon':int(d.ncon)}
  passed=max(maxtrans.values())<.00025 and max(maxrot.values())<.25 and maxvel<.05
  row['equilibrium_pass']=passed;trace.append(row);print(row,flush=True)
  np.savez_compressed(OUT/'latest.npz',qpos=d.qpos,qvel=d.qvel,initial_qpos=initialq,initial_qvel=initialv,model_joint_names=np.asarray(names))
  receipt={'source_profile':p,'source_saved_state':str(Path(__file__).parent/'baseline-v4.npz'),'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'model_nq':m.nq,'model_nv':m.nv,'model_nu':m.nu,'model_neq':m.neq,'isolated_physical_independence_assumption':True,'module_fixed_dofs':False,'qvel_resets_during_integration':False,'external_forces':False,'hinge_angle_targets':False,'trace':trace,'equilibrium_pass':passed,'wall_seconds':time.monotonic()-start,'training_eligible':False}
  (OUT/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
  if passed:break
  startwindow()
