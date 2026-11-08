"""Floor grasp acquisition scope: unchanged LEFT and final module physics only."""
import json,time,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'floor26-isolated';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
r=ET.parse(p['model_path']).getroot();active=set()
def identify(body,include=False):
 include=include or body.get('name')in('left_base','forearm_module')
 for g in body.findall('geom'):
  if include:active.add(g.get('name'))
  else:g.set('contype','0');g.set('conaffinity','0')
 for child in body.findall('body'):identify(child,include)
for body in r.find('worldbody').findall('body'):identify(body)
contact=r.find('contact')
for pair in list(contact.findall('pair')):
 if not{pair.get('geom1'),pair.get('geom2')}<=active:contact.remove(pair)
scene=OUT/'scene.xml';ET.ElementTree(r).write(scene,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/26_final_floor_grasp_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
d.eq_active[:]=False;mujoco.mj_forward(m,d)
for ac in range(m.nu):
 j=m.actuator_trnid[ac,0]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','target_servo_3_pivot')];ac=m.actuator('left_gripper').id;initialctrl=d.ctrl[ac];minload={'fixed':1e9,'moving':1e9};hs=None;hr=None;maxmotion=0.;maxrot=0.;records=[];holdstate=None;started=time.monotonic()
for step in range(round(1./m.opt.timestep)):
 if d.time<.8:
  loads=v._loaded_pads(d,pairs)['left']
  if min(loads.values())<.1:d.ctrl[ac]=max(initialctrl-.12,d.ctrl[ac]-.4*m.opt.timestep)
 elif hs is None:
  hs=d.xpos[m.body('target_servo_3_pivot').id].copy();hr=d.xmat[m.body('target_servo_3_pivot').id].reshape(3,3).copy();frozen=d.ctrl.copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,pairs)['left']
 if hs is not None:
  for k,x in loads.items():minload[k]=min(minload[k],x)
  maxmotion=max(maxmotion,float(np.linalg.norm(d.xpos[m.body('target_servo_3_pivot').id]-hs)));rr=d.xmat[m.body('target_servo_3_pivot').id].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(rr@hr.T)-1)/2,-1,1)))))
 if step%400==0:records.append({'time':float(d.time),'loads':loads});print(records[-1],flush=True)
result={'scope':'LEFT arm + free final module + table; unrelated contacts disabled, not full-scene admission','source_profile':p,'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'minimum_loads_N':minload,'max_case_hold_motion_m':maxmotion,'max_case_hold_rotation_deg':maxrot,'local_hold_pass':min(minload.values())>=.01 and maxmotion<.00025 and maxrot<.25,'controls_unchanged':np.array_equal(frozen,d.ctrl),'training_eligible':False,'wall_seconds':time.monotonic()-started,'trace':records}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'template.npz',hold_state=holdstate,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl);print('FINAL',result['local_hold_pass'],minload,maxmotion,maxrot,flush=True)
