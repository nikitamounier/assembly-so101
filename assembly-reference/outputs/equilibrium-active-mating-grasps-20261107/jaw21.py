import json,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'jaw21';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];src=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text())['21_upper_far_side'];jaw=m.joint('right_gripper').qposadr[0];ac=m.actuator('right_gripper').id;bid=m.body('target_servo_2_pivot').id;sid=m.site('right_gripperframe').id
results=[]
for delta in [-.005,-.010,-.020,-.030]:
 d=mujoco.MjData(m);d.qpos[:]=src['qpos'];d.ctrl[:]=src['ctrl'];d.qpos[jaw]+=delta
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=21>=first
 mujoco.mj_forward(m,d)
 for ai in range(m.nu):
  j=m.actuator_trnid[ai,0]
  if m.joint(j).name.startswith('right_'):d.ctrl[ai]=np.clip(d.qpos[m.jnt_qposadr[j]]+(0 if ai==ac else d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ai,0]),*m.actuator_ctrlrange[ai])
 d.ctrl[ac]-=.0005;initialq=d.qpos.copy();initialctrl=d.ctrl.copy();hold=None;both=0;count=0;motion=0;angle=0;minimum={'fixed':1e9,'moving':1e9};trace=[]
 for step in range(round(.3/m.opt.timestep)):
  if d.time<.1:
   loads=v._loaded_pads_source_extent(d,[('right','target_servo_2_pivot')])['right']
   if min(loads.values())<.01:d.ctrl[ac]-=.03*m.opt.timestep
  elif hold is None:
   hold=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,hold,mujoco.mjtState.mjSTATE_INTEGRATION);R=d.site_xmat[sid].reshape(3,3);P=R.T@(d.xpos[bid]-d.site_xpos[sid]);B=R.T@d.xmat[bid].reshape(3,3)
  physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads_source_extent(d,[('right','target_servo_2_pivot')])['right']
  if hold is not None:
   both+=min(loads.values())>=.01;count+=1
   for key,value in loads.items():minimum[key]=min(minimum[key],value)
   R=d.site_xmat[sid].reshape(3,3);motion=max(motion,float(np.linalg.norm(R.T@(d.xpos[bid]-d.site_xpos[sid])-P)));angle=max(angle,float(np.rad2deg(np.arccos(np.clip((np.trace((R.T@d.xmat[bid].reshape(3,3))@B.T)-1)/2,-1,1)))))
  if step%400==0:trace.append({'time':d.time,'loads':loads})
 row={'delta':delta,'bilateral_fraction':both/count,'motion_m':motion,'angle_deg':angle,'minimum':minimum,'trace':trace};results.append(row);label=str(abs(delta));np.savez_compressed(OUT/(label+'.npz'),qpos_candidate=initialq,ctrl=initialctrl,hold_state=hold,final_qpos=d.qpos);(OUT/(label+'.json')).write_text(json.dumps(row,indent=2));print(row,flush=True)
 if both/count>=.8 and motion<.005 and angle<10:break
(OUT/'results.json').write_text(json.dumps(results,indent=2))
