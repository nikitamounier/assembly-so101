"""Transfer the actual held state23 servo grip to higher insertion starts."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'from23-v1';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];source=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text());t=mujoco.MjData(m)
with np.load(Path(__file__).parent/'original-v2/23_upper_near.npz')as z:mujoco.mj_setState(m,t,z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_kinematics(m,t);sid=m.site('right_gripperframe').id;bid=m.body('target_servo_2_pivot').id;relative_p=t.xmat[bid].reshape(3,3).T@(t.site_xpos[sid]-t.xpos[bid]);relative_r=t.xmat[bid].reshape(3,3).T@t.site_xmat[sid].reshape(3,3);names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint('right_'+n).id for n in names];qids=m.jnt_qposadr[jids];jaw=m.joint('right_gripper').qposadr[0];lo=m.jnt_range[jids,0]+1e-7;hi=m.jnt_range[jids,1]-1e-7;results={}
for stage in ['21_upper_far_side','22_upper_half']:
 d=mujoco.MjData(m);d.qpos[:]=source[stage]['qpos'];d.ctrl[:]=source[stage]['ctrl'];q0=d.qpos.copy();ctrl0=d.ctrl.copy()
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage[:2])>=first
 mujoco.mj_kinematics(m,d);target_p=d.xpos[bid]+d.xmat[bid].reshape(3,3)@relative_p;target_r=d.xmat[bid].reshape(3,3)@relative_r;opening=float(t.qpos[jaw]);d.qpos[jaw]=opening
 def fun(q):
  d.qpos[qids]=q;mujoco.mj_kinematics(m,d);r=d.site_xmat[sid].reshape(3,3);return np.r_[1000*(d.site_xpos[sid]-target_p),10*(r[:,2]-target_r[:,2]),10*(r[:,0]-target_r[:,0])]
 fits=[]
 for x in [t.qpos[qids],q0[qids],np.array([0.,-.4,1.2,-.8,0.])]:
  f=least_squares(fun,np.clip(x,lo,hi),bounds=(lo,hi),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10);fits.append(f)
 best=min(fits,key=lambda f:np.linalg.norm(f.fun));fun(best.x)
 for name in names+['gripper']:
  ac=m.actuator('right_'+name).id;j=m.actuator_trnid[ac,0];qa=m.jnt_qposadr[j];d.ctrl[ac]=d.qpos[qa]+t.ctrl[ac]-t.qpos[qa]
 q=d.qpos.copy();ctrl=d.ctrl.copy();d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);print(stage,'IK',np.linalg.norm(best.fun),v._loaded_pads(d,[('right','target_servo_2_pivot')]),flush=True);hold=None;minimum={'fixed':1e9,'moving':1e9};both=0;count=0;maxrel=0.;maxrot=0.;trace=[];started=time.monotonic()
 for step in range(round(.3/m.opt.timestep)):
  if d.time>=.1 and hold is None:
   hold=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,hold,mujoco.mjtState.mjSTATE_INTEGRATION);frozen=d.ctrl.copy();fr=d.site_xmat[sid].reshape(3,3);refp=fr.T@(d.xpos[bid]-d.site_xpos[sid]);refr=fr.T@d.xmat[bid].reshape(3,3)
  physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,[('right','target_servo_2_pivot')])['right']
  if hold is not None:
   count+=1;both+=int(min(loads.values())>=.01)
   for k,x in loads.items():minimum[k]=min(minimum[k],x)
   fr=d.site_xmat[sid].reshape(3,3);rel=fr.T@(d.xpos[bid]-d.site_xpos[sid]);rr=fr.T@d.xmat[bid].reshape(3,3);maxrel=max(maxrel,float(np.linalg.norm(rel-refp)));maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(rr@refr.T)-1)/2,-1,1)))))
  if step%400==0:trace.append({'time':float(d.time),'loads':loads});print(stage,trace[-1],flush=True)
 r={'stage':stage,'registered_model':True,'qpos':q.tolist(),'ctrl':ctrl.tolist(),'source_template':'original-v2/23_upper_near hold_state','IK_residual':best.fun.tolist(),'bilateral_fraction':both/count,'minimum_loads_N':minimum,'relative_case_motion_m':maxrel,'relative_case_rotation_deg':maxrot,'held_retention_pass':both/count>=.8 and maxrel<.005 and maxrot<10,'controls_unchanged':np.array_equal(frozen,d.ctrl),'source_profile_sha256':p['model_sha256'],'training_eligible':False,'trace':trace,'wall_seconds':time.monotonic()-started};np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=ctrl,hold_state=hold,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(r,indent=2)+'\n');results[stage]=r;print(stage,'FINAL',r['held_retention_pass'],r['bilateral_fraction'],maxrel,maxrot,flush=True)
(OUT/'recipes.json').write_text(json.dumps(results,indent=2)+'\n')
