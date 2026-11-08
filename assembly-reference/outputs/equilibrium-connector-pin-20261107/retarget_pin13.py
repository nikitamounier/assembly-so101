"""Transfer actual stable upper-cylinder pinch; pin yaw is physically free."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'pin-retarget';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);vis=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes']
ref=mujoco.MjData(m)
with np.load(Path(__file__).parent/'pin-deeper/12_pin_pickup.npz')as z:mujoco.mj_setState(m,ref,z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,ref);site=m.site('left_gripperframe').id;body=m.body('assembly_pin').id
rp=ref.xmat[body].reshape(3,3);relp=rp.T@(ref.site_xpos[site]-ref.xpos[body]);relr=rp.T@ref.site_xmat[site].reshape(3,3)
d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/13_pin_aligned_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=13>=first
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();p0=d.xpos[body].copy();r0=d.xmat[body].reshape(3,3).copy();qids=[m.joint('left_'+n).qposadr[0]for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jids=[m.joint('left_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jaw=m.joint('left_gripper').qposadr[0];pinadr=m.joint('assembly_pin_free').qposadr[0];scratch=mujoco.MjData(vis);scratch.qpos[:]=q0
low=np.r_[m.jnt_range[jids,0]+1e-7,-np.pi];high=np.r_[m.jnt_range[jids,1]-1e-7,np.pi]
def fun(x):
 rr=r0@Rotation.from_rotvec([0,0,x[-1]]).as_matrix();ps=p0+rr@relp;rs=rr@relr;scratch.qpos[qids]=x[:5];mujoco.mj_kinematics(vis,scratch);r=scratch.site_xmat[site].reshape(3,3)
 return np.r_[1000*(scratch.site_xpos[site]-ps),3*(r[:,2]-rs[:,2]),3*(r[:,0]-rs[:,0]),.0001*(x[:5]-q0[qids])]
fits=[]
for phi in [-3.,-1.5,0.,1.5,3.]:
 fit=least_squares(fun,np.r_[q0[qids],phi],bounds=(low,high),max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-10);fits.append(fit);print('IK',phi,np.linalg.norm(fit.fun[:9]),fit.x.tolist(),flush=True)
fit=min(fits,key=lambda r:np.linalg.norm(r.fun[:9]));d.qpos[qids]=fit.x[:5];d.qpos[jaw]=ref.qpos[jaw];newr=r0@Rotation.from_rotvec([0,0,fit.x[-1]]).as_matrix();mujoco.mju_mat2Quat(d.qpos[pinadr+3:pinadr+7],newr.ravel());d.ctrl[:]=ctrl0
for ac in range(m.nu):d.ctrl[ac]=d.qpos[m.jnt_qposadr[m.actuator_trnid[ac,0]]]
mujoco.mj_forward(m,d)
for ac in range(m.nu):
 j=m.actuator_trnid[ac,0]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
d.ctrl[m.actuator('left_gripper').id]-=.0008
candidate=d.qpos.copy();candidatectrl=d.ctrl.copy();pairs=[('left','assembly_pin')];initial=d.xpos[body].copy();frozen=None;holdstate=None;hs=None;hr=None;minload={'fixed':1e9,'moving':1e9};maxmove=0.;maxrot=0.;trace=[];t0=time.monotonic()
for step in range(round(.6/m.opt.timestep)):
 if d.time>=.4 and frozen is None:
  frozen=d.ctrl.copy();hs=d.xpos[body].copy();hr=d.xmat[body].reshape(3,3).copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,pairs)['left']
 if frozen is not None:
  for k,val in loads.items():minload[k]=min(minload[k],val)
  maxmove=max(maxmove,float(np.linalg.norm(d.xpos[body]-hs)));r=d.xmat[body].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(r@hr.T)-1)/2,-1,1)))))
 if step%400==0:row={'time':float(d.time),'loads':loads,'pin_drift_m':float(np.linalg.norm(d.xpos[body]-initial))};trace.append(row);print(row,flush=True)
result={'stage':'13_pin_aligned','side':'left','part':'assembly_pin','model_sha256':p['model_sha256'],'qpos_candidate':candidate.tolist(),'ctrl_candidate':candidatectrl.tolist(),'pin_yaw_rotation_radians':float(fit.x[-1]),'pin_axis_preserved':bool(np.allclose(newr[:,2],r0[:,2])),'ik_residual':fit.fun[:9].tolist(),'hold_min_loads_N':minload,'hold_max_motion_m':maxmove,'hold_max_rotation_deg':maxrot,'total_object_drift_m':float(np.linalg.norm(d.xpos[body]-initial)),'controls_unchanged':np.array_equal(frozen,d.ctrl),'local_hold_pass':min(minload.values())>=.01 and maxmove<.00025 and maxrot<.25,'training_eligible':False,'trace':trace,'wall_seconds':time.monotonic()-t0}
(OUT/'13_pin_aligned.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'13_pin_aligned.npz',qpos_candidate=candidate,ctrl=candidatectrl,hold_state=holdstate,final_qpos=d.qpos,final_qvel=d.qvel);print('FINAL',result,flush=True)
