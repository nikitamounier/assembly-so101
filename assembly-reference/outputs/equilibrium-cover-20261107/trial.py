import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_equilibrium import gravity_preload_controls,integration_state,pose_snapshot,rotation_degrees
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--shift',type=float,required=True);ap.add_argument('--jaw',type=float,required=True);ap.add_argument('--name',required=True);ap.add_argument('--seconds',type=float,default=.7);ap.add_argument('--preload',type=float,default=.00025);a=ap.parse_args()
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);vis=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'))
d=mujoco.MjData(m)
with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{a.stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();sid=m.site('left_gripperframe').id;pos=d.site_xpos[sid].copy();rot=d.site_xmat[sid].reshape(3,3).copy();center0=d.xpos[m.body('base_side_cover').id].copy()
qs=[m.joint('left_'+n).qposadr[0]for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jids=[m.joint('left_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jaw=m.joint('left_gripper').qposadr[0];scratch=mujoco.MjData(vis);scratch.qpos[:]=q0;target=pos+rot[:,2]*a.shift

def fun(x):
 scratch.qpos[qs]=x;mujoco.mj_forward(vis,scratch);r=scratch.site_xmat[sid].reshape(3,3)
 return np.r_[1000*(scratch.site_xpos[sid]-target),2*(r[:,2]-rot[:,2]),r[:,0]-rot[:,0]]
fit=least_squares(fun,q0[qs],bounds=(np.maximum(m.jnt_range[jids,0]+1e-7,q0[qs]-.05),np.minimum(m.jnt_range[jids,1]-1e-7,q0[qs]+.05)),max_nfev=50)
d.qpos[qs]=fit.x;d.qpos[jaw]+=a.jaw
for ac in range(m.nu):d.ctrl[ac]=d.qpos[m.jnt_qposadr[m.actuator_trnid[ac,0]]]
gravity_preload_controls(m,d);d.ctrl[m.actuator('left_gripper').id]-=a.preload;mujoco.mj_forward(m,d)
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','base_side_cover')];start=time.monotonic();records=[];minimum={'fixed':1e9,'moving':1e9};holdstate=None;holdpose=None;drift=0;rd=0;holdvel=0;cid=m.body('base_side_cover').id
initialq=d.qpos.copy();initialctrl=d.ctrl.copy()
for step in range(round(a.seconds/m.opt.timestep)):
 if holdstate is None and d.time>=a.seconds-.2-1e-8:holdstate=integration_state(m,d);holdpose=(d.xpos[cid].copy(),d.xmat[cid].reshape(3,3).copy())
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
 ld=v._loaded_pads(d,pairs)['left']
 if holdstate is not None:
  for k,val in ld.items():minimum[k]=min(minimum[k],val)
  drift=max(drift,float(np.linalg.norm(d.xpos[cid]-holdpose[0])));rd=max(rd,rotation_degrees(d.xmat[cid].reshape(3,3),holdpose[1]));holdvel=max(holdvel,float(max(abs(d.qvel))))
 if step%400==0 or step==round(a.seconds/m.opt.timestep)-1:
  r={'time':float(d.time),'loads':ld,'cover_motion':(d.xpos[cid]-center0).tolist(),'cover_rotation':d.xquat[cid].tolist(),'jawq':float(d.qpos[jaw]),'jawctrl':float(d.ctrl[m.actuator('left_gripper').id]),'min_contact':float(min(d.contact.dist[:d.ncon])) if d.ncon else None};records.append(r);print(r,flush=True)
result={'stage':a.stage,'shift_m':a.shift,'jaw_delta':a.jaw,'preload_rad':a.preload,'profile':p,'left_q_initial':initialq[np.r_[qs,jaw]].tolist(),'controls_initial':initialctrl.tolist(),'minimum_loads_N':minimum,'cover_hold_motion_m':drift,'cover_hold_rotation_deg':rd,'cover_settle_motion_m':float(np.linalg.norm(holdpose[0]-center0)),'controls_unchanged':np.array_equal(initialctrl,d.ctrl),'hold_max_global_velocity':holdvel,'records':records,'wall_seconds':time.monotonic()-start,'training_eligible':False}
(OUT/f'{a.name}.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/f'{a.name}.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,initial_qpos=initialq,initial_ctrl=initialctrl,hold_state=holdstate,source_qpos=q0)
