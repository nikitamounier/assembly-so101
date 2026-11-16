"""Bounded native pad capture search, then immutable-profile hold replay."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/"connector-strong";OUT.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['07_connector_pickup','08_connector_aligned','12_pin_pickup','13_pin_aligned']);ap.add_argument('--replay',action='store_true');a=ap.parse_args()
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);vis=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'))
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes']
original_margin=m.geom_margin.copy();results=[]
for stage in a.stages:
 side='right' if stage.startswith(('07','08')) else 'left';part='shoulder_connector' if side=='right' else 'assembly_pin'
 qids=[m.joint(side+'_'+n).qposadr[0]for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jids=[m.joint(side+'_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jaw=m.joint(side+'_gripper').qposadr[0];sid=m.site(side+'_gripperframe').id
 d=mujoco.MjData(m)
 with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage.split('_')[0])>=first
 mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();p0=d.site_xpos[sid].copy();r0=d.site_xmat[sid].reshape(3,3).copy();scratch=mujoco.MjData(vis);scratch.qpos[:]=q0
 m.geom_margin[:]=original_margin
 for g in range(m.ngeom):
  name=m.geom(g).name or '';src=v.grid_records.get(name.removesuffix('__contact'),{}).get('source_mesh_name','')
  if (m.geom_bodyid[g]==m.body(part).id and name.endswith('__contact'))or(src.startswith(side)and any(x in src for x in ['wrist_roll_follower','moving_jaw'])):m.geom_margin[g]=.005
 def place(shift,delta):
  target=p0+r0[:,2]*shift
  def fun(x):
   scratch.qpos[qids]=x;mujoco.mj_kinematics(vis,scratch);r=scratch.site_xmat[sid].reshape(3,3)
   return np.r_[1000*(scratch.site_xpos[sid]-target),2*(r[:,2]-r0[:,2]),r[:,0]-r0[:,0]]
  lo=np.maximum(m.jnt_range[jids,0]+1e-7,q0[qids]-.10);hi=np.minimum(m.jnt_range[jids,1]-1e-7,q0[qids]+.10)
  fit=least_squares(fun,np.clip(q0[qids],lo,hi),bounds=(lo,hi),max_nfev=35,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  d.qpos[:]=q0;d.qpos[qids]=fit.x;d.qpos[jaw]=q0[jaw]+delta;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
  return fit
 def gaps():
  gap={};details={}
  for i in range(d.ncon):
   ga,gb=map(int,d.contact.geom[i]);ba,bb=map(int,m.geom_bodyid[[ga,gb]])
   if ba==m.body(part).id:other=gb
   elif bb==m.body(part).id:other=ga
   else:continue
   src=v.grid_records.get(m.geom(other).name.removesuffix('__contact'),{}).get('source_mesh_name','')
   if not src.startswith(side)or not any(x in src for x in ['wrist_roll_follower','moving_jaw']):continue
   pad='moving'if'moving_jaw'in src else'fixed';r=d.site_xmat[sid].reshape(3,3);point=(d.contact.pos[i]-d.site_xpos[sid])@r
   normal=d.contact.frame[i,:3]*(1 if other==ga else-1);sign=1 if pad=='fixed'else-1
   if not(-.035<point[0]<.004 and abs(point[1])<.01 and sign*np.dot(normal,r[:,2])>.25):continue
   value=float(d.contact.dist[i])
   if value<gap.get(pad,1):gap[pad]=value;details[pad]={'local_point':point.tolist(),'normal_dot':float(np.dot(normal,r[:,2]))}
  return gap,details
 candidates=[]
 for shift in [0.,.0003,.0006,.0010]:
  for delta in [0.,-.01,-.02,-.03,-.04]:
   fit=place(shift,delta);gp,detail=gaps();row={'shift':shift,'jaw_delta':delta,'gaps':gp,'detail':detail,'q':d.qpos[qids+[jaw]].tolist(),'fit':float(np.linalg.norm(fit.fun))}
   candidates.append(row)
 score=lambda r:sum(abs(r['gaps'].get(pad,.01)+.000005)for pad in ['fixed','moving'])
 best=min(candidates,key=score);print(stage,'scanbest',best,flush=True)
 # Bounded two-parameter refinement uses geometric distances, not force/acceleration residuals.
 def geometric(x):
  place(x[0]*.001,x[1]*.01);gp,_=gaps();return np.array([gp.get(k,.01)+.000005 for k in ['fixed','moving']])*1000
 x0=[best['shift']/.001,best['jaw_delta']/.01]
 opt=least_squares(geometric,x0,bounds=([-.5,-8],[3,1]),diff_step=.01,max_nfev=40,xtol=1e-9,ftol=1e-9,gtol=1e-8)
 fit=place(opt.x[0]*.001,opt.x[1]*.01);gp,detail=gaps();q=d.qpos.copy();record={'stage':stage,'side':side,'part':part,'qpos':q.tolist(),'initial_qpos':q0.tolist(),'active_q':q[qids+[jaw]].tolist(),'shift_m':float(opt.x[0]*.001),'jaw_delta_rad':float(opt.x[1]*.01),'gaps':gp,'pad_details':detail,'geometric_residual':opt.fun.tolist(),'initial_scan':candidates,'model_sha256':p['model_sha256'],'training_eligible':False}
 m.geom_margin[:]=original_margin
 if a.replay:
  d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=ctrl0
  for ac in range(m.nu):d.ctrl[ac]=d.qpos[m.jnt_qposadr[m.actuator_trnid[ac,0]]]
  mujoco.mj_forward(m,d)
  # Existing bounded gravity preloads keep the real servos at intended pose.
  for ac in range(m.nu):
   j=m.actuator_trnid[ac,0]
   if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
  d.ctrl[m.actuator(side+'_gripper').id]-=.003
  pairs=[(side,part)];bid=m.body(part).id;initial=d.xpos[bid].copy();holdstate=None;hs=None;hr=None;minload={'fixed':1e9,'moving':1e9};maxmotion=0.;maxrot=0.;trace=[]
  frozen=None;t0=time.monotonic()
  for step in range(round(.6/m.opt.timestep)):
   if d.time>=.4 and frozen is None:
    frozen=d.ctrl.copy();hs=d.xpos[bid].copy();hr=d.xmat[bid].reshape(3,3).copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
   physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
   ld=v._loaded_pads(d,pairs)[side]
   if frozen is not None:
    for key,val in ld.items():minload[key]=min(minload[key],val)
    maxmotion=max(maxmotion,float(np.linalg.norm(d.xpos[bid]-hs)));r=d.xmat[bid].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(r@hr.T)-1)/2,-1,1)))))
   if step%400==0:trace.append({'time':float(d.time),'loads':ld,'object_drift_m':float(np.linalg.norm(d.xpos[bid]-initial))});print(stage,trace[-1],flush=True)
  record.update(hold_min_loads_N=minload,hold_max_motion_m=maxmotion,hold_max_rotation_deg=maxrot,total_object_drift_m=float(np.linalg.norm(d.xpos[bid]-initial)),controls_unchanged=np.array_equal(frozen,d.ctrl),replay_seconds=.6,hold_seconds=.2,wall_seconds=time.monotonic()-t0,trace=trace,local_hold_pass=min(minload.values())>=.01 and maxmotion<.00025 and maxrot<.25)
  np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=d.ctrl,hold_state=holdstate,final_qpos=d.qpos,final_qvel=d.qvel)
 (OUT/(stage+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.append(record);print(stage,'FINAL', {k:v for k,v in record.items()if k not in ['qpos','initial_qpos','initial_scan','trace']},flush=True)
(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
