"""Coherent base-swivel reset search; cases/connector rotate together, no guides."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS,WowRoboPhysicalValidator,physical_split_step
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['21_upper_far_side','22_upper_half','23_upper_near']);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
p6=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());p5=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p6['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p6['model_path']);old=mujoco.MjModel.from_xml_path(p6['model_path']);t=mujoco.MjData(old)
with np.load(ROOT/'outputs/equilibrium-active-mating-grasps-20261107/original-v2/23_upper_near.npz')as z:t.qpos[:]=z['final_qpos'];t.ctrl[:]=z['ctrl']
mujoco.mj_kinematics(old,t);tb=old.body('target_servo_2_pivot').id;ts=old.site('right_gripperframe').id;tr=t.xmat[tb].reshape(3,3);relative_p=tr.T@(t.site_xpos[ts]-t.xpos[tb]);relative_r=tr.T@t.site_xmat[ts].reshape(3,3);template_jaw=float(t.joint('right_gripper').qpos[0]);template_ctrl_delta=float(t.ctrl[old.actuator('right_gripper').id]-template_jaw)
rows=json.loads((ROOT/'outputs/equilibrium-reset-20261107/minimum-initializers-v1.json').read_text());grid=json.loads(Path(p6['grid_manifest']).read_text())['meshes'];v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=grid;site=m.site('right_gripperframe').id;case=m.body('target_servo_2_pivot').id;names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint('right_'+n).id for n in names];qids=m.jnt_qposadr[jids];bounds=m.jnt_range[jids];jaw=m.joint('right_gripper').qposadr[0];margins=m.geom_margin.copy();results={}
for g in range(m.ngeom):
 name=m.geom(g).name or '';src=grid.get(name.removesuffix('__contact'),{}).get('source_mesh_name','')
 if m.geom_bodyid[g]==case and name.endswith('__contact') or src.startswith('right')and any(x in src for x in ['wrist_roll_follower','moving_jaw']):m.geom_margin[g]=.003

def update(d):mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
def gaps(d):
 gp={};details={};r=d.site_xmat[site].reshape(3,3)
 for c in d.contact:
  ga,gb=map(int,c.geom);ba,bb=map(int,m.geom_bodyid[[ga,gb]])
  if ba==case:other=gb
  elif bb==case:other=ga
  else:continue
  src=grid.get((m.geom(other).name or '').removesuffix('__contact'),{}).get('source_mesh_name','')
  if not src.startswith('right')or not any(x in src for x in ['wrist_roll_follower','moving_jaw']):continue
  pad='moving'if'moving_jaw'in src else'fixed';point=(c.pos-d.site_xpos[site])@r;normal=c.frame[:3]*(1 if other==ga else-1);sign=1 if pad=='fixed'else-1
  if -.035<point[0]<.004 and abs(point[1])<.01 and sign*(normal@r[:,2])>.25 and float(c.dist)<gp.get(pad,1):gp[pad]=float(c.dist);details[pad]={'point':point.tolist(),'normal_dot':float(normal@r[:,2])}
 return gp,details

def ik(d,pg,rg,start):
 def fun(x):
  d.qpos[qids]=x;mujoco.mj_kinematics(m,d);r=d.site_xmat[site].reshape(3,3);return np.r_[100*(d.site_xpos[site]-pg),2*(r[:,2]-rg[:,2]),.2*(r[:,0]-rg[:,0])]
 best=None
 for q in [start,np.array([start[0],.3,-.5,1.4,start[-1]]),np.array([start[0],.2,1.2,-1.5,start[-1]])]:
  sol=least_squares(fun,np.clip(q,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  score=np.linalg.norm(fun(sol.x))
  if best is None or score<best[0]:best=(score,sol.x.copy())
 fun(best[1]);r=d.site_xmat[site].reshape(3,3);return float(np.linalg.norm(d.site_xpos[site]-pg)),float(np.linalg.norm(r[:,2]-rg[:,2])),float(np.linalg.norm(r[:,0]-rg[:,0]))

for stage in args.stages:
 d=mujoco.MjData(m);qbase=np.asarray(rows[stage]['full_qpos']);d.qpos[:]=qbase;d.ctrl[:]=rows[stage]['controls'];
 for n,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(n).id]=int(stage.split('_')[0])>=first
 update(d);j1=m.joint('target_servo_1_shaft');axis=d.xaxis[j1.id].copy();anchor=d.xanchor[j1.id].copy();scans=[];chosen=None
 for yaw in [0,-5,5,-10,10,-15,15]:
  d.qpos[:]=qbase;change=Rotation.from_rotvec(axis*np.deg2rad(yaw)).as_matrix();mujoco.mj_kinematics(m,d)
  for part in ['shoulder_connector','assembly_pin','upper_arm_module']:
   b=m.body(part).id;pos=d.xpos[b].copy();rot=d.xmat[b].reshape(3,3).copy();a=m.joint(part+'_free').qposadr[0];d.qpos[a:a+3]=anchor+change@(pos-anchor);mujoco.mju_mat2Quat(d.qpos[a+3:a+7],(change@rot).ravel())
  d.qpos[j1.qposadr[0]]=qbase[j1.qposadr[0]]+np.deg2rad(yaw);mujoco.mj_kinematics(m,d);qyaw=d.qpos.copy();cp=d.xpos[case].copy();cr=d.xmat[case].reshape(3,3).copy();rg=cr@relative_r
  for slide in [-.016,-.020,-.024,0,-.008,.008]:
   d.qpos[:]=qyaw;d.qpos[jaw]=template_jaw;pg=cp+cr@(relative_p+np.array([0,0,slide]));err,span,approach=ik(d,pg,rg,qbase[qids]);update(d);gp,detail=gaps(d);row={'yaw_deg':yaw,'slide_m':slide,'ik_pos_error_m':err,'span_error':span,'approach_error':approach,'gaps':gp,'details':detail,'qpos':d.qpos.tolist()};scans.append(row)
   print(stage,yaw,slide,err,span,approach,gp,flush=True)
   if err<.0005 and span<.20 and all(k in gp and abs(gp[k])<.003 for k in ['fixed','moving']):chosen=row;break
  if chosen:break
 if chosen is None:results[stage]={'candidate':None,'scan':scans};print(stage,'NO_CANDIDATE',flush=True);continue
 # Fit only the jaw and a tiny hand shift around the achievable pose.
 d.qpos[:]=chosen['qpos'];update(d);capturebase=d.qpos.copy();sp=d.site_xpos[site].copy();sr=d.site_xmat[site].reshape(3,3).copy()
 def fit_gap(x):
  d.qpos[:]=capturebase;pg=sp+sr[:,2]*x[0]*.001;ik(d,pg,sr,capturebase[qids]);d.qpos[jaw]=capturebase[jaw]+x[1]*.01;update(d);gg,_=gaps(d);return np.array([gg.get(k,.01)+.00003 for k in ['fixed','moving']])*1000
 opt=least_squares(fit_gap,np.zeros(2),bounds=([-3,-8],[3,8]),diff_step=.02,max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8);fit_gap(opt.x);q=d.qpos.copy();gg,detail=gaps(d);m.geom_margin[:]=margins;mujoco.mj_forward(m,d)
 for a in range(m.nu):
  j=m.actuator_trnid[a,0];d.ctrl[a]=d.qpos[m.jnt_qposadr[j]]
  if not m.joint(j).name.endswith('_gripper'):d.ctrl[a]+=d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[a,0]
  d.ctrl[a]=np.clip(d.ctrl[a],*m.actuator_ctrlrange[a])
 d.ctrl[m.actuator('right_gripper').id]=np.clip(d.qpos[jaw]+template_ctrl_delta,*m.actuator_ctrlrange[m.actuator('right_gripper').id]);ctrl=d.ctrl.copy();record={'stage':stage,'yaw_deg':chosen['yaw_deg'],'slide_m':chosen['slide_m'],'qpos_candidate':q.tolist(),'ctrl_candidate':ctrl.tolist(),'gaps':gg,'gap_fit_residual':opt.fun.tolist(),'pad_details':detail,'source_profile':p6,'model_sha256':p6['model_sha256'],'template_control_delta':template_ctrl_delta,'training_eligible':False}
 if args.replay:
  loads_min={'fixed':1e9,'moving':1e9};trace=[];start=time.monotonic();initial_site=d.site_xpos[site].copy();frozen=None
  for i in range(round(.25/m.opt.timestep)):
   if d.time>=.05 and frozen is None:frozen=d.ctrl.copy()
   physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,[('right','target_servo_2_pivot')])['right']
   if frozen is not None:
    for k,val in loads.items():loads_min[k]=min(loads_min[k],val)
   if i%200==0:trace.append({'time':float(d.time),'loads':loads});print(stage,trace[-1],flush=True)
  record.update(minimum_loads_N=loads_min,controls_unchanged=np.array_equal(frozen,d.ctrl),replay_seconds=.25,hold_seconds=.2,final_qpos=d.qpos.tolist(),final_qvel=d.qvel.tolist(),wall_seconds=time.monotonic()-start,trace=trace,local_hold_pass=min(loads_min.values())>=.01)
  np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=ctrl,final_qpos=d.qpos,final_qvel=d.qvel)
 results[stage]={'candidate':record,'scan':scans};(OUT/(stage+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(stage,'RESULT',record.get('local_hold_pass'),record.get('minimum_loads_N'),flush=True)
 for g in range(m.ngeom):
  name=m.geom(g).name or '';src=grid.get(name.removesuffix('__contact'),{}).get('source_mesh_name','')
  if m.geom_bodyid[g]==case and name.endswith('__contact')or src.startswith('right')and any(x in src for x in ['wrist_roll_follower','moving_jaw']):m.geom_margin[g]=.003
(OUT/'diagonal-results.json').write_text(json.dumps(results,indent=2)+'\n')
