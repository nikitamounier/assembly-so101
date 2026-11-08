"""Keep coherent motor/pocket geometry and original hand orientation; capture jaws."""
import json,time,argparse
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'original-v2';OUT.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--stages',nargs='+',default=['21_upper_far_side','22_upper_half','23_upper_near']);args=a.parse_args()
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];records=v.grid_records;source=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text());results={}
for stage in args.stages:
 side='right';part='target_servo_2_pivot';bid=m.body(part).id;sid=m.site(side+'_gripperframe').id;names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint(side+'_'+x).id for x in names];qids=m.jnt_qposadr[jids];jaw=m.joint(side+'_gripper').qposadr[0];d=mujoco.MjData(m);d.qpos[:]=source[stage]['qpos'];d.ctrl[:]=source[stage]['ctrl']
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage[:2])>=first
 mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();p0=d.site_xpos[sid].copy();r0=d.site_xmat[sid].reshape(3,3).copy();margin=m.geom_margin.copy()
 for g in range(m.ngeom):
  src=records.get((m.geom(g).name or '').removesuffix('__contact'),{}).get('source_mesh_name','')
  if m.geom_bodyid[g]==bid or(src.startswith(side)and any(k in src for k in ['wrist_roll_follower','moving_jaw'])):m.geom_margin[g]=.012
 lo=np.maximum(m.jnt_range[jids,0]+1e-6,q0[qids]-.25);hi=np.minimum(m.jnt_range[jids,1]-1e-6,q0[qids]+.25);cache={}
 def hand(shift,deep):
  key=(round(shift,10),round(deep,10))
  if key in cache:return cache[key]
  target=p0+r0[:,2]*shift+r0[:,0]*deep
  def residual(q):
   d.qpos[:]=q0;d.qpos[qids]=q;mujoco.mj_kinematics(m,d);r=d.site_xmat[sid].reshape(3,3);return np.r_[1000*(d.site_xpos[sid]-target),1.0*(r[:,2]-r0[:,2]),.10*(r[:,0]-r0[:,0])]
  fit=least_squares(residual,np.clip(q0[qids],lo,hi),bounds=(lo,hi),max_nfev=45,ftol=1e-9,xtol=1e-9,gtol=1e-8);cache[key]=fit.x.copy();return fit.x
 def place(shift,deep,delta):
  q=hand(shift,deep);d.qpos[:]=q0;d.qpos[qids]=q;d.qpos[jaw]=q0[jaw]+delta;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
 def gaps():
  gp={};details={}
  for ii in range(d.ncon):
   ga,gb=map(int,d.contact.geom[ii]);ba,bb=map(int,m.geom_bodyid[[ga,gb]])
   if ba==bid:other=gb
   elif bb==bid:other=ga
   else:continue
   src=records.get((m.geom(other).name or '').removesuffix('__contact'),{}).get('source_mesh_name','')
   if not src.startswith(side)or not any(k in src for k in ['wrist_roll_follower','moving_jaw']):continue
   label='moving'if'moving_jaw'in src else'fixed';rr=d.site_xmat[sid].reshape(3,3);pt=(d.contact.pos[ii]-d.site_xpos[sid])@rr;normal=d.contact.frame[ii,:3]*(1 if other==ga else -1);sign=1 if label=='fixed' else -1
   if not(-.035<pt[0]<.004 and abs(pt[1])<.010 and sign*np.dot(normal,rr[:,2])>.6):continue
   dist=float(d.contact.dist[ii])
   if dist<gp.get(label,1):gp[label]=dist;details[label]={'point':pt.tolist(),'dot':float(np.dot(normal,rr[:,2]))}
  return gp,details
 candidates=[]
 for deep in [0.,.005,.010]:
  for shift in [-.0015,0,.0015,.003,.0045,.006]:
   for delta in [-.05,-.10,-.15,-.20,-.25]:
    place(shift,deep,delta);gp,details=gaps();score=sum(abs(gp.get(k,.05)+.0002)for k in ['fixed','moving']);candidates.append({'deep':deep,'shift':shift,'delta':delta,'gaps':gp,'details':details,'score':score})
 best=min(candidates,key=lambda r:r['score']);deep=best['deep']
 def fitgap(x):place(x[0]*.001,deep,x[1]*.01);gp,_=gaps();return np.array([gp.get(k,.05)+.0002 for k in ['fixed','moving']])*1000
 opt=least_squares(fitgap,[best['shift']/.001,best['delta']/.01],bounds=([-3,-35],[10,5]),max_nfev=50,diff_step=.01,ftol=1e-9,xtol=1e-9,gtol=1e-8);fitgap(opt.x);gp,details=gaps();q=d.qpos.copy();m.geom_margin[:]=margin;d.qacc_warmstart[:]=0
 for ac in range(m.nu):d.ctrl[ac]=d.qpos[m.jnt_qposadr[m.actuator_trnid[ac,0]]]
 mujoco.mj_forward(m,d)
 for ac in range(m.nu):
  j=m.actuator_trnid[ac,0]
  if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
 ac=m.actuator('right_gripper').id;d.ctrl[ac]-=.0005;controls=d.ctrl.copy();print(stage,'INITIAL',gp,details,'actual',v._loaded_pads(d,[(side,part)]),flush=True)
 initialq=q.copy();initialctrl=controls.copy();initialcase=d.xpos[bid].copy();hold=None;minimum={'fixed':1e9,'moving':1e9};both=0;count=0;maxrel=0.;maxrot=0.;trace=[];started=time.monotonic()
 for step in range(round(.3/m.opt.timestep)):
  if d.time<.1:
   loads=v._loaded_pads(d,[(side,part)])[side]
   if min(loads.values())<.1:d.ctrl[ac]=max(initialctrl[ac]-.05,d.ctrl[ac]-.4*m.opt.timestep)
  elif hold is None:
   hold=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,hold,mujoco.mjtState.mjSTATE_INTEGRATION);frozen=d.ctrl.copy();fr=d.site_xmat[sid].reshape(3,3);refp=fr.T@(d.xpos[bid]-d.site_xpos[sid]);refr=fr.T@d.xmat[bid].reshape(3,3)
  physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,[(side,part)])[side]
  if hold is not None:
   count+=1;both+=int(min(loads.values())>=.01)
   for k,val in loads.items():minimum[k]=min(minimum[k],val)
   fr=d.site_xmat[sid].reshape(3,3);rel=fr.T@(d.xpos[bid]-d.site_xpos[sid]);rr=fr.T@d.xmat[bid].reshape(3,3);maxrel=max(maxrel,float(np.linalg.norm(rel-refp)));maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(rr@refr.T)-1)/2,-1,1)))))
  if step%400==0:trace.append({'time':float(d.time),'loads':loads});print(stage,trace[-1],flush=True)
 result={'stage':stage,'side':side,'part':part,'registered_model':True,'qpos':initialq.tolist(),'ctrl':initialctrl.tolist(),'qpos_candidate':initialq.tolist(),'ctrl_candidate':initialctrl.tolist(),'shift_m':float(opt.x[0]*.001),'deep_m':deep,'jaw_delta_rad':float(opt.x[1]*.01),'diagnostic_gaps':gp,'pad_details':details,'held_min_loads_N':minimum,'bilateral_fraction':both/count,'relative_case_motion_m':maxrel,'relative_case_rotation_deg':maxrot,'held_retention_pass':both/count>=.95 and maxrel<.002 and maxrot<3,'controls_unchanged':np.array_equal(frozen,d.ctrl),'source_profile_sha256':p['model_sha256'],'training_eligible':False,'trace':trace,'wall_seconds':time.monotonic()-started}
 np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=initialq,ctrl=initialctrl,hold_state=hold,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(result,indent=2)+'\n');results[stage]=result;print(stage,'FINAL',result['held_retention_pass'],result['bilateral_fraction'],maxrel,maxrot,flush=True)
(OUT/'recipes.json').write_text(json.dumps(results,indent=2)+'\n')
