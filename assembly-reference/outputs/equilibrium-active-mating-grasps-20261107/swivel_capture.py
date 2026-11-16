"""Use the real base swivel to obtain reachable opposed inner-face servo grips."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'swivel-v1';OUT.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--stages',nargs='+',default=['21_upper_far_side','22_upper_half','23_upper_near']);args=a.parse_args()
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=records;source=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text());t=mujoco.MjData(m)
with np.load(ROOT/'outputs/equilibrium-transport-20261107/transfer-v2/20_upper_transport.npz')as z:mujoco.mj_setState(m,t,z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION);t.ctrl[:]=z['ctrl']
mujoco.mj_kinematics(m,t);gid=m.geom('servo_2_case__contact').id;sid=m.site('right_gripperframe').id;casebody=m.body('target_servo_2_pivot').id;gpos=t.geom_xpos[gid]+.0004*t.xaxis[m.joint('target_servo_2_shaft').id];grot=t.geom_xmat[gid].reshape(3,3);relp=grot.T@(t.site_xpos[sid]-gpos);relr=grot.T@t.site_xmat[sid].reshape(3,3);names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint('right_'+n).id for n in names];qids=m.jnt_qposadr[jids];jaw=m.joint('right_gripper').qposadr[0];opening=float(t.qpos[jaw]);jaw_preload=float(t.ctrl[m.actuator('right_gripper').id]-opening);lo=m.jnt_range[jids,0]+1e-6;hi=m.jnt_range[jids,1]-1e-6;results={}
for stage in args.stages:
 d=mujoco.MjData(m);d.qpos[:]=source[stage]['qpos'];d.ctrl[:]=source[stage]['ctrl'];q0=d.qpos.copy();ctrl0=d.ctrl.copy()
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage[:2])>=first
 mujoco.mj_kinematics(m,d);shaft=m.joint('target_servo_1_shaft');anchor=d.xanchor[shaft.id].copy();axis=d.xaxis[shaft.id].copy();oldposes={name:(d.xpos[m.body(name).id].copy(),d.xmat[m.body(name).id].reshape(3,3).copy())for name in ['shoulder_connector','assembly_pin','upper_arm_module']};candidates=[];started=time.monotonic()
 for yaw in [0,5,-5,10,-10,15,-15,20,-20]:
  d.qpos[:]=q0;rot=Rotation.from_rotvec(axis*np.deg2rad(yaw)).as_matrix()
  for piece,(pos,rr) in oldposes.items():
   qa=m.joint(piece+'_free').qposadr[0];d.qpos[qa:qa+3]=anchor+rot@(pos-anchor);mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(rot@rr).ravel())
  d.qpos[shaft.qposadr[0]]=q0[shaft.qposadr[0]]+np.deg2rad(yaw);baseq=d.qpos.copy();mujoco.mj_kinematics(m,d);gp=d.geom_xpos[gid].copy();gr=d.geom_xmat[gid].reshape(3,3).copy();hp=gp+gr@relp;hr=gr@relr
  for depth in [0,.005,-.005,.010]:
   target=hp+hr[:,0]*depth
   def residual(q):
    d.qpos[:]=baseq;d.qpos[qids]=q;d.qpos[jaw]=opening;mujoco.mj_kinematics(m,d);r=d.site_xmat[sid].reshape(3,3);return np.r_[1000*(d.site_xpos[sid]-target),10*(r[:,2]-hr[:,2]),.10*(r[:,0]-hr[:,0])]
   fits=[]
   for start in [q0[qids],t.qpos[qids]]:
    fit=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-8);fits.append(fit)
   fit=min(fits,key=lambda f:np.linalg.norm(f.fun));residual(fit.x);poserr=float(np.linalg.norm(d.site_xpos[sid]-target));spanerr=float(np.linalg.norm(d.site_xmat[sid].reshape(3,3)[:,2]-hr[:,2]));d.ctrl[:]=ctrl0
   for n in names+['gripper']:ac=m.actuator('right_'+n).id;d.ctrl[ac]=d.qpos[m.jnt_qposadr[m.actuator_trnid[ac,0]]]
   mujoco.mj_forward(m,d)
   for n in names:
    ac=m.actuator('right_'+n).id;j=m.actuator_trnid[ac,0];d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
   d.ctrl[m.actuator('right_gripper').id]+=jaw_preload;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);loads=v._loaded_pads(d,[('right','target_servo_2_pivot')])['right'];other=[];intended=[]
   for con in d.contact:
    ga,gb=map(int,con.geom);sa=records.get((m.geom(ga).name or '').removesuffix('__contact'),{}).get('source_mesh_name','');sb=records.get((m.geom(gb).name or '').removesuffix('__contact'),{}).get('source_mesh_name','');ha=sa.startswith('right');hb=sb.startswith('right')
    if ha==hb:continue
    hand=ga if ha else gb;targetg=gb if ha else ga;src=sa if ha else sb
    if m.geom_bodyid[targetg]==casebody and any(k in src for k in ['wrist_roll_follower','moving_jaw']):intended.append(float(con.dist))
    elif con.dist<0:other.append({'pair':[m.geom(ga).name,m.geom(gb).name],'distance_m':float(con.dist)})
   otherpen=max([-x['distance_m']for x in other],default=0);fingerpen=max([-x for x in intended],default=0);score=1e6*otherpen+1e6*max(0,fingerpen-.0004)+1000*poserr+30*spanerr+(20 if min(loads.values())<.01 else 0)+.0001*abs(yaw)
   row={'yaw_degrees':yaw,'deep_m':depth,'initial_loads_N':loads,'other_contacts':other,'max_other_penetration_m':otherpen,'max_intended_penetration_m':fingerpen,'site_error_m':poserr,'span_error':spanerr,'score':score,'qpos':d.qpos.tolist(),'ctrl':d.ctrl.tolist()};candidates.append(row)
 best=min(candidates,key=lambda r:r['score']);print(stage,'BEST',{k:x for k,x in best.items()if k not in ['qpos','ctrl','other_contacts']},flush=True);d.qpos[:]=best['qpos'];d.qvel[:]=0;d.ctrl[:]=best['ctrl'];mujoco.mj_forward(m,d);initialq=d.qpos.copy();initialctrl=d.ctrl.copy();hold=None;minimum={'fixed':1e9,'moving':1e9};both=0;count=0;maxrel=0.;maxrot=0.;trace=[]
 for step in range(round(.3/m.opt.timestep)):
  if d.time>=.1 and hold is None:
   hold=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,hold,mujoco.mjtState.mjSTATE_INTEGRATION);frozen=d.ctrl.copy();fr=d.site_xmat[sid].reshape(3,3);refp=fr.T@(d.xpos[casebody]-d.site_xpos[sid]);refr=fr.T@d.xmat[casebody].reshape(3,3)
  physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,[('right','target_servo_2_pivot')])['right']
  if hold is not None:
   count+=1;both+=int(min(loads.values())>=.01)
   for k,x in loads.items():minimum[k]=min(minimum[k],x)
   fr=d.site_xmat[sid].reshape(3,3);rel=fr.T@(d.xpos[casebody]-d.site_xpos[sid]);rr=fr.T@d.xmat[casebody].reshape(3,3);maxrel=max(maxrel,float(np.linalg.norm(rel-refp)));maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(rr@refr.T)-1)/2,-1,1)))))
  if step%400==0:trace.append({'time':float(d.time),'loads':loads});print(stage,trace[-1],flush=True)
 result={'stage':stage,'registered_model':True,'qpos':initialq.tolist(),'ctrl':initialctrl.tolist(),'best':best,'bilateral_fraction':both/count,'minimum_loads_N':minimum,'relative_case_motion_m':maxrel,'relative_case_rotation_deg':maxrot,'held_retention_pass':both/count>=.95 and maxrel<.002 and maxrot<3,'source_profile_sha256':p['model_sha256'],'training_eligible':False,'trace':trace,'wall_seconds':time.monotonic()-started};np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=initialq,ctrl=initialctrl,hold_state=hold,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(result,indent=2)+'\n');results[stage]=result;print(stage,'FINAL',result['held_retention_pass'],result['bilateral_fraction'],maxrel,maxrot,flush=True)
(OUT/'recipes.json').write_text(json.dumps(results,indent=2)+'\n')
