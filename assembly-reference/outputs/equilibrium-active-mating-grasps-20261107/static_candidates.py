"""Actual inner-case grasp transfer, case/pocket pose fixed, active helper only."""
import json,time,argparse
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near']);args=ap.parse_args()
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=records
source=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text());results={}
for stage in args.stages:
 side='right'if stage.startswith(('21','22','23'))else'left';servo=2 if side=='right' else 3;case=f'target_servo_{servo}_pivot';casebody=m.body(case).id;sid=m.site(side+'_gripperframe').id;gid=m.geom(f'servo_{servo}_case__contact').id
 templatepath=ROOT/('outputs/equilibrium-transport-20261107/transfer-v2/20_upper_transport.npz'if servo==2 else'outputs/equilibrium-transport-20261107/canonical28-v1/28_final_transport.npz')
 t=mujoco.MjData(m)
 with np.load(templatepath)as z:mujoco.mj_setState(m,t,z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION);t.ctrl[:]=z['ctrl']
 mujoco.mj_kinematics(m,t);temp_gid=m.geom(f'servo_{servo}_case__contact').id;temp_sid=m.site(side+'_gripperframe').id;axis=t.xaxis[m.joint(f'target_servo_{servo}_shaft').id];gpos=t.geom_xpos[temp_gid]+.0004*axis;grot=t.geom_xmat[temp_gid].reshape(3,3);hpos=t.site_xpos[temp_sid].copy();hrot=t.site_xmat[temp_sid].reshape(3,3).copy();relative_p=grot.T@(hpos-gpos);relative_r=grot.T@hrot
 q0=np.array(source[stage]['qpos']);ctrl0=np.array(source[stage]['ctrl']);d=mujoco.MjData(m);d.qpos[:]=q0;d.ctrl[:]=ctrl0
 for n,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(n).id]=int(stage.split('_')[0])>=first
 mujoco.mj_kinematics(m,d);wantedcase=d.xpos[casebody].copy();wantedcaser=d.xmat[casebody].copy();goalp=d.geom_xpos[gid].copy();goalr=d.geom_xmat[gid].reshape(3,3).copy();hp=goalp+goalr@relative_p;hr=goalr@relative_r
 names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint(side+'_'+n).id for n in names];qids=m.jnt_qposadr[jids];jaw=m.joint(side+'_gripper').qposadr[0];opening=float(t.joint(side+'_gripper').qpos[0]);offset=np.array([-.014,0,.0125]);pivot=hp+hr@offset
 lo=m.jnt_range[jids,0]+1e-6;hi=m.jnt_range[jids,1]-1e-6;prev=q0[qids].copy();candidates=[];started=time.monotonic()
 for height in [-.010,-.005,0,.005,.010]:
  for angle in [-30,-15,0,15,30]:
   desired_r=Rotation.from_rotvec(hr[:,2]*np.deg2rad(angle)).as_matrix()@hr;desired_p=pivot-desired_r@offset+goalr[:,1]*height
   def residual(q):
    d.qpos[:]=q0;d.qpos[qids]=q;d.qpos[jaw]=opening;mujoco.mj_kinematics(m,d);r=d.site_xmat[sid].reshape(3,3)
    return np.r_[1000*(d.site_xpos[sid]-desired_p),3*(r[:,2]-desired_r[:,2]),.05*(r[:,0]-desired_r[:,0]),.0002*(q-q0[qids])]
   fit=least_squares(residual,np.clip(prev,lo,hi),bounds=(lo,hi),max_nfev=65,ftol=1e-9,xtol=1e-9,gtol=1e-8);residual(fit.x);prev=fit.x.copy();d.ctrl[:]=ctrl0
   for name in names+['gripper']:
    ac=m.actuator(side+'_'+name).id;j=m.actuator_trnid[ac,0];d.ctrl[ac]=d.qpos[m.jnt_qposadr[j]]
   mujoco.mj_forward(m,d)
   for name in names:
    ac=m.actuator(side+'_'+name).id;j=m.actuator_trnid[ac,0];d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
   d.ctrl[m.actuator(side+'_gripper').id]-=.0015;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
   loads=v._loaded_pads(d,[(side,case)])[side];other=[];intended=[]
   for con in d.contact:
    ga,gb=map(int,con.geom);sa=records.get((m.geom(ga).name or '').removesuffix('__contact'),{}).get('source_mesh_name','');sb=records.get((m.geom(gb).name or '').removesuffix('__contact'),{}).get('source_mesh_name','');ha=sa.startswith(side);hb=sb.startswith(side)
    if ha==hb:continue
    hand=ga if ha else gb;target=gb if ha else ga;src=sa if ha else sb
    if m.geom_bodyid[target]==casebody and any(key in src for key in ['wrist_roll_follower','moving_jaw']):intended.append(float(con.dist))
    elif con.dist<0:other.append({'pair':[m.geom(ga).name,m.geom(gb).name],'distance_m':float(con.dist)})
   maxother=max([-r['distance_m']for r in other],default=0);poserr=float(np.linalg.norm(d.site_xpos[sid]-desired_p));spanerr=float(np.linalg.norm(d.site_xmat[sid].reshape(3,3)[:,2]-desired_r[:,2]));score=1e6*maxother+1000*poserr+5*spanerr+(10 if min(loads.values())<.01 else 0)+.001*abs(height/.01)
   row={'height_slide_m':height,'in_plane_grasp_rotation_deg':angle,'loads_initial_N':loads,'other_contacts':other,'max_other_penetration_m':maxother,'intended_distances_m':intended,'site_position_error_m':poserr,'span_error':spanerr,'score':score,'qpos':d.qpos.tolist(),'ctrl':d.ctrl.tolist(),'held_case_position_error_m':float(np.linalg.norm(d.xpos[casebody]-wantedcase)),'held_case_rotation_error':float(np.max(np.abs(d.xmat[casebody]-wantedcaser)))};candidates.append(row)
 best=min(candidates,key=lambda r:r['score']);result={'stage':stage,'side':side,'servo':servo,'source_profile_sha256':p['model_sha256'],'registered_model':True,'source_template':str(templatepath),'best':best,'candidates':candidates,'wall_seconds':time.monotonic()-started,'training_eligible':False};results[stage]=result;(OUT/(stage+'-static.json')).write_text(json.dumps(result,indent=2)+'\n');print(stage,{k:v for k,v in best.items()if k not in ['qpos','ctrl','other_contacts']},flush=True)
(OUT/'static-candidates.json').write_text(json.dumps(results,indent=2)+'\n')
