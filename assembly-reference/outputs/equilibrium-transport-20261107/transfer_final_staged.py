"""Transfer an actual loaded floor grasp into a reachable transport pose."""
import argparse,json,time
from pathlib import Path
from scipy.optimize import least_squares
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'transfer-final-v2';OUT.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--stage',default='20_upper_transport');a.add_argument('--wrist-relax',type=float,default=.06);a.add_argument('--deep',type=float,default=.008);a.add_argument('--initial-hinge',type=float,default=10.);a.add_argument('--preload',type=float,default=.003);args=a.parse_args();stage=args.stage
side='right' if stage.startswith('20') else 'left';piece='upper_arm_module' if side=='right' else 'forearm_module';servo=2 if side=='right' else 3;part=f'target_servo_{servo}_pivot';template='18_upper_floor_grasp' if side=='right' else '26_final_floor_grasp'
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path'] if servo==2 else str(Path(__file__).parent/'floor26-isolated/scene.xml'));d=mujoco.MjData(m);t=mujoco.MjData(m)
with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage.split('_')[0])>=first
with np.load(ROOT/f'outputs/equilibrium-reset-20261107/native-floor-v5-pilot/{template}.npz' if servo==2 else Path(__file__).parent/'floor26-isolated/template.npz')as z:mujoco.mj_setState(m,t,z['states'][0] if servo==2 else z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION);t.ctrl[:]=z['controls'][0] if servo==2 else z['ctrl']
if servo==3:d.eq_active[:]=False
mujoco.mj_forward(m,d);mujoco.mj_forward(m,t);oldq=d.qpos.copy();oldcase=d.xpos[m.body(part).id].copy();oldcaser=d.xmat[m.body(part).id].reshape(3,3).copy();site=m.site(side+'_gripperframe').id;names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];qids=[m.joint(side+'_'+n).qposadr[0]for n in names]
d.qpos[qids[-1]]=t.qpos[qids[-1]];d.qpos[qids[3]]-=args.wrist_relax;mujoco.mj_kinematics(m,d)
change=d.site_xmat[site].reshape(3,3)@t.site_xmat[site].reshape(3,3).T;trans=d.site_xpos[site]-change@t.site_xpos[site]
root=m.body(piece).id;qa=m.joint(piece+'_free').qposadr[0];d.qpos[qa:qa+3]=change@t.xpos[root]+trans;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(change@t.xmat[root].reshape(3,3)).ravel())
for si in ([2]if servo==2 else[3,4]):j=m.joint(f'target_servo_{si}_shaft').qposadr[0];d.qpos[j]=t.qpos[j]
mujoco.mj_forward(m,d)
handr=d.site_xmat[site].reshape(3,3).copy();d.qpos[qa:qa+3]-=handr[:,0]*args.deep
baseq=d.qpos.copy();origmargin=m.geom_margin.copy()
for gg in range(m.ngeom):
 name=m.geom(gg).name or '';src=json.loads(Path(p['grid_manifest']).read_text())['meshes'].get(name.removesuffix('__contact'),{}).get('source_mesh_name','')
 if (m.geom_bodyid[gg]==m.body(part).id and name.endswith('__contact'))or(src.startswith(side)and any(key in src for key in ['wrist_roll_follower','moving_jaw'])):m.geom_margin[gg]=.005
def gapfit(x):
 d.qpos[:]=baseq;d.qpos[qa:qa+3]+=handr[:,2]*x[0]*.001;d.qpos[qids[-1]]+=x[1]*.01;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);gp={}
 for ii in range(d.ncon):
  aa,bb=map(int,d.contact.geom[ii]);ba,bc=map(int,m.geom_bodyid[[aa,bb]])
  if ba==m.body(part).id:other=bb
  elif bc==m.body(part).id:other=aa
  else:continue
  src=json.loads(Path(p['grid_manifest']).read_text())['meshes'].get((m.geom(other).name or '').removesuffix('__contact'),{}).get('source_mesh_name','')
  if not src.startswith(side)or not any(key in src for key in ['wrist_roll_follower','moving_jaw']):continue
  pad='moving'if'moving_jaw'in src else'fixed';rr=d.site_xmat[site].reshape(3,3);pp=(d.contact.pos[ii]-d.site_xpos[site])@rr;nn=d.contact.frame[ii,:3]*(1 if other==aa else-1);sign=1 if pad=='fixed'else-1
  if -.035<pp[0]<.004 and abs(pp[1])<.01 and sign*np.dot(nn,rr[:,2])>.25:gp[pad]=min(gp.get(pad,1),float(d.contact.dist[ii]))
 return np.array([gp.get(key,.01)+.00005 for key in ['fixed','moving']])*1000
fit=least_squares(gapfit,np.zeros(2),bounds=([-3,-8],[3,4]),diff_step=.01,max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-9);gapfit(fit.x);print('fit',fit.x,fit.fun,flush=True)
m.geom_margin[:]=origmargin;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
# Candidate-only passive shape; the live hinge remains unpowered throughout.
wantp=d.xpos[m.body(part).id].copy();wantr=d.xmat[m.body(part).id].reshape(3,3).copy();rootp=d.xpos[root].copy();rootr=d.xmat[root].reshape(3,3).copy();d.qpos[m.joint(f'target_servo_{servo}_shaft').qposadr[0]]=np.deg2rad(args.initial_hinge);mujoco.mj_kinematics(m,d);dr=wantr@d.xmat[m.body(part).id].reshape(3,3).T;shift=wantp-dr@d.xpos[m.body(part).id];d.qpos[qa:qa+3]=dr@rootp+shift;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(dr@rootr).ravel());mujoco.mj_forward(m,d)
for ac in range(m.nu):
 j=m.actuator_trnid[ac,0];d.ctrl[ac]=d.qpos[m.jnt_qposadr[j]]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]+=d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0]
 d.ctrl[ac]=np.clip(d.ctrl[ac],*m.actuator_ctrlrange[ac])
aid=m.actuator(side+'_gripper').id;template_delta=-args.preload;d.ctrl[aid]=d.qpos[qids[-1]]+template_delta
q=d.qpos.copy();ctrl=d.ctrl.copy();initial=d.xpos[m.body(part).id].copy();initial_r=d.xmat[m.body(part).id].reshape(3,3).copy();v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[(side,part)];records=[];minimum={'fixed':1e9,'moving':1e9};maxmotion=0.;maxrot=0.;hs=None;holdstate=None;start=time.monotonic()
for step in range(round(1./m.opt.timestep)):
 if d.time>=.8 and hs is None:
  hs=d.xpos[m.body(part).id].copy();hr=d.xmat[m.body(part).id].reshape(3,3).copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,pairs)[side]
 if hs is not None:
  for k,x in loads.items():minimum[k]=min(minimum[k],x)
  maxmotion=max(maxmotion,float(np.linalg.norm(d.xpos[m.body(part).id]-hs)));r=d.xmat[m.body(part).id].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(r@hr.T)-1)/2,-1,1)))))
 if step%400==0:records.append({'time':float(d.time),'loads':loads,'case_drift_m':float(np.linalg.norm(d.xpos[m.body(part).id]-initial))});print(records[-1],flush=True)
result={'scope':'Full native scene' if servo==2 else 'LEFT arm + final module + table; unrelated contacts disabled; full-scene pending','stage':stage,'side':side,'part':part,'piece':piece,'qpos_candidate':q.tolist(),'ctrl_candidate':ctrl.tolist(),'original_qpos':oldq.tolist(),'initial_hinge_degrees':args.initial_hinge,'finger_deepening_m':args.deep,'wrist_relax_rad':args.wrist_relax,'case_translation_from_old_m':(initial-oldcase).tolist(),'case_rotation_from_old_degrees':float(np.rad2deg(np.arccos(np.clip((np.trace(initial_r@oldcaser.T)-1)/2,-1,1)))),'jaw_preload_actual_rad':float(template_delta),'minimum_loads_N':minimum,'hold_max_motion_m':maxmotion,'hold_max_rotation_deg':maxrot,'case_settle_translation_m':float(np.linalg.norm(d.xpos[m.body(part).id]-initial)),'controls_unchanged':np.array_equal(ctrl,d.ctrl),'bracket_target':False,'external_forces':False,'local_hold_pass':min(minimum.values())>=.01 and maxmotion<.00025 and maxrot<.25,'training_eligible':False,'wall_seconds':time.monotonic()-start,'trace':records,'model_sha256':p['model_sha256']}
np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=ctrl,hold_state=holdstate,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(result,indent=2)+'\n');print('FINAL',result['local_hold_pass'],minimum,maxmotion,maxrot,flush=True)
