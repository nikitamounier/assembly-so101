"""Transfer an actual loaded floor grasp into a reachable transport pose."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'transfer-v1';OUT.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--stage',default='20_upper_transport');a.add_argument('--wrist-relax',type=float,default=.06);args=a.parse_args();stage=args.stage
side='right' if stage.startswith('20') else 'left';piece='upper_arm_module' if side=='right' else 'forearm_module';servo=2 if side=='right' else 3;part=f'target_servo_{servo}_pivot';template='18_upper_floor_grasp' if side=='right' else '26_final_floor_grasp'
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m);t=mujoco.MjData(m)
with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage.split('_')[0])>=first
with np.load(ROOT/f'outputs/equilibrium-reset-20261107/native-floor-v5-pilot/{template}.npz')as z:mujoco.mj_setState(m,t,z['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION);t.ctrl[:]=z['controls'][0]
mujoco.mj_forward(m,d);mujoco.mj_forward(m,t);oldq=d.qpos.copy();oldcase=d.xpos[m.body(part).id].copy();oldcaser=d.xmat[m.body(part).id].reshape(3,3).copy();site=m.site(side+'_gripperframe').id;names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];qids=[m.joint(side+'_'+n).qposadr[0]for n in names]
d.qpos[qids[-1]]=t.qpos[qids[-1]];d.qpos[qids[3]]-=args.wrist_relax;mujoco.mj_kinematics(m,d)
change=d.site_xmat[site].reshape(3,3)@t.site_xmat[site].reshape(3,3).T;trans=d.site_xpos[site]-change@t.site_xpos[site]
root=m.body(piece).id;qa=m.joint(piece+'_free').qposadr[0];d.qpos[qa:qa+3]=change@t.xpos[root]+trans;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(change@t.xmat[root].reshape(3,3)).ravel())
for si in ([2]if servo==2 else[3,4]):j=m.joint(f'target_servo_{si}_shaft').qposadr[0];d.qpos[j]=t.qpos[j]
mujoco.mj_forward(m,d)
for ac in range(m.nu):
 j=m.actuator_trnid[ac,0];d.ctrl[ac]=d.qpos[m.jnt_qposadr[j]]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]+=d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0]
 d.ctrl[ac]=np.clip(d.ctrl[ac],*m.actuator_ctrlrange[ac])
aid=m.actuator(side+'_gripper').id;template_delta=t.ctrl[aid]-t.qpos[qids[-1]];d.ctrl[aid]=d.qpos[qids[-1]]+template_delta
q=d.qpos.copy();ctrl=d.ctrl.copy();initial=d.xpos[m.body(part).id].copy();initial_r=d.xmat[m.body(part).id].reshape(3,3).copy();v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[(side,part)];records=[];minimum={'fixed':1e9,'moving':1e9};maxmotion=0.;maxrot=0.;hs=None;holdstate=None;start=time.monotonic()
for step in range(round(1./m.opt.timestep)):
 if d.time>=.8 and hs is None:
  hs=d.xpos[m.body(part).id].copy();hr=d.xmat[m.body(part).id].reshape(3,3).copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,pairs)[side]
 if hs is not None:
  for k,x in loads.items():minimum[k]=min(minimum[k],x)
  maxmotion=max(maxmotion,float(np.linalg.norm(d.xpos[m.body(part).id]-hs)));r=d.xmat[m.body(part).id].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(r@hr.T)-1)/2,-1,1)))))
 if step%400==0:records.append({'time':float(d.time),'loads':loads,'case_drift_m':float(np.linalg.norm(d.xpos[m.body(part).id]-initial))});print(records[-1],flush=True)
result={'stage':stage,'side':side,'part':part,'piece':piece,'qpos_candidate':q.tolist(),'ctrl_candidate':ctrl.tolist(),'original_qpos':oldq.tolist(),'wrist_relax_rad':args.wrist_relax,'case_translation_from_old_m':(initial-oldcase).tolist(),'case_rotation_from_old_degrees':float(np.rad2deg(np.arccos(np.clip((np.trace(initial_r@oldcaser.T)-1)/2,-1,1)))),'jaw_preload_actual_rad':float(template_delta),'minimum_loads_N':minimum,'hold_max_motion_m':maxmotion,'hold_max_rotation_deg':maxrot,'case_settle_translation_m':float(np.linalg.norm(d.xpos[m.body(part).id]-initial)),'controls_unchanged':np.array_equal(ctrl,d.ctrl),'bracket_target':False,'external_forces':False,'local_hold_pass':min(minimum.values())>=.01 and maxmotion<.00025 and maxrot<.25,'training_eligible':False,'wall_seconds':time.monotonic()-start,'trace':records,'model_sha256':p['model_sha256']}
np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=ctrl,hold_state=holdstate,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(result,indent=2)+'\n');print('FINAL',result['local_hold_pass'],minimum,maxmotion,maxrot,flush=True)
