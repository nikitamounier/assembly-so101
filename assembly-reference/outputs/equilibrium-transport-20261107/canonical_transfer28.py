"""Identical normalized servo-case geometry transfers a real loaded grasp."""
import json,time,hashlib
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,physical_split_step
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'canonical28-v1';OUT.mkdir(exist_ok=True);stage='28_final_transport'
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'floor26-isolated/scene.xml'));d=mujoco.MjData(m);t=mujoco.MjData(m)
with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
d.eq_active[:]=False;original=d.qpos.copy()
with np.load(Path(__file__).parent/'transfer-v2/20_upper_transport.npz')as z:mujoco.mj_setState(m,t,z['hold_state'],mujoco.mjtState.mjSTATE_INTEGRATION);t.ctrl[:]=z['ctrl']
mujoco.mj_forward(m,t);sr=m.site('right_gripperframe').id;sl=m.site('left_gripperframe').id;g2=m.geom('servo_2_case__contact').id;g3=m.geom('servo_3_case__contact').id
names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];leftq=[m.joint('left_'+n).qposadr[0]for n in names];rightq=[m.joint('right_'+n).qposadr[0]for n in names];d.qpos[leftq[-1]]=t.qpos[rightq[-1]];d.qpos[leftq[3]]-=.06;mujoco.mj_forward(m,d)
delta=d.site_xmat[sl].reshape(3,3)@t.site_xmat[sr].reshape(3,3).T;shift=d.site_xpos[sl]-delta@t.site_xpos[sr];wantp=delta@t.geom_xpos[g2]+shift;wantr=delta@t.geom_xmat[g2].reshape(3,3)
rot=wantr@d.geom_xmat[g3].reshape(3,3).T;move=wantp-rot@d.geom_xpos[g3];root=m.body('forearm_module').id;qa=m.joint('forearm_module_free').qposadr[0];rootp=d.xpos[root].copy();rootr=d.xmat[root].reshape(3,3).copy();d.qpos[qa:qa+3]=rot@rootp+move;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(rot@rootr).ravel());mujoco.mj_forward(m,d)
for ac in range(m.nu):
 j=m.actuator_trnid[ac,0];d.ctrl[ac]=d.qpos[m.jnt_qposadr[j]]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]+=d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0]
 d.ctrl[ac]=np.clip(d.ctrl[ac],*m.actuator_ctrlrange[ac])
aid=m.actuator('left_gripper').id;sourceaid=m.actuator('right_gripper').id;d.ctrl[aid]+=t.ctrl[sourceaid]-t.qpos[rightq[-1]]
q=d.qpos.copy();controls=d.ctrl.copy();v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','target_servo_3_pivot')];bid=m.body('target_servo_3_pivot').id;startp=d.xpos[bid].copy();records=[];minimum={'fixed':1e9,'moving':1e9};maxmotion=0.;maxrot=0.;hs=None;holdstate=None;started=time.monotonic()
for step in range(round(.6/m.opt.timestep)):
 if d.time>=.4 and hs is None:
  hs=d.xpos[bid].copy();hr=d.xmat[bid].reshape(3,3).copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.});loads=v._loaded_pads(d,pairs)['left']
 if hs is not None:
  for k,x in loads.items():minimum[k]=min(minimum[k],x)
  maxmotion=max(maxmotion,float(np.linalg.norm(d.xpos[bid]-hs)));r=d.xmat[bid].reshape(3,3);maxrot=max(maxrot,float(np.rad2deg(np.arccos(np.clip((np.trace(r@hr.T)-1)/2,-1,1)))))
 if step%400==0:records.append({'time':float(d.time),'loads':loads,'case_drift_m':float(np.linalg.norm(d.xpos[bid]-startp))});print(records[-1],flush=True)
r={'stage':stage,'side':'left','part':'target_servo_3_pivot','piece':'forearm_module','scope':'LEFT arm + final module + table; unrelated contacts disabled, full-scene pending','source_profile_sha256':p['model_sha256'],'source_case_field_sha256':v.grid_records['servo_2_case']['grid_sha256'],'target_case_field_sha256':v.grid_records['servo_3_case']['grid_sha256'],'qpos_candidate':q.tolist(),'ctrl_candidate':controls.tolist(),'original_qpos':original.tolist(),'minimum_loads_N':minimum,'hold_max_motion_m':maxmotion,'hold_max_rotation_deg':maxrot,'total_case_drift_m':float(np.linalg.norm(d.xpos[bid]-startp)),'local_hold_pass':min(minimum.values())>=.01 and maxmotion<.00025 and maxrot<.25,'held_retention_pass':min(minimum.values())>=.01 and maxmotion<.002 and maxrot<3,'controls_unchanged':np.array_equal(controls,d.ctrl),'prescribed_hinge_angle':False,'external_forces':False,'training_eligible':False,'trace':records,'wall_seconds':time.monotonic()-started}
np.savez_compressed(OUT/(stage+'.npz'),qpos_candidate=q,ctrl=controls,hold_state=holdstate,final_qpos=d.qpos,final_qvel=d.qvel);(OUT/(stage+'.json')).write_text(json.dumps(r,indent=2)+'\n');print('FINAL',r['held_retention_pass'],minimum,maxmotion,maxrot,flush=True)
