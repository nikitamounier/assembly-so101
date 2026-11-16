"""Unchanged native cover hold after removing modeled bearing collisions."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step,WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--profile',default='outputs/equilibrium-model-20261107/profile-bearing-v4');ap.add_argument('--gravity',action='store_true');ap.add_argument('--name',default='baseline-v4');ap.add_argument('--seconds',type=float,default=.8);a=ap.parse_args()
p=json.loads((ROOT/a.profile/'profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/18_upper_floor_grasp_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=18>=first
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();si=m.site('left_gripperframe').id;site0=d.site_xpos[si].copy()
if a.gravity:
 for ac in range(m.nu):
  j=m.actuator_trnid[ac,0]
  if not m.joint(j).name.endswith('_gripper'):d.ctrl[ac]=np.clip(d.ctrl[ac]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0],*m.actuator_ctrlrange[ac])
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','base_side_cover'),('right','target_servo_2_pivot')]
records=[];start=time.monotonic();frozen=None;holdstate=None;minloads={s:{'fixed':1e9,'moving':1e9}for s,_ in pairs};maxsite=0.;maxvel=0.;hs=None
for step in range(round(a.seconds/m.opt.timestep)):
 if d.time<a.seconds-.2:
  loads=v._loaded_pads(d,pairs)
  ac=m.actuator('right_gripper').id
  if min(loads['right'].values())<.05:d.ctrl[ac]=max(ctrl0[ac]-.12,d.ctrl[ac]-.4*m.opt.timestep)
 else:
  if frozen is None:
   frozen=d.ctrl.copy();hs=d.site_xpos[si].copy();holdstate=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,holdstate,mujoco.mjtState.mjSTATE_INTEGRATION)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
 if frozen is not None:
  for s,ld in v._loaded_pads(d,pairs).items():
   for pad,val in ld.items():minloads[s][pad]=min(minloads[s][pad],val)
  maxsite=max(maxsite,float(np.linalg.norm(d.site_xpos[si]-hs)));maxvel=max(maxvel,float(max(abs(d.qvel))))
 if step%400==0 or step==round(a.seconds/m.opt.timestep)-1:
  r={'time':float(d.time),'loads':v._loaded_pads(d,pairs),'leftsite_delta':(d.site_xpos[si]-site0).tolist(),'maxqvel':float(max(abs(d.qvel))),'left_pan_q':float(d.joint('left_shoulder_pan').qpos[0]),'left_pan_ctrl':float(d.ctrl[m.actuator('left_shoulder_pan').id]),'mincontact':float(min(d.contact.dist[:d.ncon]))};records.append(r);print(r,flush=True)
r={'profile':p,'gravity_compensation':a.gravity,'records':records,'hold_min_loads_N':minloads,'hold_controls_unchanged':np.array_equal(frozen,d.ctrl),'hold_max_leftsite_drift_m':maxsite,'hold_max_velocity':maxvel,'wall_seconds':time.monotonic()-start,'training_eligible':False}
(OUT/(a.name+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(OUT/(a.name+'.npz'),qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,hold_state=holdstate,initial_qpos=q0,initial_ctrl=ctrl0)
