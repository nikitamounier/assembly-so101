"""Actual low-force jaw capture, real arms and unchanged-control hold diagnostic."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import lsq_linear
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step,WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--profile',default='outputs/equilibrium-model-20261107/profile-bearing-v2');p.add_argument('--shift-mm',type=float,default=.5);p.add_argument('--torque',type=float,default=.05);p.add_argument('--name',default='trial');p.add_argument('--seconds',type=float,default=1.0);a=p.parse_args()
profile=json.loads((ROOT/a.profile/'profile.json').read_text());load_backend(profile['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(profile['model_path']);d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/18_upper_floor_grasp_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=18>=first
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();site=m.site('left_gripperframe').id;site0=d.site_xpos[site].copy();span=d.site_xmat[site].reshape(3,3)[:,2].copy()
for act in range(m.nu):
 j=m.actuator_trnid[act,0]
 if not m.joint(j).name.endswith('_gripper'):d.ctrl[act]=np.clip(d.ctrl[act]+d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[act,0],*m.actuator_ctrlrange[act])
armacts=[i for i in range(m.nu)if m.actuator(i).name.startswith('left_')and not m.actuator(i).name.endswith('_gripper')];dofs=[m.jnt_dofadr[m.actuator_trnid[i,0]]for i in armacts]
jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site)
mat=np.vstack([jp[:,dofs],.05*jr[:,dofs]]);target=np.r_[span*a.shift_mm*.001,np.zeros(3)]
sol=lsq_linear(mat,target,bounds=(np.maximum(m.actuator_ctrlrange[armacts,0]-d.ctrl[armacts],-.05),np.minimum(m.actuator_ctrlrange[armacts,1]-d.ctrl[armacts],.05)),tol=1e-12)
d.ctrl[armacts]+=sol.x
ja=m.actuator('left_gripper').id;j=m.joint('left_gripper');qa=int(j.qposadr[0]);dv=int(j.dofadr[0]);kp=m.actuator_gainprm[ja,0];kv=-m.actuator_biasprm[ja,2]
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(profile['grid_manifest']).read_text())['meshes'];pairs=[('left','base_side_cover'),('right','target_servo_2_pivot')]
records=[];start=time.monotonic();frozen=None;minload={s:{'fixed':1e9,'moving':1e9}for s,_ in pairs};holdq=None;holdsite=None
for step in range(round(a.seconds/m.opt.timestep)):
 t=d.time
 if t<.15:d.ctrl[ja]=ctrl0[ja]+.045
 elif t<a.seconds-.2:
  # Constant small closing torque through the real position-servo command.
  d.ctrl[ja]=d.qpos[qa]+(kv*d.qvel[dv]-a.torque)/kp
 else:
  if frozen is None:frozen=d.ctrl.copy();holdq=d.qpos.copy();holdsite=d.site_xpos[site].copy()
 if t<a.seconds-.2:
  loads=v._loaded_pads(d,pairs)
  r=m.actuator('right_gripper').id
  if min(loads['right'].values())<.05:d.ctrl[r]=max(ctrl0[r]-.12,d.ctrl[r]-.4*m.opt.timestep)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
 if frozen is not None:
  for side,ld in v._loaded_pads(d,pairs).items():
   for pad,f in ld.items():minload[side][pad]=min(minload[side][pad],f)
 if step%400==0 or step==round(a.seconds/m.opt.timestep)-1:
  row={'time':float(d.time),'padloads':v._loaded_pads(d,pairs),'left_jaw_q':float(d.qpos[qa]),'left_jaw_ctrl':float(d.ctrl[ja]),'site_displacement':(d.site_xpos[site]-site0).tolist(),'max_qvel':float(max(abs(d.qvel))),'min_contact':float(min(d.contact.dist[:d.ncon]))};records.append(row);print(row,flush=True)
r={'profile':profile,'shift_mm':a.shift_mm,'torque_Nm':a.torque,'arm_ctrl_delta':sol.x.tolist(),'records':records,'hold_min_load_N':minload,'hold_ctrl_unchanged':np.array_equal(frozen,d.ctrl),'hold_site_drift_m':float(np.linalg.norm(d.site_xpos[site]-holdsite)),'wall_seconds':time.monotonic()-start,'training_eligible':False}
(OUT/(a.name+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(OUT/(a.name+'.npz'),qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,initial_qpos=q0,initial_ctrl=ctrl0,hold_qpos=holdq)
