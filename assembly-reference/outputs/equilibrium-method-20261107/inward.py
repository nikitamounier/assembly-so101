"""Real servo-command gravity compensation, no external forces/welds/velocity reset."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step,WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,_ancestor
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/18_upper_floor_grasp_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=18>=first
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();motorcomp={}
for a in range(m.nu):
 j=m.actuator_trnid[a,0];n=m.joint(j).name
 if n.endswith('_gripper'):continue
 assert m.actuator_gear[a,0]==1
 delta=d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[a,0]
 d.ctrl[a]=np.clip(d.ctrl[a]+delta,*m.actuator_ctrlrange[a]);motorcomp[n]=float(d.ctrl[a]-ctrl0[a])
mujoco.mj_forward(m,d)
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','base_side_cover'),('right','target_servo_2_pivot')]
# Command offset only: the authored qpos stays identical before integration.
from scipy.optimize import lsq_linear
jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);sid=m.site('left_gripperframe').id
mujoco.mj_jacSite(m,d,jp,jr,sid)
act=[a for a in range(m.nu)if m.actuator(a).name.startswith('left_') and not m.actuator(a).name.endswith('_gripper')]
dofs=[m.jnt_dofadr[m.actuator_trnid[a,0]] for a in act]
mat=np.vstack([jp[:,dofs],.05*jr[:,dofs]])
target=np.r_[d.site_xmat[sid].reshape(3,3)[:,2]*.001,np.zeros(3)]
lo=np.maximum(m.actuator_ctrlrange[act,0]-d.ctrl[act],-.05)
hi=np.minimum(m.actuator_ctrlrange[act,1]-d.ctrl[act],.05)
solution=lsq_linear(mat,target,bounds=(lo,hi),tol=1e-12)
d.ctrl[act]+=solution.x
ctrlgravity=d.ctrl.copy();records=[];t0=time.monotonic();initialpos=d.xpos.copy()
offset_receipt={'requested_site_displacement_m':target[:3].tolist(),'linear_achieved_displacement_m':(jp[:,dofs]@solution.x).tolist(),'linear_achieved_rotation_rad':(jr[:,dofs]@solution.x).tolist(),'command_offsets_rad':{m.actuator(a).name:float(v) for a,v in zip(act,solution.x)},'qpos_bitwise_unchanged':np.array_equal(q0,d.qpos)}
print('offset',offset_receipt,flush=True)
def rawcontacts():
 fixed=m.body('left_gripper').id;target=m.body('base_side_cover').id;site=m.site('left_gripperframe').id;frame=d.site_xmat[site].reshape(3,3);rows=[]
 for i in range(d.ncon):
  a,b=map(int,d.contact.geom[i]);ba,bb=map(int,m.geom_bodyid[[a,b]])
  if not ((ba==fixed and _ancestor(m,bb,target))or(bb==fixed and _ancestor(m,ba,target))):continue
  hg=a if ba==fixed else b;normal=d.contact.frame[i,:3]*(1 if hg==a else -1);point=(d.contact.pos[i]-d.site_xpos[site])@frame;f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  rows.append({'geom_names':[m.geom(a).name,m.geom(b).name],'dist_m':float(d.contact.dist[i]),'point_site':point.tolist(),'normal_dot_siteZ':float(normal@frame[:,2]),'force_N':f.tolist()})
 return rows

def rec():
 r={'time':float(d.time),'max_abs_qvel':float(max(abs(d.qvel))),'pad_loads_N':v._loaded_pads(d,pairs),'fixed_cover_raw':rawcontacts(),'left_gripper_displacement_m':(d.xpos[m.body('left_gripper').id]-initialpos[m.body('left_gripper').id]).tolist(),'min_contact_m':float(d.contact.dist[:d.ncon].min())};records.append(r);print(json.dumps(r),flush=True)
rec()
hold_start=None;minimum={s:{"fixed":float("inf"),"moving":float("inf")}for s,_ in pairs}
for i in range(2400):
 loads=v._loaded_pads(d,pairs)
 for side,_ in pairs:
  if i<1600 and min(loads[side].values())<.05 and loads[side]['moving']<5.:
   a=m.actuator(side+'_gripper').id;d.ctrl[a]=max(ctrl0[a]-.12,d.ctrl[a]-.4*m.opt.timestep)
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
 if i==1599:
  hold_start=d.qpos.copy();hold_ctrl=d.ctrl.copy()
 if i>=1600:
  for side,pads in v._loaded_pads(d,pairs).items():
   for pad,load in pads.items():minimum[side][pad]=min(minimum[side][pad],load)
 if(i+1)%400==0:rec()
result={'wall_seconds':time.monotonic()-t0,'profile':p,'motor_compensation_radians':motorcomp,'jaw_command_delta':{s:float(d.ctrl[m.actuator(s+'_gripper').id]-ctrl0[m.actuator(s+'_gripper').id])for s in('left','right')},'offset':offset_receipt,'hold_minimum_pad_loads_N':minimum,'hold_ctrl_bitwise_unchanged':np.array_equal(hold_ctrl,d.ctrl),'hold_max_qpos_difference':float(np.max(np.abs(d.qpos-hold_start))),'records':records,'qvel_reset_during_integration':False,'external_forces':False,'training_eligible':False}
(OUT/'inward18-1mm.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'inward18-1mm.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,initial_qpos=q0,initial_ctrl=ctrl0,gravity_ctrl=ctrlgravity)
