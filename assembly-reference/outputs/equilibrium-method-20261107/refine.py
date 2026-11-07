"""Bounded static native-force equilibrium candidate; requires independent dynamic hold."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step,WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/18_upper_floor_grasp_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=18>=first
mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();scale=np.ones(m.nv)*.05;bound=np.ones(m.nv)*np.deg2rad(8)
for j in range(m.njnt):
 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE: a=m.jnt_dofadr[j];scale[a:a+3]=1;bound[a:a+3]=.004;bound[a+3:a+6]=np.deg2rad(12)
for i in (2,3,4):bound[m.joint(f'target_servo_{i}_shaft').dofadr[0]]=np.deg2rad(20)
# normalized joint displacements, to avoid translating by radians.
scratch=mujoco.MjData(m);scratch.eq_active[:]=d.eq_active;calls=0;t0=time.monotonic()
def evaluate(x):
 global calls
 calls+=1;scratch.qpos[:]=q0;mujoco.mj_integratePos(m,scratch.qpos,x[:m.nv]*bound,1);scratch.qvel[:]=0;scratch.ctrl[:]=ctrl0+x[m.nv:]*.03
 mujoco.mj_forward(m,scratch)
 # Native complete-contact acceleration, actual position actuator force only.
 return np.r_[scratch.qacc*scale, x[:m.nv]*.001, x[m.nv:]*.001]
x=np.zeros(m.nv+m.nu)
# A small symmetric jaw preload is a real motor command, not external hold.
for side in ('left','right'):
 j=m.joint(side+'_gripper').id;a=int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]);x[m.nv+a]=-.3
print('initial',np.linalg.norm(evaluate(x)[:m.nv]),'max',max(abs(scratch.qacc)),flush=True)
r=least_squares(evaluate,x,bounds=(-np.ones_like(x),np.ones_like(x)),max_nfev=20,diff_step=1e-4,ftol=1e-8,xtol=1e-8,gtol=1e-6,verbose=1)
evaluate(r.x);q=scratch.qpos.copy();ctrl=scratch.ctrl.copy();qacc=scratch.qacc.copy();d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes'];pairs=[('left','base_side_cover'),('right','target_servo_2_pivot')]
records=[]
def rec():
 records.append({'time':float(d.time),'max_abs_qvel':float(max(abs(d.qvel))),'max_abs_qacc':float(max(abs(d.qacc))),'pad_loads_N':v._loaded_pads(d,pairs),'min_contact_m':float(d.contact.dist[:d.ncon].min()),'weighted_position_diff':float(np.linalg.norm((d.qpos-q)[:3]))});print(records[-1],flush=True)
rec()
for i in range(round(.1/m.opt.timestep)):
 physical_split_step(m,d,{'finger_part':1.,'part_part':1.,'part_table':1.})
 if(i+1)%100==0:rec()
result={'elapsed_seconds':time.monotonic()-t0,'calls':calls,'nfev':r.nfev,'cost':r.cost,'message':r.message,'initial_qacc_max':float(max(abs(qacc))),'records':records,'qpos_candidate':q.tolist(),'controls_candidate':ctrl.tolist(),'x':r.x.tolist(),'training_eligible':False}
(OUT/'refine18.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'refine18.npz',qpos=q,ctrl=ctrl,hold_qpos=d.qpos,hold_qvel=d.qvel)
