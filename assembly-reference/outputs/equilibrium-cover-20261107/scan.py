import json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);vis=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'))
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes']
qs=[m.joint('left_'+n).qposadr[0]for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jids=[m.joint('left_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];jaw=m.joint('left_gripper').qposadr[0];sid=m.site('left_gripperframe').id
results=[]
for stage in ['01_cover_pickup','03_cover_aligned','04_cover_entry']:
 d=mujoco.MjData(m)
 with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 mujoco.mj_forward(m,d);q0=d.qpos.copy();p0=d.site_xpos[sid].copy();r0=d.site_xmat[sid].reshape(3,3).copy();scratch=mujoco.MjData(vis);scratch.qpos[:]=q0
 for shift in [0.,.0002,.0004,.0006]:
  target=p0+r0[:,2]*shift
  def fun(x):
   scratch.qpos[qs]=x;mujoco.mj_forward(vis,scratch);r=scratch.site_xmat[sid].reshape(3,3)
   return np.r_[1000*(scratch.site_xpos[sid]-target),2*(r[:,2]-r0[:,2]),r[:,0]-r0[:,0]]
  fit=least_squares(fun,q0[qs],bounds=(np.maximum(m.jnt_range[jids,0]+1e-7,q0[qs]-.05),np.minimum(m.jnt_range[jids,1]-1e-7,q0[qs]+.05)),max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  for delta in [0.,-.01,-.02,-.03,-.04]:
   d.qpos[:]=q0;d.qpos[qs]=fit.x;d.qpos[jaw]=q0[jaw]+delta;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);ld=v._loaded_pads(d,[('left','base_side_cover')])['left']
   r={'stage':stage,'shift_m':shift,'jaw_delta':delta,'loads':ld,'fit':float(np.linalg.norm(fun(fit.x))),'left_q':np.r_[fit.x,d.qpos[jaw]].tolist(),'min_contact':float(min(d.contact.dist[:d.ncon]))}
   print(r,flush=True);results.append(r)
(OUT/'scan.json').write_text(json.dumps(results,indent=2)+'\n')
