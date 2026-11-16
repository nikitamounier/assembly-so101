import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import ROOT,WowRoboPhysicalValidator,physical_split_step
out=ROOT/'outputs/equilibrium-model-20261107'
p=json.loads((ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);z=np.load(ROOT/'outputs/cover-support-fix-20261107/physical-precontact/18_upper_floor_grasp.npz')
v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes']
js=[m.joint('left_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];qa=m.jnt_qposadr[js];dofs=m.jnt_dofadr[js];acts=[int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0])for j in js]
site=m.site('left_gripperframe').id;ja=int(np.flatnonzero(m.actuator_trnid[:,0]==m.joint('left_gripper').id)[0])
results=[]
for shift_mm,jaw_delta,adaptive in [(0,0,False),(.1,0,False),(.2,0,False),(.2,0,True)]:
 d=mujoco.MjData(m);d.qpos[:]=z['input_qpos'];d.ctrl[:]=z['input_controls'];d.eq_active[:2]=True;mujoco.mj_forward(m,d)
 target=d.site_xpos[site]+d.site_xmat[site].reshape(3,3)[:,2]*shift_mm*.001
 for it in range(12):
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site)
  e=target-d.site_xpos[site];J=np.vstack([jp[:,dofs],.02*jr[:,dofs]]);rhs=np.r_[e,np.zeros(3)]
  dq=np.linalg.lstsq(J,rhs,rcond=1e-8)[0];d.qpos[qa]=np.clip(d.qpos[qa]+dq,m.jnt_range[js,0]+1e-7,m.jnt_range[js,1]-1e-7);mujoco.mj_forward(m,d)
 initial_q=d.qpos[qa].copy();d.ctrl[acts]=initial_q;d.ctrl[ja]+=jaw_delta;initial_ctrl=float(d.ctrl[ja]);minimum={'fixed':1e9,'moving':1e9};maximumdrift=0;startsite=d.site_xpos[site].copy();trace=[]
 for step in range(800):
  loads=v._loaded_pads(d,[('left','base_side_cover')])['left']
  if adaptive and step<600 and min(loads.values())<.05:d.ctrl[ja]=max(initial_ctrl-.12,d.ctrl[ja]-.4*m.opt.timestep)
  physical_split_step(m,d,dict(finger_part=1.,part_table=1.,part_part=1.))
  if step>=600:
   loads=v._loaded_pads(d,[('left','base_side_cover')])['left'];minimum={k:min(minimum[k],loads[k])for k in minimum}
  if step%40==0:trace.append(dict(time=float(d.time),loads=loads,site=d.site_xpos[site].tolist()))
 result=dict(shift_mm=shift_mm,jaw_delta=jaw_delta,adaptive=adaptive,q_left_initial=initial_q.tolist(),minloads=minimum,final_loads=v._loaded_pads(d,[('left','base_side_cover')]),site_drift_m=(d.site_xpos[site]-startsite).tolist(),left_pan_error=float(d.qpos[qa[0]]-d.ctrl[acts[0]]),trace=trace)
 results.append(result);(out/'left-preload.json').write_text(json.dumps(results,indent=2)+'\n');print({k:val for k,val in result.items()if k!='trace'},flush=True)
