import json,copy
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit,world_mesh
out=Path(__file__).resolve().parent; best=json.loads((out/'best.json').read_text());bank=Path('assets_local/wowrobo/photo-resets-repaired-v1')
m=mujoco.MjModel.from_xml_path(str(bank/'scene.xml'));d=mujoco.MjData(m)
with np.load(bank/'stages/18_upper_floor_grasp_train.npz') as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
ids=[int(m.joint('left_'+n).qposadr[0]) for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']];s=m.site('left_gripperframe').id;A=NativeBoundaryAudit(m)
d.qpos[ids]=best['q_left'];mujoco.mj_forward(m,d);p0=d.site_xpos[s].copy();r0=d.site_xmat[s].reshape(3,3).copy();q0=d.qpos[ids].copy();target=world_mesh(m,d,next(g for g in A.targets if A.info[g]['piece']=='base_side_cover'))
bounds=np.array([m.jnt_range[m.joint('left_'+n).id] for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']])
def pads():
 frame=d.site_xmat[s].reshape(3,3);rows=[]
 for g in [19,20]:
  mesh=world_mesh(m,d,g);pts=np.vstack([mesh.vertices,mesh.triangles_center]);ns=np.vstack([mesh.vertex_normals,mesh.face_normals]);loc=(pts-d.site_xpos[s])@frame;ln=ns@frame;mask=(loc[:,0]>-.025)&(loc[:,0]<.004)&(abs(loc[:,1])<.01)&(ln[:,2]>.5 if g==19 else ln[:,2]<-.5);pts=pts[mask];ns=ns[mask]
  near,dist,faces=trimesh.proximity.closest_point(target,pts);k=np.argmin(dist)
  rows.append(dict(gap_m=float(dist[k]),point_world_m=pts[k].tolist(),target_point_world_m=near[k].tolist(),point_site_m=((pts[k]-d.site_xpos[s])@frame).tolist(),pad_normal_world=ns[k].tolist(),target_normal_world=target.face_normals[faces[k]].tolist(),normal_dot=float(ns[k]@target.face_normals[faces[k]]),geom=g,source_mesh=m.mesh(m.geom_dataid[g]).name))
 return rows
rows=[]
for shift in [.00014,.00015,.00016]:
 goal=p0+r0[:,2]*shift
 def residual(q):
  d.qpos[ids[:5]]=q;mujoco.mj_forward(m,d)
  return np.r_[(d.site_xpos[s]-goal)*1000,Rotation.from_matrix(r0.T@d.site_xmat[s].reshape(3,3)).as_rotvec()*20,(q-q0[:5])*.01]
 sol=least_squares(residual,np.clip(q0[:5],bounds[:,0]+1e-12,bounds[:,1]-1e-12),bounds=(bounds[:,0],bounds[:,1]),xtol=1e-13,ftol=1e-13,gtol=1e-11,max_nfev=150)
 qarm=sol.x.copy()
 for jawdelta in [-.0035,-.004,-.0045]:
  d.qpos[ids[:5]]=qarm;d.qpos[ids[-1]]=q0[-1]+jawdelta;mujoco.mj_forward(m,d);ps=pads();au=A.audit(d,target_target=False);left=[p for p in au['helper_target'] if p['a']['helper_side']=='left'];pen=max(p['penetration_m'] for p in left)
  row=dict(shift_m=shift,jaw_delta_rad=jawdelta,q_left=d.qpos[ids].tolist(),pads=ps,left_penetration_m=pen,site_translation=(d.site_xpos[s]-p0).tolist(),orientation_delta_rad=Rotation.from_matrix(r0.T@d.site_xmat[s].reshape(3,3)).as_rotvec().tolist())
  rows.append(row);(out/'precontact-family.json').write_text(json.dumps(rows,indent=2));print(shift,jawdelta,'gaps',[p['gap_m']*1e6 for p in ps],'pen',pen*1e6,flush=True)
