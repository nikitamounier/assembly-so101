import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit,world_mesh
out=Path(__file__).resolve().parent;row=next(r for r in json.loads((out/'lower-candidates.json').read_text()) if r['i']==2)
m=mujoco.MjModel.from_xml_path('assets_local/wowrobo/photo-resets-repaired-v1/scene.xml');d=mujoco.MjData(m)
with np.load('assets_local/wowrobo/photo-resets-repaired-v1/stages/25_right_support_train.npz') as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
base=d.qpos.copy();ids=[int(m.joint('left_'+n).qposadr[0]) for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']];site=m.site('left_gripperframe').id
A=NativeBoundaryAudit(m);results=[]
for delta in [-.0115]:
 d.qpos[:]=base;d.qpos[ids]=row['q_left'];d.qpos[ids[-1]]+=delta;mujoco.mj_forward(m,d)
 audit=A.audit(d,target_target=False)
 left=[p for p in audit['helper_target'] if p['a']['helper_side']=='left'];worst=max(left,key=lambda p:p['penetration_m'])
 target=world_mesh(m,d,next(g for g in A.targets if A.info[g]['piece']=='base_side_cover'));frame=d.site_xmat[site].reshape(3,3);pads=[]
 for hg in [19,20]:
  hand=world_mesh(m,d,hg);points=np.vstack([hand.vertices,hand.triangles_center]);normals=np.vstack([hand.vertex_normals,hand.face_normals]);local=(points-d.site_xpos[site])@frame;ln=normals@frame
  mask=(local[:,0]>-.025)&(local[:,0]<.004)&(abs(local[:,1])<.010)&(ln[:,2]>.5 if hg==19 else ln[:,2]<-.5)
  pts=points[mask];ns=normals[mask];near,dist,faces=trimesh.proximity.closest_point(target,pts);k=int(np.argmin(dist))
  pads.append({'gap_m':float(dist[k]),'point_world_m':pts[k].tolist(),'target_point_world_m':near[k].tolist(),'pad_normal_world':ns[k].tolist(),'target_normal_world':target.face_normals[faces[k]].tolist(),'normal_dot':float(ns[k]@target.face_normals[faces[k]]),'point_site_m':((pts[k]-d.site_xpos[site])@frame).tolist(),'geom':hg,'source_mesh':m.mesh(m.geom_dataid[hg]).name})
 r=dict(row,q_left=d.qpos[ids].tolist(),closing_delta_rad=float(delta),pads=pads,max_left_penetration_m=worst['penetration_m'],worst_pair=worst)
 results.append(r);(out/'final-contact-check.json').write_text(json.dumps(results,indent=2))
 print(delta,round(worst['penetration_m']*1000,3),worst['a']['mesh_name'],worst['b']['geom_name'],[round(p['gap_m']*1000,3) for p in pads],[round(p['normal_dot'],3) for p in pads],flush=True)
