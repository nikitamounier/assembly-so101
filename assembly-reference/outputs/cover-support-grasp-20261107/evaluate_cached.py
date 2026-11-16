import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit,world_mesh
OUT=Path(__file__).resolve().parent
rows=json.loads((OUT/'cached-candidates.json').read_text())
m=mujoco.MjModel.from_xml_path('assets_local/wowrobo/photo-resets-repaired-v1/scene.xml')
d=mujoco.MjData(m)
with np.load('assets_local/wowrobo/photo-resets-repaired-v1/stages/18_upper_floor_grasp_train.npz') as f:mujoco.mj_setState(m,d,f['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
base=d.qpos.copy();lq=[m.joint('left_'+n).qposadr[0] for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']];site=m.site('left_gripperframe').id
A=NativeBoundaryAudit(m)
for index in [14,15,12,3,0]:
 row=next(r for r in rows if r['i']==index)
 d.qpos[:]=base;d.qpos[lq]=row['q_left'];mujoco.mj_forward(m,d)
 cg=next(g for g in A.targets if A.info[g]['piece']=='base_side_cover');target=world_mesh(m,d,cg);frame=d.site_xmat[site].reshape(3,3)
 pads=[]
 for hg in [19,20]:
  hand=world_mesh(m,d,hg);points=np.vstack([hand.vertices,hand.triangles_center]);normals=np.vstack([hand.vertex_normals,hand.face_normals]);local=(points-d.site_xpos[site])@frame;ln=normals@frame
  mask=(local[:,0]>-.025)&(local[:,0]<.008)&(abs(local[:,1])<.010)&(ln[:,2]>.5 if hg==19 else ln[:,2]<-.5)
  pts=points[mask];ns=normals[mask]
  near,dist,faces=trimesh.proximity.closest_point(target,pts);k=int(np.argmin(dist))
  pads.append(dict(gap_m=float(dist[k]),point_world_m=pts[k].tolist(),target_point_world_m=near[k].tolist(),pad_normal_world=ns[k].tolist(),target_normal_world=target.face_normals[faces[k]].tolist(),normal_dot=float(ns[k]@target.face_normals[faces[k]]),point_site_m=((pts[k]-d.site_xpos[site])@frame).tolist(),geom=hg,source_mesh=m.mesh(m.geom_dataid[hg]).name))
 row['pads']=pads;row['target_normals_dot']=float(np.dot(pads[0]['target_normal_world'],pads[1]['target_normal_world']))
 print(index,row['max_left_penetration_m'],[p['gap_m'] for p in pads],[p['normal_dot'] for p in pads],row['target_normals_dot'],flush=True)
 (OUT/f'evaluated-{index}.json').write_text(json.dumps(row,indent=2))
