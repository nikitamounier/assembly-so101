import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_physical_resets import ROOT
from assembly_policy.so101.wowrobo_boundaries import world_mesh
out=ROOT/'outputs/equilibrium-model-20261107'
r=json.loads((ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json').read_text())
n=mujoco.MjModel.from_xml_path(r['audit_model_path'])
z=np.load(ROOT/'outputs/cover-support-fix-20261107/physical-precontact/18_upper_floor_grasp.npz')
# INTEGRATION starts at t, qpos. qpos state offset is 1.
qs={'input':z['input_qpos'],'settled':z['settled_state'][1:1+n.nq]}
allrows={}
for label,q in qs.items():
 d=mujoco.MjData(n);d.qpos[:]=q;mujoco.mj_forward(n,d)
 site=n.site('left_gripperframe').id;frame=d.site_xmat[site].reshape(3,3)
 target=world_mesh(n,d,n.geom('base_side_cover_printed_0').id);rows=[]
 for g in [19,20]:
  hand=world_mesh(n,d,g);points=np.vstack([hand.vertices,hand.triangles_center]);normals=np.vstack([hand.vertex_normals,hand.face_normals]);local=(points-d.site_xpos[site])@frame;ln=normals@frame
  mask=(local[:,0]>-.035)&(local[:,0]<.004)&(abs(local[:,1])<.010)&(ln[:,2]>.25 if g==19 else ln[:,2]<-.25)
  pts=points[mask];near,dist,faces=trimesh.proximity.closest_point(target,pts);k=int(np.argmin(dist))
  rows.append(dict(geom=g,source_mesh=n.mesh(n.geom_dataid[g]).name,gap_m=float(dist[k]),point_site_m=((pts[k]-d.site_xpos[site])@frame).tolist(),toward_cover_world_m=(near[k]-pts[k]).tolist(),pad_point_world_m=pts[k].tolist(),cover_point_world_m=near[k].tolist()))
 allrows[label]=rows
 print(label,rows)
(out/'left-mesh-gap.json').write_text(json.dumps(allrows,indent=2)+'\n')
