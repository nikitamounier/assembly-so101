from pathlib import Path
import json,mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import world_mesh
p=Path(__file__).resolve().parent;root=p.parents[2];r=json.load(open(p/'alignment.json'));m=mujoco.MjModel.from_xml_path(str(root/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m);d.qpos[:]=r['reference_qpos'];mujoco.mj_forward(m,d);result={}
for name in ['shoulder_connector_printed_1','servo_2_case']:
 mesh=world_mesh(m,d,m.geom(name).id);mesh.merge_vertices();groups={}
 for normal,area,center in zip(mesh.face_normals,mesh.area_faces,mesh.triangles_center):
  if area<1e-12:continue
  n=normal.copy()
  if n[np.argmax(abs(n))]<0:n=-n
  key=tuple(np.round(n,5));row=groups.setdefault(key,[0.,np.zeros(3)])
  row[0]+=area;row[1]+=area*center
 rows=[]
 for n,(area,center) in sorted(groups.items(),key=lambda x:-x[1][0])[:20]:rows.append({'normal_world':list(n),'area_m2':float(area),'centroid_world_m':(center/area).tolist()})
 result[name]=rows
(p/'feature-planes.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
