import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import world_mesh
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);report={}
for servo,task in [(2,'upper_arm_module'),(3,'forearm_module')]:
 h=m.joint(f'target_servo_{servo}_shaft');axis=d.xaxis[h.id];anchor=d.xanchor[h.id];rows={}
 for g in range(m.ngeom):
  name=m.geom(g).name or ''
  if name not in [f'servo_{servo}_horns',f'servo_{servo}_case'] and not name.startswith(task+'_printed'):continue
  if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
  mesh=world_mesh(m,d,g);v=mesh.vertices-anchor;ax=v@axis;rad=np.linalg.norm(v-ax[:,None]*axis,axis=1);tri=mesh.triangles_center-anchor;tax=tri@axis;trad=np.linalg.norm(tri-tax[:,None]*axis,axis=1);an=mesh.face_normals@axis;row={}
  for band in [(0,.0096),(.0096,.016),(.016,.025),(.025,.05)]:
   mask=(trad>=band[0])&(trad<band[1])&(np.abs(an)>.99);values=np.round(tax[mask]*1e6).astype(int);areas=mesh.area_faces[mask];sums={}
   for a,area in zip(values,areas):sums[int(a)]=sums.get(int(a),0.)+float(area)
   row[str(band)]={'vertex_axial_range_m':[float(ax[(rad>=band[0])&(rad<band[1])].min()),float(ax[(rad>=band[0])&(rad<band[1])].max())]if np.any((rad>=band[0])&(rad<band[1]))else None,'axial_face_planes_by_area':sorted([{'axial_m':a*1e-6,'area_m2':area}for a,area in sums.items()],key=lambda x:-x['area_m2'])[:10]}
  rows[name]=row
 report[str(servo)]=rows
 print(servo,json.dumps(rows,indent=2),flush=True)
(OUT/'axial-registration.json').write_text(json.dumps(report,indent=2)+'\n')
