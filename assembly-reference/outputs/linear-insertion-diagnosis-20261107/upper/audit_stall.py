from pathlib import Path
import ctypes,json
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_contact_backend import load_backend,surface_library
from assembly_policy.so101.wowrobo_boundaries import world_mesh,penetration
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;p=ROOT/'outputs/linear-insertion-aligned-20261107/upper';load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(p/'guide-scene.xml'));d=mujoco.MjData(m);r=json.load(open(p/'receipt.json'));m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.body_contype[:]=0;m.body_conaffinity[:]=0
for key,a,b in [('active_contact_geoms',1,2),('receiver_contact_geoms',2,1)]:
 for name in r[key]:
  gid=m.geom(name).id;bid=m.geom_bodyid[gid];m.geom_contype[gid]=a;m.geom_conaffinity[gid]=b;m.body_contype[bid]|=a;m.body_conaffinity[bid]|=b
d.qpos[:]=np.load(p/'final-state.npz')['qpos'];mujoco.mj_forward(m,d)
fn=surface_library().wowrobo_sdf_distance;fn.restype=ctypes.c_double;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
rows=[]
for c in d.contact[:d.ncon]:
 row={'pair':[m.geom(c.geom1).name,m.geom(c.geom2).name],'native_distance_m':float(c.dist),'position_world_m':c.pos.tolist(),'normal_world':c.frame[:3].tolist(),'surfaces':[]}
 meshes=[]
 for gid in [c.geom1,c.geom2]:
  name=m.geom(gid).name.removesuffix('__contact');mesh=world_mesh(m,d,m.geom(name).id);meshes.append(mesh)
  point=np.array([c.pos]);closest,distance,face=trimesh.proximity.closest_point(mesh,point)
  row['surfaces'].append({'name':name,'distance_at_contact_native_sdf_m':fn(m._address,d._address,gid,(ctypes.c_double*3)(*c.pos)),'exact_unsigned_triangle_distance_at_contact_m':float(distance[0]),'exact_triangle_closest_world_m':closest[0].tolist(),'exact_triangle_face_normal_world':mesh.face_normals[face[0]].tolist(),'source_mesh_watertight':bool(mesh.is_watertight)})
 a,b=meshes;sa=np.vstack([a.vertices,a.triangles_center]);sb=np.vstack([b.vertices,b.triangles_center]);row['2048_reciprocal_source_sample_penetration_m']=max(penetration(a,sb,2048),penetration(b,sa,2048))
 rows.append(row)
result={'final_axial_m':float(d.qpos[0]),'rows':rows,'scope':'Actual final native contact compared with unchanged triangle surfaces; exact unsigned distances, finite2048surfaceinside samples. No geometry change; nonwatertight source sign remains provisional.'};(OUT/'stall-source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
