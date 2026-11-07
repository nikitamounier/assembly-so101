from pathlib import Path
import json,ctypes
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_case_grid_reuse import build_profile,reframe_scene
from assembly_policy.so101.wowrobo_contact_backend import prepare_scene,surface_library
from scripts.check_wowrobo_contact_backend import grid_values
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
manifest_path=build_profile(ROOT/'outputs/insertion-fix-20261107/combined-v1/grids/manifest.json',ROOT/'outputs/insertion-fix-20261107/pin/consistent-v3/grids/manifest.json',OUT/'grids');prepare_scene(ROOT/'outputs/insertion-fix-20261107/combined-v1/source-scene.xml',manifest_path,OUT/'unframed.xml');reframe_scene(OUT/'unframed.xml',manifest_path,OUT/'scene.xml');m=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));d=mujoco.MjData(m);rows=json.loads(manifest_path.read_text())['meshes'];query=surface_library().wowrobo_sdf_distance;ptr=ctypes.POINTER(ctypes.c_double);query.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr];query.restype=ctypes.c_double;report=[]
for state in ['pin','upper','final']:
 d.qpos[:]=json.loads((ROOT/f'outputs/insertion-fix-20261107/coherence/{state}-alignment.json').read_text())['start_qpos'];mujoco.mj_forward(m,d)
 for name,row in rows.items():
  if not row.get('field_to_source'):continue
  mesh=trimesh.load_mesh(row['source_file'],process=False);points=np.array(mesh.vertices)[np.linspace(0,len(mesh.vertices)-1,256,dtype=int)];g=m.geom(name).id;c=m.geom(name+'__contact').id;mid=int(m.geom_dataid[g]);r=Rotation.from_quat(m.mesh_quat[mid],scalar_first=True).as_matrix();principal=(points-m.mesh_pos[mid])@r;world=principal@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];mapp=np.array(row['field_to_source']);field=(points-mapp[:3,3])@mapp[:3,:3];expected,_=grid_values(row['grid_file'],field);actual=np.array([query(m._address,d._address,c,np.asarray(p,dtype=np.float64).ctypes.data_as(ptr))for p in world]);err=float(np.max(abs(expected-actual)));assert err<1e-12,(state,name,err);report.append({'state':state,'case':name,'samples':len(points),'native_frame_error_m':err,'source_vertex_rigid_match_max_error_m':row['source_rigid_match_max_error_m']})
(OUT/'validation.json').write_text(json.dumps({'pass':True,'case_geometry_changed':False,'field':'.1mm full sourcecase1 reused by rigidmaps, all originalcasevisuals unchanged','manifest':str(manifest_path),'scene':str(OUT/'scene.xml'),'checks':report},indent=2)+'\n');print(json.dumps(report,indent=2))
