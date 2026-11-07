"""Validate final field/source/world registration, including reused case frames."""
from pathlib import Path
import json,ctypes,hashlib,xml.etree.ElementTree as ET
import mujoco,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,surface_library
from scripts.check_wowrobo_contact_backend import grid_values
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 path=OUT/'grids/manifest.json';manifest=json.loads(path.read_text());assert manifest['status']=='complete';load_backend(path);m=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);fn=surface_library().wowrobo_sdf_distance;ptr=ctypes.POINTER(ctypes.c_double);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr];fn.restype=ctypes.c_double;rows={}
 for name,r in manifest['meshes'].items():
  if sha(r['grid_file'])!=r['grid_sha256']:raise RuntimeError('Grid identity mismatch '+name)
  mesh=trimesh.load_mesh(r['source_file'],process=False);points=np.vstack((mesh.bounds.mean(0),mesh.vertices[np.linspace(0,len(mesh.vertices)-1,16,dtype=int)]));visual=m.geom(name).id;contact=m.geom(name+'__contact').id;mi=int(m.geom_dataid[visual]);R=Rotation.from_quat(m.mesh_quat[mi],scalar_first=True).as_matrix();native=(points-m.mesh_pos[mi])@R;world=native@d.geom_xmat[visual].reshape(3,3).T+d.geom_xpos[visual]
  field=points
  if r.get('requires_sdf_geom_reframe'):
   transform=np.array(r['field_to_source']);field=(points-transform[:3,3])@transform[:3,:3]
  expected,_=grid_values(r['grid_file'],field);actual=np.array([fn(m._address,d._address,contact,np.ascontiguousarray(p).ctypes.data_as(ptr)) for p in world]);error=float(abs(actual-expected).max());rows[name]={'samples':len(points),'max_raw_field_vs_native_world_distance_error_m':error,'case_field_to_source_transform_applied':bool(r.get('requires_sdf_geom_reframe')),'passed':error<1e-9}
 flags=ET.parse(OUT/'scene.xml').getroot().findall("custom/numeric[@name='wowrobo_case_fields_reframed']");report={'nq':m.nq,'nv':m.nv,'plugin_count':m.nplugin,'sparse_jacobian':int(m.opt.jacobian)==1,'case_reframe_marker_count':len(flags),'rows':rows,'passed':m.nq==51 and m.nv==46 and m.nplugin==16 and len(flags)==1 and all(x['passed'] for x in rows.values()),'scope':'Source/field/world frame and frozen-file identity check only. Integrated insertion/release/negative controls on this final snapshot are pending parent execution.'};(OUT/'native-frame-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 if not report['passed']:raise RuntimeError('Final frame audit failed')
 receipt={'status':'FROZEN_FIELDS_AND_FRAME_AUDIT_COMPLETE','mechanical_qualification':'PENDING_FINAL_SNAPSHOT_RERUNS','scene':str((OUT/'scene.xml').resolve()),'scene_sha256':sha(OUT/'scene.xml'),'source_scene':str((OUT/'source-scene.xml').resolve()),'source_scene_sha256':sha(OUT/'source-scene.xml'),'manifest':str(path.resolve()),'manifest_sha256':sha(path),'frame_audit_sha256':sha(OUT/'native-frame-audit.json'),'nq':m.nq,'nv':m.nv,'plugin_count':m.nplugin,'fine_case_reframe_count':len(flags),'max_native_frame_error_m':max(x['max_raw_field_vs_native_world_distance_error_m'] for x in rows.values()),'frozen_assets':manifest['frozen_assets']};temp=OUT/'FINAL_READY.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(OUT/'FINAL_READY.json');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
