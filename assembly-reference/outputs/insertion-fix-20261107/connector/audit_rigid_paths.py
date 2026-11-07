from pathlib import Path
import json,numpy as np,trimesh,igl,mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];out=Path(__file__).parent;p=ROOT/'outputs/linear-insertion-aligned-20261107/connector/audit_alignment.py';ctx={'__file__':str(p)};code=p.read_text();exec(code[:code.index('qa=int')],ctx);m=ctx['m'];d=ctx['d'];a=json.loads((p.parent/'alignment.json').read_text());reference=np.array(a['reference_qpos']);normal=np.array(a['receiving_face_normal_world']);axis=np.array(a['axis_world']);qa=20;ja=int(m.joint('target_servo_1_shaft').qposadr[0]);d.qpos[:]=reference;mujoco.mj_forward(m,d);pivot=d.xanchor[m.joint('target_servo_1_shaft').id].copy();cross=np.cross(normal,axis);R0=Rotation.from_quat(reference[qa+3:qa+7],scalar_first=True).as_matrix()
def audit(q):
 depth,pairs=ctx['audit'](q);worst=max(pairs,key=lambda x:x['max_overlap_m']) if pairs else None;exact=0
 if worst and depth>0:
  mesh=ctx['meshes'][worst['target']];point=np.array([worst['worst_point_target_source_m']]);dist=trimesh.proximity.closest_point(mesh,point)[1][0];w=igl.winding_number(np.ascontiguousarray(mesh.vertices),np.ascontiguousarray(mesh.faces,dtype=np.int64),point)[0];exact=float(dist if w>.5 else 0)
 return {'max_source_sdf_overlap_m':depth,'exact_triangle_witness_overlap_m':exact,'worst_pair':worst}
relative=[]
for degree in [-6,-3,0,3,6]:
 q=reference.copy();q[ja]=np.deg2rad(degree);r=audit(q);r['relative_shaft_index_deg']=degree;relative.append(r);print('index',degree,r['max_source_sdf_overlap_m'],flush=True)
rock=[]
for degree in [-15,-12,-9,-6,-3,0,3,6,9,12,15]:
 R=Rotation.from_rotvec(cross*np.deg2rad(degree)).as_matrix()
 for normalshift in [-.001,-.0005,0,.0005,.001]:
  q=reference.copy();q[qa:qa+3]=pivot+R@(reference[qa:qa+3]-pivot)+normalshift*normal;q[qa+3:qa+7]=Rotation.from_matrix(R@R0).as_quat(scalar_first=True);r=audit(q);r.update(rocking_angle_deg=degree,normal_shift_m=normalshift);rock.append(r)
 print('rock',degree,min(x['max_source_sdf_overlap_m'] for x in rock if x['rocking_angle_deg']==degree),flush=True)
r={'scope':'Independent STATIC endpoint feasibility tests, source CAD and default poses unchanged. Shaft indexing is relative (fork fixed), unlike previous coherent scan; rocking pivots fork about actual shaft anchor and crossspan axis.','relative_shaft_index_scan':relative,'rocking_endpoint_scan':rock,'best_relative_index':min(relative,key=lambda x:x['max_source_sdf_overlap_m']),'best_rocking_endpoint':min(rock,key=lambda x:x['max_source_sdf_overlap_m']),'normal_shift_scan_is_relative_to_current_balanced_fit':True};(out/'rigid-endpoint-audit.json').write_text(json.dumps(r,indent=2)+'\n')
