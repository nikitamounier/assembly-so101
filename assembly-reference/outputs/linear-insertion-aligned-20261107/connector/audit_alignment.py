from pathlib import Path
import json,sys,importlib.util
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('gridcheck',ROOT/'scripts/check_wowrobo_contact_backend.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
SOURCE=ROOT/'outputs/linear-insertion-20261107/source-snapshot'
M=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text())['meshes'];m=mujoco.MjModel.from_xml_path(str(SOURCE/'scene.xml'));d=mujoco.MjData(m)
q=[]
for s in ['08_connector_aligned','11_connector_seated']:
 mujoco.mj_setState(m,d,np.load(SOURCE/'stages'/f'{s}_train.npz')['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);q.append(d.qpos.copy())
active=['shoulder_connector_printed_0','shoulder_connector_printed_1'];receiver=['base_servo_printed_0','servo_1_case','servo_1_horns','base_side_cover_printed_0'];meshes={};samples={}
for name in active+receiver:
 mesh=trimesh.load_mesh(M[name]['source_file'],process=False);meshes[name]=mesh;samples[name]=np.vstack((mesh.vertices,mesh.triangles_center))

def transform(name):
 geom=m.geom(name).id;mid=int(m.geom_dataid[geom]);rot=Rotation.from_quat(m.mesh_quat[mid],scalar_first=True).as_matrix();world_r=d.geom_xmat[geom].reshape(3,3)@rot.T;world_p=d.geom_xpos[geom]-world_r@m.mesh_pos[mid];return world_r,world_p

def audit(qpos):
 d.qpos[:]=qpos;mujoco.mj_forward(m,d);trans={n:transform(n) for n in meshes};worst=0;rows=[]
 for a in active:
  for b in receiver:
   for source,target in [(a,b),(b,a)]:
    rs,ps=trans[source];rt,pt=trans[target];points=(samples[source]@rs.T+ps-pt)@rt
    bounds=np.array(M[target]['bounds']);mask=np.all((points>=bounds[0])&(points<=bounds[1]),axis=1);points=points[mask]
    if len(points):
     values,_=g.grid_values(M[target]['grid_file'],points);idx=int(np.argmin(values));depth=max(0.,float(-values[idx]));worst=max(worst,depth);rows.append({'source':source,'target':target,'max_overlap_m':depth,'sample_count':len(points),'worst_point_target_source_m':points[idx].tolist()})
 return worst,rows

qa=int(m.joint('shoulder_connector_free').qposadr[0]);start,final=q;delta=final[qa:qa+3]-start[qa:qa+3];axis=delta/np.linalg.norm(delta);out=[]
for f in np.linspace(0,1,17):
 state=start.copy();state[qa:qa+3]+=f*delta;worst,rows=audit(state);out.append({'fraction':float(f),'travel_m':float(f*np.linalg.norm(delta)),'max_sampled_overlap_m':worst,'pairs':rows});print(f,worst,flush=True)
report={'source_start_qpos':start.tolist(),'source_reference_qpos':final.tolist(),'source_start':'08_connector_aligned','source_reference':'11_connector_seated','axis_world':axis.tolist(),'authored_travel_m':float(np.linalg.norm(delta)),'orientation_identical':bool(np.array_equal(start[qa+3:qa+7],final[qa+3:qa+7])),'root_qpos_address':qa,'finite_sweep':out,'max_sampled_overlap_m':max(r['max_sampled_overlap_m'] for r in out),'method':'All original-source vertices and triangle centroids, bidirectional, trilinear source-SDF queries. Finite 17-pose sweep; not continuous clearance proof.'}
Path(__file__).with_name('axis-audit.json').write_text(json.dumps(report,indent=2)+'\n')
