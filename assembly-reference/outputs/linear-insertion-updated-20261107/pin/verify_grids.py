"""Independent pin diameter and receiving corridor check against source triangles."""
from pathlib import Path
import hashlib,json,struct,sys
import igl,mujoco,numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[3]
GRID=ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json'
def field(record,points):
 p=Path(record['grid_file'])
 with p.open('rb') as f:nx,ny,nz,ox,oy,oz,h=struct.unpack('<3i4d',f.read(44))
 grid=np.memmap(p,dtype='<f2',mode='r',offset=44,shape=(nz,ny,nx))
 ijk=(points-np.array([ox,oy,oz]))/h;base=np.floor(ijk).astype(int);frac=ijk-base
 if (base<0).any() or (base>=np.array([nx,ny,nz])-1).any():raise ValueError('query outside grid')
 value=np.zeros(len(points))
 for z in(0,1):
  for y in(0,1):
   for x in(0,1):
    index=base+[x,y,z];weight=np.prod(np.where([x,y,z],frac,1-frac),axis=1)
    value+=weight*grid[index[:,2],index[:,1],index[:,0]].astype(float)
 return value

def main():
 manifest=json.loads(GRID.read_text());records=manifest['meshes'];pin=records['pin']
 # Exact original source pin axis, with 0.2mm grid and no convexification.
 x=np.linspace(-.002,.002,1601);points=np.column_stack([x,np.zeros(len(x)),np.full(len(x),.039)])
 value=field(pin,points);cross=np.flatnonzero(value[:-1]*value[1:]<0)
 roots=[float(x[i]-value[i]*(x[i+1]-x[i])/(value[i+1]-value[i])) for i in cross]
 mesh=trimesh.load_mesh(pin['source_file'],process=False)
 section=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,.039]).reshape(-1,3)
 truth_diameter=float(np.ptp(section[:,0]));diameter=roots[-1]-roots[0] if len(roots)==2 else None
 source=ROOT/'outputs/linear-insertion-20261107/source-snapshot';m=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));d=mujoco.MjData(m)
 mujoco.mj_setState(m,d,np.load(source/'stages/13_pin_aligned_train.npz')['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,d)
 a=m.joint('assembly_pin_free').qposadr[0];origin=d.qpos[a:a+3]
 z=np.linspace(origin[2]-.05-.032,origin[2]-.05,129);angles=np.arange(16)*2*np.pi/16
 xy=np.vstack([[0,0],np.column_stack([np.cos(angles),np.sin(angles)])*.00135]);world=np.vstack([np.column_stack([origin[0]+off[0]+np.zeros(len(z)),origin[1]+off[1]+np.zeros(len(z)),z])for off in xy])
 rows=[]
 for name in ['base_servo_printed_0','servo_1_case','servo_1_horns','base_side_cover_printed_0','shoulder_connector_printed_0','shoulder_connector_printed_1']:
  rec=records[name];gid=m.geom(name).id;mid=m.geom_dataid[gid];native=(world-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3,3)
  rot=np.empty(9);mujoco.mju_quat2Mat(rot,m.mesh_quat[mid]);local=native@rot.reshape(3,3).T+m.mesh_pos[mid]
  mesh=trimesh.load_mesh(rec['source_file'],process=False);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces,dtype=np.int64)
  tree=igl.AABB();tree.init(v,f);sq,_,_=tree.squared_distance(v,f,local);w=igl.fast_winding_number(v,f,local);truth=np.sqrt(sq)*np.where(w>.5,-1,1)
  # Grid plugin outside-grid extension clamps then adds Euclidean distance.
  with Path(rec['grid_file']).open('rb')as stream:nx,ny,nz,ox,oy,oz,h=struct.unpack('<3i4d',stream.read(44))
  low=np.array([ox,oy,oz]);high=low+(np.array([nx,ny,nz])-1)*h
  clipped=np.clip(local,low+1e-12,high-1e-12);sdf=field(rec,clipped)+np.linalg.norm(local-clipped,axis=1)
  error=np.abs(sdf-truth)
  rows.append({'mesh':name,'query_count':len(world),'source_watertight':bool(mesh.is_watertight),'grid_pitch_mm':h*1000,'triangle_min_distance_mm':float(truth.min()*1000),'grid_min_distance_mm':float(sdf.min()*1000),'maximum_distance_error_mm':float(error.max()*1000),'p99_distance_error_mm':float(np.quantile(error,.99)*1000),'false_filled_clear_points_gt1pitch':int(np.count_nonzero((truth>h)&(sdf<0))),'sign_disagreements_gt1pitch':int(np.count_nonzero((np.abs(truth)>h)&(truth*sdf<0)))})
 report={'grid_manifest_sha256':hashlib.sha256(GRID.read_bytes()).hexdigest(),'pin_pitch_mm':pin['spacing']*1000,'source_shaft_diameter_mm':truth_diameter*1000,'sdf_shaft_crossing_x_mm':[r*1000 for r in roots],'sdf_shaft_diameter_mm':diameter*1000 if diameter else None,'shaft_diameter_error_mm':abs(diameter-truth_diameter)*1000 if diameter else None,'shaft_resolution_cells':truth_diameter/pin['spacing'],'receiver_corridor_samples':rows,'method':'Independent trilinear interpolation vs original OBJ triangle AABB distance + fast winding sign. Corridor is current pin shaft axis and16radius1.35mmring lines over32mm travel.','limitations':'Finite samples; open source meshes retain uncertain winding sign. No contact/dynamics qualification.'}
 (Path(__file__).parent/'grid-geometry-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
