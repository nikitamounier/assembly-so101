from pathlib import Path
import argparse,json,numpy as np,trimesh,igl
ROOT=Path(__file__).resolve().parents[3]
def main(a):
 original=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text())['meshes'];fork=trimesh.load_mesh(a.mesh,process=False);horn=trimesh.load_mesh(original['servo_1_case']['source_file'],process=False);meshes={'fork':fork,'case':horn};samples={n:np.vstack((x.vertices,x.triangles_center)) for n,x in meshes.items()};trees={};data={}
 for n,m in meshes.items():
  v=np.ascontiguousarray(m.vertices,dtype=np.float64);f=np.ascontiguousarray(m.faces,dtype=np.int64);tree=igl.AABB();tree.init(v,f);trees[n]=tree;data[n]=(v,f)
 alignment=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/connector/alignment.json').read_text());axis=np.array(alignment['axis_world']);quat=np.array(alignment['start_qpos'])[23:27];from scipy.spatial.transform import Rotation
 R=Rotation.from_quat(quat,scalar_first=True).as_matrix();start=np.array(alignment['start_qpos'])[20:23];base=np.array([-.00607785,0,-.0003943]);rows=[]
 for f in np.linspace(0,1,a.poses):
  root=start+axis*.065*f;delta=(root-base)@R;pairrows=[]
  for source,target,offset in [('fork','case',delta),('case','fork',-delta)]:
   p=samples[source]+offset;b=meshes[target].bounds;mask=np.all((p>=b[0])&(p<=b[1]),axis=1);p=np.ascontiguousarray(p[mask]);v,faces=data[target]
   if not len(p):continue
   sq,_,_=trees[target].squared_distance(v,faces,p);w=igl.fast_winding_number(v,faces,p);sd=np.sqrt(sq)*np.where(w>.5,-1,1);i=int(sd.argmin());pairrows.append({'source':source,'target':target,'samples':len(p),'min_signed_triangle_distance_m':float(sd[i]),'max_overlap_m':max(0.,float(-sd[i])),'worst_target_source_point_m':p[i].tolist()})
  rows.append({'fraction':float(f),'pairs':pairrows,'max_triangle_overlap_m':max([x['max_overlap_m'] for x in pairrows],default=0.)});print(f,rows[-1]['max_triangle_overlap_m'],flush=True)
 result={'candidate_mesh':str(a.mesh.resolve()),'poses':a.poses,'sampling':'All original horn and candidate fork vertices+trianglecentroids, reciprocal exact nearesttriangle distance, fast winding source sign; no grid interpolation. Finite sampled path, not continuouscollisionproof.','rows':rows,'max_triangle_overlap_m':max(x['max_triangle_overlap_m'] for x in rows)};a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--poses',type=int,default=33);main(p.parse_args())
