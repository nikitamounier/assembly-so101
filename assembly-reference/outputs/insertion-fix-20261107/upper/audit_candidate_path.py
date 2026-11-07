"""Full unchanged source-case surface set against derived receiver CAD meshes."""
from pathlib import Path
import json
import igl,numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'combined-clearance-v4';a=np.load(OUT.parent/'case-sweep-source.npz');points=a['points_reference'];up=a['up_direction'];results=[]
for index in range(2):
 mesh=trimesh.load_mesh(OUT/f'shoulder_connector_printed_{index}.obj',process=False);v=np.array(mesh.vertices,dtype=float);f=np.array(mesh.faces,dtype=np.int64);tree=igl.AABB();tree.init(v,f);rows=[]
 for depth in np.linspace(.025,0,33):
  p=points+up*depth;lo,hi=mesh.bounds;selected=p[np.all((p>=lo)&(p<=hi),axis=1)]
  if len(selected):
   inside=igl.fast_winding_number(v,f,selected)>.5;bad=selected[inside];sq,_,_=tree.squared_distance(v,f,bad) if len(bad) else (np.array([]),None,None);dist=np.sqrt(sq);maximum=float(dist.max()) if len(dist) else 0.;witness=bad[int(np.argmax(dist))].tolist() if len(dist) else None
  else:maximum=0.;witness=None;bad=[]
  rows.append({'depth_remaining_m':float(depth),'candidate_inside_source_surface_samples':len(bad),'maximum_penetration_m':maximum,'deepest_witness_source_m':witness})
 result={'constituent':index,'max_full_path_source_case_penetration_m':max(r['maximum_penetration_m'] for r in rows),'samples_per_pose':len(points),'pose_count':33,'rows':rows};results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
report={'scope':'All223kuntouchedsourcecasevertices+trianglecentroidsper33axialposes,AABBfiltered,derivedreceiverCADmeshfastwindingandexacttriangledistance. Finiteproof,notcontinuouscertificate. Nativefullstroke/blockedwall/releasepending.','results':results,'maximum_penetration_m':max(r['max_full_path_source_case_penetration_m'] for r in results)};(OUT/'full-source-path-audit.json').write_text(json.dumps(report,indent=2)+'\n')
