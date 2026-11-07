"""Explicit manufacturing cutter, never a collision replacement for the case."""
from pathlib import Path
import json,struct
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;data=np.load(OUT/'case-sweep-source.npz');p=data['points_reference'];up=data['up_direction'];p=p[p[:,2]<=.1037];p=np.unique(np.vstack([p,p+.025*up]),axis=0);h=ConvexHull(p);eq=h.equations.copy();eq[:,3]-=.00005;v=HalfspaceIntersection(eq,p.mean(0)).intersections;hull=ConvexHull(v);np.savez_compressed(OUT/'case-swept-cutter.npz',vertices=v,faces=hull.simplices,equations=hull.equations)
manifest=json.load(open(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json'));rows=[]
for name in ['shoulder_connector_printed_0','shoulder_connector_printed_1']:
 rec=manifest['meshes'][name];path=Path(rec['grid_file']);dims=np.array(rec['shape']);origin=np.array(rec['origin']);spacing=rec['spacing'];grid=np.memmap(path,dtype='<f2',mode='r',offset=44,shape=tuple(dims[::-1]));z,y,x=np.nonzero(grid<0);points=origin+np.column_stack([x,y,z])*spacing;dist=grid[z,y,x].astype(float);inside=[]
 for start in range(0,len(points),4096):inside.extend(np.all(points[start:start+4096]@hull.equations[:,:3].T+hull.equations[:,3]<=1e-10,axis=1).tolist())
 inside=np.array(inside);rows.append({'constituent':name,'removed_interior_grid_samples':int(inside.sum()),'coarse_max_original_material_depth_m':float(-dist[inside].min()) if inside.any() else 0.,'coarse_removed_volume_mm3':float(inside.sum()*spacing**3*1e9),'removed_sample_bbox_m':[points[inside].min(0).tolist(),points[inside].max(0).tolist()] if inside.any() else None})
result={'cutter':'ConvexexternalenvelopeofactualcaseSURFACEbelowreceivermaxZ103.7mm,union25mmaxialsweep,50umexplicitmachiningclearance;notreplacementcollisionhull. SourcecaseSTLretainedunchanged.','cutter_vertices':len(v),'cutter_faces':len(hull.simplices),'cutter_bounds_m':[v.min(0).tolist(),v.max(0).tolist()],'pin_protected_min_z_gap_mm':float((v[:,2].min()-.07578)*1000),'coarse_precheck':rows,'scope':'Preliminarygridremovedmaterialbound;exactsourceBRepanddenseinsidechecksrequiredbeforeartifactselection.'};(OUT/'case-swept-cutter.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
