"""Dense removed-VOLUME and boundary checks against untouched original source."""
from pathlib import Path
import json,hashlib
import igl,numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'combined-clearance-v4';manifest=json.load(open(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json'));rows=[]

def interp(grid,points,origin,h):
 c=(points-origin)/h;base=np.floor(c).astype(int);base=np.clip(base,0,np.array(grid.shape[::-1])-2);f=np.clip(c-base,0,1);r=np.zeros(len(points))
 for z in (0,1):
  for y in (0,1):
   for x in (0,1):
    w=np.prod(np.where([x,y,z],f,1-f),axis=1);r+=w*grid[base[:,2]+z,base[:,1]+y,base[:,0]+x]
 return r
for index in range(2):
 name=f'shoulder_connector_printed_{index}';original=trimesh.load_mesh(ROOT/f'assets_local/wowrobo/components-v2/meshes/{name}.obj',process=False);candidate=trimesh.load_mesh(OUT/f'{name}.obj',process=False);v=np.array(original.vertices,dtype=float);f=np.array(original.faces,dtype=np.int64);cv=np.array(candidate.vertices,dtype=float);cf=np.array(candidate.faces,dtype=np.int64);tree=igl.AABB();tree.init(v,f)
 surface=np.vstack([cv,cv[cf].mean(1)]);sq,_,_=tree.squared_distance(v,f,surface);dist=np.sqrt(sq);boundary_max=float(dist.max());mask=surface[:,2]>.08;upper_boundary_max=float(dist[mask].max());print('boundary',index,len(surface),boundary_max,upper_boundary_max,flush=True)
 rec=manifest['meshes'][name];grid=np.memmap(rec['grid_file'],dtype='<f2',mode='r',offset=44,shape=tuple(rec['shape'][::-1]));origin=np.array(rec['origin']);h=rec['spacing'];spacing=.00015;x=np.arange(-.020,.0231,spacing);y=np.arange(-.052,-.0249,spacing);rows_removed=[];rows_added=[];total_classified=0
 for z in np.arange(.084,.1041,spacing):
  xx,yy=np.meshgrid(x,y,indexing='xy');q=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,z)]);coarse=interp(grid,q,origin,h);q=q[coarse<.00025]
  if not len(q):continue
  ow=igl.fast_winding_number(v,f,q);cw=igl.fast_winding_number(cv,cf,q);removed=(ow>.5)&(cw<=.5);added=(ow<=.5)&(cw>.5)
  if removed.any():rows_removed.append(q[removed])
  if added.any():rows_added.append(q[added])
  total_classified+=len(q)
 removed=np.vstack(rows_removed) if rows_removed else np.empty((0,3));added=np.vstack(rows_added) if rows_added else np.empty((0,3));rsq,_,_=tree.squared_distance(v,f,removed) if len(removed) else (np.array([]),None,None);rd=np.sqrt(rsq);asq,_,_=tree.squared_distance(v,f,added) if len(added) else (np.array([]),None,None);ad=np.sqrt(asq);row={'constituent':name,'derived_mesh_sha256':hashlib.sha256((OUT/f'{name}.obj').read_bytes()).hexdigest(),'all_derived_boundary_samples':len(surface),'all_boundary_max_original_distance_m':boundary_max,'upper_region_boundary_max_original_distance_m':upper_boundary_max,'interior_sample_spacing_m':spacing,'interior_classified_points':total_classified,'removed_interior_samples':len(removed),'max_removed_volume_source_depth_m':float(rd.max()) if len(rd) else 0.,'removed_volume_estimate_mm3':float(len(removed)*spacing**3*1e9),'added_samples':len(added),'max_added_source_distance_m':float(ad.max()) if len(ad) else 0.,'deepest_removed_point_m':removed[int(np.argmax(rd))].tolist() if len(rd) else None,'material_depth_guard_pass':not len(rd) or float(rd.max())<=.0005,'boundary_guard_pass':boundary_max<=.00052,'exact_pin_band_z_mm':[75.5,76.0],'upper_case_tool_min_z_mm':84.50649852995974};rows.append(row);np.savez_compressed(OUT/f'{name}-removed-volume-samples.npz',points=removed,source_distance=rd,added_points=added,added_source_distance=ad);print(json.dumps(row),flush=True)
report={'scope':'Actualremovedvolume at0.15mmgrid inupperpocket, original/candidatefastwindinginsideclassification plus exactoriginaltriangleboundarydistance;allnewboundaryvertices+centroids. Finiteverification,notcontinuouscertificate. Lowerhorn/postsourceproofseparate. NonwatertightsourcewindingambiguityrequiresBRepcrosschecks ofdeepestwitnesses.','rows':rows,'all_material_guards_pass':all(r['material_depth_guard_pass'] for r in rows),'all_boundary_guards_pass':all(r['boundary_guard_pass'] for r in rows)};(OUT/'derived-material-audit.json').write_text(json.dumps(report,indent=2)+'\n')
