"""Conforming shared-node surface meshes from unchanged valid STEP solids."""
from pathlib import Path
import hashlib,json,time
import gmsh,numpy as np,trimesh
ROOT=Path.cwd();SOURCE=ROOT/'outputs/insertion-fix-20261107/upper/combined-clearance-v4';OUT=ROOT/'outputs/insertion-fix-20261107/coherence/conforming-mesh';OUT.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(mesh):
 _,c=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
 return {'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'boundary_edges':int((c==1).sum()),'nonmanifold_edges':int((c>2).sum()),'signed_volume_mm3':float(mesh.volume*1e9),'zero_area_faces':int((mesh.area_faces<1e-24).sum()),'euler_number':int(mesh.euler_number)}
gmsh.initialize();gmsh.option.setNumber('General.Terminal',1);gmsh.option.setNumber('General.NumThreads',4);gmsh.option.setNumber('Mesh.MeshSizeMin',.08);gmsh.option.setNumber('Mesh.MeshSizeMax',.7);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',48);gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0);gmsh.option.setNumber('Mesh.Algorithm',6);gmsh.option.setNumber('Mesh.AlgorithmSwitchOnFailure',1);gmsh.option.setNumber('Mesh.ElementOrder',1);results=[]
for index in [0,1]:
 name=f'shoulder_connector_printed_{index}';source=SOURCE/f'{name}.step';before=digest(source);start=time.monotonic();gmsh.clear();gmsh.model.add(name);entities=gmsh.model.occ.importShapes(str(source));gmsh.model.occ.synchronize();volumes=gmsh.model.getEntities(3);assert len(volumes)==1;cad_volume=gmsh.model.occ.getMass(*volumes[0]);print('START',name,'vol',cad_volume,'surfaces',len(gmsh.model.getEntities(2)),flush=True);gmsh.model.mesh.generate(2);tags,coords,_=gmsh.model.mesh.getNodes();vertices=np.array(coords).reshape(-1,3);sort=np.argsort(tags);sortedtags=np.array(tags)[sort];_,nt=gmsh.model.mesh.getElementsByType(2);tri=sort[np.searchsorted(sortedtags,np.array(nt).reshape(-1,3))];mesh=trimesh.Trimesh(vertices/1000,tri,process=False);rawstats=stats(mesh);orientation='native Gmsh orientation'
 if not mesh.is_winding_consistent:
  signed=gmsh.model.getBoundary(volumes,combined=False,oriented=True,recursive=False)
  reverse=[(2,abs(tag))for dim,tag in signed if tag<0];gmsh.model.mesh.reverse(reverse);_,nt=gmsh.model.mesh.getElementsByType(2);tri=sort[np.searchsorted(sortedtags,np.array(nt).reshape(-1,3))];mesh=trimesh.Trimesh(vertices/1000,tri,process=False);orientation='oriented from signed volume boundary'
 if mesh.volume<0:mesh.invert();orientation+='; globally reversed outward'
 resultstats=stats(mesh)
 if not mesh.is_watertight or not mesh.is_winding_consistent:raise RuntimeError(f'Conforming topology failure {name}: {resultstats}')
 # Every triangle centroid and edge midpoint is compared with its exact CAD
 # support surface. Orthogonal surface distance is a finite deflection audit.
 maximum=0.;worst=[];surface_records=[]
 for _,tag in gmsh.model.getEntities(2):
  _,localnodes=gmsh.model.mesh.getElementsByType(2,tag);localtri=sort[np.searchsorted(sortedtags,np.array(localnodes).reshape(-1,3))]
  if not len(localtri):continue
  vv=vertices[localtri];q=np.vstack((vv.mean(1),(vv[:,0]+vv[:,1])/2,(vv[:,1]+vv[:,2])/2,(vv[:,2]+vv[:,0])/2));closest,param=gmsh.model.getClosestPoint(2,tag,q.ravel());projected=np.array(closest).reshape(-1,3);dist=np.linalg.norm(projected-q,axis=1);mx=float(dist.max());maximum=max(maximum,mx);ids=np.argsort(dist)[-3:];worst.extend([{'surface':int(tag),'point_mm':q[i].tolist(),'surface_distance_mm':float(dist[i])}for i in ids]);surface_records.append({'surface':int(tag),'triangles':len(localtri),'max_centroid_edge_midpoint_distance_mm':mx})
 output=OUT/f'{name}.obj';mesh.export(output);roundtrip=trimesh.load_mesh(output,process=False);roundstats=stats(roundtrip);assert roundstats['watertight']and roundstats['winding_consistent'];assert maximum<=.04,f'{name} sample deflection {maximum} mm exceeds40um';assert digest(source)==before
 result={'name':name,'source_STEP':str(source),'source_sha256':before,'source_unchanged':True,'output_OBJ':str(output),'output_sha256':digest(output),'mesh_stats':resultstats,'raw_stats':rawstats,'OBJ_roundtrip_stats':roundstats,'orientation':orientation,'OCC_volume_mm3':cad_volume,'relative_volume_difference':float((mesh.volume*1e9-cad_volume)/cad_volume),'max_centroid_edge_midpoint_CAD_support_distance_mm':maximum,'surface_records':surface_records,'worst_support_projection_queries':sorted(worst,key=lambda x:x['surface_distance_mm'],reverse=True)[:50],'elapsed_s':time.monotonic()-start};results.append(result);(OUT/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',name,resultstats,'deflectionmm',maximum,flush=True)
gmsh.finalize();(OUT/'manifest.json').write_text(json.dumps({'gmsh_version':gmsh.__version__,'method':'Conforming global-node surface-only mesh from unchanged STEP; no tetrahedra, CAD repairs, cuts or source overwrites.','options':{'minimum_size_mm':.08,'maximum_size_mm':.7,'curvature_segments_per_circle':48,'extend_boundary_sizes':False,'algorithm':6},'parts':results,'validation_scope':'Watertight oriented two-manifold edge topology; signed volume positive; every triangle centroid and edge midpoint projected to its exact CAD support surface. Support projection can lie outside trimmed face; independent trimmed-face spot checks remain required before treating this as a Hausdorff certificate.'},indent=2)+'\n')
