from pathlib import Path
import hashlib,json,collections,time
import numpy as np,trimesh,cadquery as cq
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepTools import BRepTools
ROOT=Path.cwd();FOLDER=ROOT/'outputs/insertion-fix-20261107/upper/combined-clearance-v4';OUT=ROOT/'outputs/insertion-fix-20261107/coherence/upper-export-review.json';meta=json.loads((FOLDER/'combined-candidate.json').read_text());files=[p for p in FOLDER.iterdir()if p.is_file()];files +=[Path(meta['source_step']),ROOT/'assets_local/wowrobo/source/STEP/SO101/SO101 Assembly.step']+[ROOT/f'assets_local/wowrobo/components-v2/meshes/{n}.obj'for n in ['shoulder_connector_printed_0','shoulder_connector_printed_1','servo_2_case']];files=sorted(set(p for p in files if p.exists()))
def hashes():return {str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}for p in files}
before=hashes()
def check(shape,details=False):
 analyzer=BRepCheck_Analyzer(shape.wrapped,True,False,False);errors=[];counts=collections.Counter()
 if details and not analyzer.IsValid():
  for typ,parts in [('shape',[shape]),('solid',shape.Solids()),('shell',shape.Shells()),('face',shape.Faces()),('wire',shape.Wires()),('edge',shape.Edges()),('vertex',shape.Vertices())]:
   for i,p in enumerate(parts):
    result=analyzer.Result(p.wrapped)
    if result is None:continue
    statuses=[str(s)for s in result.Status()if not str(s).endswith('BRepCheck_NoError')]
    for status in statuses:counts[status]+=1
    if statuses and len(errors)<50:errors.append({'type':typ,'index':i,'statuses':statuses})
 return {'cadquery_valid':bool(shape.isValid()),'OCC_BRepCheck_valid':bool(analyzer.IsValid()),'solid_count':len(shape.Solids()),'volume_mm3':float(shape.Volume()),'faces':len(shape.Faces()),'error_counts':dict(counts),'errors_first50':errors}
steps=[]
for name in ['shoulder_connector_printed_0.step','shoulder_connector_printed_1.step','shoulder_connector_combined_clearance.step']:
 shape=cq.importers.importStep(str(FOLDER/name)).val();entry={'file':name,'reimport':check(shape,True)}
 if name!='shoulder_connector_combined_clearance.step':
  start=time.monotonic();vertices,faces=shape.tessellate(.04,.08);entry['tessellation_counts']={'vertices':len(vertices),'faces':len(faces),'seconds':time.monotonic()-start};entry['after_in_memory_tessellate']=check(shape,True);BRepTools.Clean_s(shape.wrapped,True);entry['after_in_memory_polygon_cache_clean']=check(shape,True)
 steps.append(entry);print(name,entry,flush=True)
def meshstats(mesh):
 _,counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True);return {'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'signed_volume_m3':float(mesh.volume),'signed_volume_mm3':float(mesh.volume*1e9),'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),'zero_area_faces':int((mesh.area_faces<1e-20).sum()),'finite_vertices':bool(np.isfinite(mesh.vertices).all()),'euler_number':int(mesh.euler_number)}
meshes=[]
for name in ['shoulder_connector_printed_0.obj','shoulder_connector_printed_1.obj']:
 mesh=trimesh.load_mesh(FOLDER/name,process=False);entry={'file':name,'raw':meshstats(mesh),'in_memory_vertex_weld_checks':[]}
 for digits in [12,10,9,8,7]:
  copy=mesh.copy();copy.merge_vertices(digits_vertex=digits);entry['in_memory_vertex_weld_checks'].append({'digits_meters':digits,**meshstats(copy)})
 meshes.append(entry);print('MESH',entry,flush=True)
after=hashes();unchanged=before==after
report={'mode':'READONLY files; in-memorySTEPimport,tessellation,cachecleanup,meshweldingonly. No upper/sourcefilewritesorfieldrebuilds.','input_snapshot_before':before,'input_snapshot_after':after,'all_input_bytes_unchanged':unchanged,'step_checks':steps,'mesh_checks':meshes,'reported_solids_valid':meta['solids_valid'],'diagnosis':'Pending independent result interpretation','recommendation':'Pending independent result interpretation'}
OUT.write_text(json.dumps(report,indent=2)+'\n');print('INPUTSUNCHANGED',unchanged,flush=True)
