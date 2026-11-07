"""Reusable exact-case swept corridor cut on an explicit fork-fit STEP candidate."""
from pathlib import Path
import argparse,hashlib,json,time
import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec,gp_Trsf
from scripts.build_wowrobo_components import load_assembly,leaves
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'combined-clearance-v4';OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'outputs/insertion-fix-20261107/connector/clearance-review-v4/shoulder_connector_clearance_review.step');p.add_argument('--clearance-mm',type=float,default=.05);a=p.parse_args();source=a.source;source_hash=hashlib.sha256(source.read_bytes()).hexdigest();receiver=cq.importers.importStep(str(source)).val();solids=sorted(receiver.Solids(),key=lambda s:-s.BoundingBox().zlen)
trans=json.load(open(OUT.parent/'case-transform.json'));T=np.array(trans['matrix_source_case_to_receiver_mm']);trsf=gp_Trsf();trsf.SetValues(*T[:3,:].ravel());location=cq.Location(trsf);up=np.array(trans['up_direction_source']);doc,tool,labels=load_assembly(ROOT/'assets_local/wowrobo/source/STEP/SO101/SO101 Assembly.step');case=[];records=[]
for name,shape in leaves(tool,labels['ST3215 Servo v2:2']):
 if '金属舵盘' in name:continue
 original=shape
 if not shape.isValid():shape=cq.importers.importStep(str(OUT.parent/'parent-rebuilt-case-cutter.step')).val();print('usingvalidshellreconstruction',name,shape.isValid(),flush=True)
 if not shape.isValid():raise RuntimeError('Invalidcasecuttool')
 shape=shape.moved(location);case.append((name,shape));records.append({'name':name,'original_valid':original.isValid(),'tool_valid':shape.isValid(),'tool_volume_mm3':shape.Volume()})
base_tools=[];rejected=[]
for name,shape in case:
 if shape.distance(receiver)<=a.clearance_mm+.001:base_tools.append((name+'-seat',shape))
 for index,face in []:
  if face.BoundingBox().zmin>103.70:continue
  if np.dot(np.array(face.normalAt().toTuple()),up)<=1e-5:continue
  try:
   prism=cq.Shape.cast(BRepPrimAPI_MakePrism(face.wrapped,gp_Vec(*(up*25.)),False,True).Shape())
   if not prism.isValid() or abs(prism.Volume())<1e-8:raise ValueError('invalid/zero-volume')
   if prism.distance(receiver)<=a.clearance_mm+.001:base_tools.append((name+f'-face{index}',prism))
  except Exception as e:rejected.append({'name':name,'face':index,'error':str(e)})
print('relevantexactcase-sweepprimitives',len(base_tools),'rejected',len(rejected),flush=True)
(OUT/'tool-checkpoint.json').write_text(json.dumps({'leaf_records':records,'base_primitives':[n for n,_ in base_tools],'rejected_prisms':rejected},indent=2)+'\n')
# Six directional gauge offsets create an explicit <=sqrt(3)*50um review
# machining allowance. Every tool remains an actual source-shape translation.
shifts=[np.zeros(3)]+[sign*a.clearance_mm*np.eye(3)[axis] for axis in range(3) for sign in [-1,1]];tools=[];tool_records=[]
for name,shape in base_tools:
 for shift in shifts:
  shifted=shape.translate(cq.Vector(*shift))
  if shifted.distance(receiver)>1e-5:continue
  tools.append(shifted);bb=shifted.BoundingBox();tool_records.append({'name':name,'shift_mm':shift.tolist(),'min_z_mm':bb.zmin,'max_z_mm':bb.zmax})
print('intersectingoffsettools',len(tools),flush=True)
results=[]
for index,solid in enumerate(solids):
 revised=solid
 for i,tool in enumerate(tools):
  if tool.distance(revised)>1e-6:continue
  candidate=revised.cut(tool,tol=1e-5)
  if not candidate.isValid() or len(candidate.Solids())!=1:
   before_v=candidate.Volume();fixed=candidate.fix();diagnostic={'constituent':index,'tool':i,'valid':candidate.isValid(),'solid_volumes_mm3':[q.Volume() for q in candidate.Solids()],'fixed_valid':fixed.isValid(),'fixed_solid_volumes_mm3':[q.Volume() for q in fixed.Solids()],'fix_volume_delta_mm3':fixed.Volume()-before_v};print('cutrepair',diagnostic,flush=True);(OUT/f'debug-cut-{index}-{i}.json').write_text(json.dumps(diagnostic,indent=2)+'\n');revised.exportStep(str(OUT/f'debug-last-valid-{index}.step'));tool.exportStep(str(OUT/f'debug-tool-{index}-{i}.step'));candidate.exportStep(str(OUT/f'debug-invalid-{index}-{i}.step'))
   if fixed.isValid() and len(fixed.Solids())==1 and abs(fixed.Volume()-before_v)<1e-6:candidate=fixed
   else:raise RuntimeError(f'Invalidcut constituent{index} tool{i}')
  revised=candidate
  if i%10==0:
   print('cutprogress',index,i,'removed',solid.Volume()-revised.Volume(),flush=True);revised.exportStep(str(OUT/f'debug-last-valid-{index}.step'))
 revised.exportStep(str(OUT/f'shoulder_connector_printed_{index}.step'));vertices,faces=revised.tessellate(.04,.08);mesh=trimesh.Trimesh(np.array([v.toTuple() for v in vertices])/1000,faces,process=False);mesh.export(OUT/f'shoulder_connector_printed_{index}.obj');results.append(revised)
 print('exportedconstituent',index,'removedmm3',solid.Volume()-revised.Volume(),flush=True)
cq.Compound.makeCompound(results).exportStep(str(OUT/'shoulder_connector_combined_clearance.step'))
report={'profile':'combined-exact-case-seat-clearance-review-v4','source_step':str(source),'source_sha256':source_hash,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,'original_case_geometry_unchanged':True,'case_cutting_tools':records,'swept_case_primitive_count':len(base_tools),'offset_tool_count':len(tools),'tool_records':tool_records,'rejected_prisms':rejected,'pin_headstop75p78mm_protection_min_tool_gap_mm':min(x['min_z_mm'] for x in tool_records)-75.78,'prior_horn_clearance_source_preserved':True,'combined_step':str(OUT/'shoulder_connector_combined_clearance.step'),'derived_meshes':{f'shoulder_connector_printed_{i}':str(OUT/f'shoulder_connector_printed_{i}.obj') for i in range(2)},'source_volume_mm3':sum(s.Volume() for s in solids),'candidate_volume_mm3':sum(s.Volume() for s in results),'removed_volume_mm3':sum(s.Volume() for s in solids)-sum(s.Volume() for s in results),'solids_valid':all(s.isValid() for s in results),'scope':'Actualvalidsourcecaseatmechanicalseatingpose,explicit50umdirectionalgaugeallowance,onlyreceivercut;fullaxialpathrequiresdirectsource/nativeverification. NEEDSdenseoriginalmaterial-depth<=0.5mmproof/sourcecontainedsubset/protectedborechecks/nativeforce+gravityvalidation beforeclaimingworkingfit.'};(OUT/'combined-candidate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
