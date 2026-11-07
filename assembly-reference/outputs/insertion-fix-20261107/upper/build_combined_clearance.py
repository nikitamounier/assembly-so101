"""Measured local case-pocket relief added to the isolated fork-fit candidate."""
from pathlib import Path
import hashlib,json
import cadquery as cq
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'combined-clearance-v2';OUT.mkdir(exist_ok=True);SOURCE=ROOT/'outputs/insertion-fix-20261107/connector/clearance-review-v1/shoulder_connector_clearance_review.step';original_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest();full=cq.importers.importStep(str(SOURCE)).val();solids=sorted(full.Solids(),key=lambda s:-s.BoundingBox().zlen);sweep=np.load(OUT.parent/'case-sweep-source.npz');points=sweep['points_reference']*1000;up=sweep['up_direction'];depths=sweep['depths']*1000;modifications=[];outputs=[]
for index,solid in enumerate(solids):
 targets=[(-15.59943246979,1),(16.200567530205,-1)] if index==0 else [(-15.699432469795,1),(16.100567530205,-1)]
 tools=[]
 for plane,sign in targets:
  selected=[f for f in solid.Faces() if np.dot(np.array(f.normalAt().toTuple()),[sign,0,0])>.999999 and abs(f.Center().x-plane)<.001 and f.Area()>50]
  if len(selected)!=1:raise ValueError(f'Expectedoneinnerface {index} {plane}: {len(selected)}')
  face=selected[0];n=np.array([sign,0,0.]);bb=face.BoundingBox();test=face.translate(cq.Vector(*(n*.001)));prism=cq.Solid.extrudeLinear(test.outerWire(),test.innerWires(),cq.Vector(*(-n*3.001)))
  candidates=[]
  for depth in depths:
   p=points+depth*up;intrusion=(plane-p[:,0])*sign;mask=(intrusion>0)&(intrusion<=3.000001)&(p[:,1]>=bb.ymin-1e-6)&(p[:,1]<=bb.ymax+1e-6)&(p[:,2]>=bb.zmin-1e-6)&(p[:,2]<=bb.zmax+1e-6)
   if np.any(mask):candidates.append(np.column_stack([intrusion[mask],p[mask]]))
  maximum=0.;witness=None;checked=0
  if candidates:
   c=np.vstack(candidates);c=c[np.argsort(-c[:,0])];seen=set()
   for row in c:
    key=tuple(np.round(row[1:],3))
    if key in seen:continue
    seen.add(key);checked+=1
    if prism.isInside(cq.Vector(*row[1:]),1e-7) and solid.isInside(cq.Vector(*row[1:]),1e-7):maximum=float(row[0]);witness=row[1:].tolist();break
    if checked>=5000:raise RuntimeError('Too many footprint-rejected source witnesses')
  relief=maximum+.05 if maximum>0 else 0.
  if relief>.5:raise RuntimeError(f'Requiredcut{relief}exceeds0.5mmmaterialbudget')
  if relief:
   shifted=face.translate(cq.Vector(*(n*.001)));tool=cq.Solid.extrudeLinear(shifted.outerWire(),shifted.innerWires(),cq.Vector(*(-n*(relief+.001))));tools.append(tool)
  modifications.append({'constituent':f'shoulder_connector_printed_{index}','source_plane_x_mm':plane,'outward_normal_source':n.tolist(),'source_face_area_mm2':face.Area(),'bounds_mm':[[bb.xmin,bb.ymin,bb.zmin],[bb.xmax,bb.ymax,bb.zmax]],'maximum_measured_case_sweep_intrusion_mm':maximum,'required_review_clearance_mm':.05,'inward_relief_mm':relief,'source_witness_mm':witness,'witnesses_classified':checked})
 revised=solid
 for tool in tools:revised=revised.cut(tool)
 if not revised.isValid() or len(revised.Solids())!=1:raise RuntimeError('Derivedconstituentmustremainonevalidsolid')
 outputs.append(revised);revised.exportStep(str(OUT/f'shoulder_connector_printed_{index}.step'));vertices,faces=revised.tessellate(.04,.08);mesh=trimesh.Trimesh(np.array([v.toTuple() for v in vertices])/1000,faces,process=False);mesh.export(OUT/f'shoulder_connector_printed_{index}.obj')
full=cq.Compound.makeCompound(outputs);full.exportStep(str(OUT/'shoulder_connector_combined_clearance.step'));result={'profile':'combined-fork-and-case-clearance-review-v2','source_step':str(SOURCE),'source_sha256':original_hash,'source_unchanged':hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash,'preserves_prior_fork_0p475mm_reliefs':True,'case_wall_modifications':modifications,'all_solids_valid':all(s.isValid() for s in outputs),'solids':len(full.Solids()),'removed_from_prior_candidate_mm3':sum(s.Volume() for s in solids)-sum(s.Volume() for s in outputs),'combined_step':str(OUT/'shoulder_connector_combined_clearance.step'),'derived_meshes':{f'shoulder_connector_printed_{i}':str(OUT/f'shoulder_connector_printed_{i}.obj') for i in range(2)},'qualification':'Explicitderivedmanufacturingreviewcandidate. OriginalsourceCADpreserved. Eachlocalcasewallrecess=minmeasuredsourcecasesweepintrusion+50um;cap0.5mm. Priorforkcutregionsatz29/65mmdisjointfromcasewallregions76..103mm. Nativefullstroke/release/closedwallchecksrequiredbeforeclaimingworkingphysics.'};(OUT/'combined-candidate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
