from pathlib import Path
import json,hashlib,cadquery as cq,numpy as np,trimesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
ROOT=Path.cwd();OUT=ROOT/'outputs/insertion-fix-20261107/final/local-cad';SOURCE=OUT/'original-upper.step';shape=cq.importers.importStep(str(SOURCE)).val().Solids()[0];oldvol=shape.Volume();data=np.load(OUT/'case-sweep-inside-source.npz');points=data['points'];distances=data['distances'];normals=data['normals'];px=np.array([.9999619234843208,.008726155048712941,.00007615512646753803]);center=np.array([1.96,74,123]);tools=[];mod=[]
for idx,face in enumerate(shape.Faces()):
 if BRepAdaptor_Surface(face.wrapped).GetType()!=GeomAbs_Plane:continue
 n=np.array(face.normalAt().toTuple());fc=np.array(face.Center().toTuple());bb=face.BoundingBox()
 if not(bb.ymax>58 and bb.ymin<92 and bb.zmax>112 and bb.zmin<135 and abs(n@px)>.25 and (center-fc)@n>0):continue
 projected=-((points-fc)@n);mask=(projected>1e-5)&(projected<=.5)&(abs(projected-distances)<.03)&((normals@n)>.995)&(points[:,0]>=bb.xmin-.51)&(points[:,0]<=bb.xmax+.51)&(points[:,1]>=bb.ymin-.51)&(points[:,1]<=bb.ymax+.51)&(points[:,2]>=bb.zmin-.51)&(points[:,2]<=bb.zmax+.51)
 if not np.any(mask):continue
 ids=np.flatnonzero(mask);ids=ids[np.argsort(-projected[ids])];prism=cq.Solid.extrudeLinear(face.translate(cq.Vector(*(n*.001))).outerWire(),face.translate(cq.Vector(*(n*.001))).innerWires(),cq.Vector(*(-n*.501)));witness=None;maximum=0.;checked=0;seen=set()
 for j in ids:
  key=tuple(np.round(points[j],3))
  if key in seen:continue
  seen.add(key);checked+=1
  if prism.isInside(cq.Vector(*points[j]),1e-7) and shape.isInside(cq.Vector(*points[j]),1e-7):maximum=float(projected[j]);witness=points[j].tolist();break
  if checked>5000:raise RuntimeError('Unexpected footprint rejection explosion')
 if not maximum:continue
 relief=maximum+.05
 if relief>.5:raise RuntimeError('Over local0.5mmreliefbudget')
 shifted=face.translate(cq.Vector(*(n*.001)));tool=cq.Solid.extrudeLinear(shifted.outerWire(),shifted.innerWires(),cq.Vector(*(-n*(relief+.001))));tools.append(tool);row={'source_face_index':idx,'area_mm2':face.Area(),'normal':n.tolist(),'center_mm':fc.tolist(),'bbox_mm':[[bb.xmin,bb.ymin,bb.zmin],[bb.xmax,bb.ymax,bb.zmax]],'source_case_sweep_intrusion_mm':maximum,'clearance_mm':.05,'relief_mm':relief,'witness_mm':witness,'matched_case_points':int(mask.sum()),'witnesses_checked':checked};mod.append(row);print(row,flush=True)
revised=shape
for i,tool in enumerate(tools):
 revised=revised.cut(tool)
 if not revised.isValid():raise RuntimeError(f'Invalid afterfacecut{i}')
if len(revised.Solids())!=1:raise RuntimeError('Derivedshape must remain one solid')
revised.exportStep(str(OUT/'upper_arm_module_printed_0.step'));v,f=revised.tessellate(.02,.05);mesh=trimesh.Trimesh(np.array([x.toTuple() for x in v])/1000,f,process=True);mesh.export(OUT/'upper_arm_module_printed_0.obj');removed=shape.cut(revised);bbox0=shape.BoundingBox();bbox1=revised.BoundingBox();r={'profile':'final-local-inner-pocket-clearance-review-v1','source_step':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_unchanged':True,'source_volume_mm3':oldvol,'new_volume_mm3':revised.Volume(),'removed_mm3':oldvol-revised.Volume(),'new_shape_valid':revised.isValid(),'solid_count':len(revised.Solids()),'topology_faces_before':len(shape.Faces()),'topology_faces_after':len(revised.Faces()),'outer_bounding_box_unchanged':all(abs(getattr(bbox0,k)-getattr(bbox1,k))<1e-7 for k in ['xmin','xmax','ymin','ymax','zmin','zmax']),'cuts':mod,'mesh':str(OUT/'upper_arm_module_printed_0.obj'),'mesh_sha256':hashlib.sha256((OUT/'upper_arm_module_printed_0.obj').read_bytes()).hexdigest(),'qualification':'Derivedmanufacturingreviewcandidate. Onlyactualcase-sweep-facing innerplanar/chamferface footprints recessed by measuredintrusion+50um. OriginalvendorSTEP/OBJ unchanged. Noouterwall/floor resizing, nocase shrink. Requiresnativeoldlawfullstroke/release/closedwall/sourcecontainment checks.'};(OUT/'local-relief.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='cuts'},indent=2),flush=True)
