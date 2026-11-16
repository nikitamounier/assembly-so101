import sys,json
from pathlib import Path
import cadquery as cq
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));from scripts.build_wowrobo_components import load_assembly,leaves,ASSEMBLY
out=ROOT/'outputs/insertion-fix-20261107/final/local-cad-v3';doc,tool,labels=load_assembly(ROOT/'assets_local/wowrobo/source'/ASSEMBLY);parts=leaves(tool,labels['ST3215 Servo v2:3']);solids=[];info=[]
for name,shape in parts:
 if '金属舵盘' in name:continue
 for i,s in enumerate(shape.Solids()):
  valid=s.isValid();fixed=s
  if not valid:
   shells=s.Shells();fixed=cq.Solid.makeSolid(shells[0])
   for shell in shells[1:]:fixed=fixed.cut(cq.Solid.makeSolid(shell))
   if not fixed.isValid():raise RuntimeError('Failed sourceboundary-preserving repair')
  solids.append((name,fixed));info.append({'name':name,'original_valid':valid,'repaired_valid':fixed.isValid(),'shells':len(s.Shells()),'volume_mm3':fixed.Volume()})
data=json.loads((out/'stall-witnesses.json').read_text());results=[]
for row in data['contacts']:
 if 'receiver_surface_case_source_mm' not in row:continue
 p=cq.Vector(*row['receiver_surface_case_source_mm']);inside=[name for name,s in solids if s.isInside(p,1e-6)];r={'case_point_mm':row['receiver_surface_case_source_mm'],'case_sdf_mm':row['receiver_surface_native_case_SDF_mm'],'triangle_unsigned_mm':row['receiver_surface_case_unsigned_mm'],'triangle_winding':row['receiver_surface_case_winding'],'inside_valid_source_case_material':bool(inside),'material_components':inside};results.append(r);print(r,flush=True)
(out/'valid-case-brep-witnesses.json').write_text(json.dumps({'source_case':3,'boundary_reconstruction':'Only invalid source leaf: makeSolid outer shell, sequentially subtract seven original closed voidshells; allothercasesolids unchanged','components':info,'contacts':results},indent=2)+'\n');cq.exporters.export(cq.Compound.makeCompound([s for _,s in solids]),str(out/'valid-source-case3.step'))
