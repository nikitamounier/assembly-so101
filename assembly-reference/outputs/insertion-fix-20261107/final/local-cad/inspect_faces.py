import sys,json
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));import numpy as np,cadquery as cq
from scripts.build_wowrobo_components import load_assembly,ASSEMBLY
_,tool,labels=load_assembly(ROOT/'assets_local/wowrobo/source'/ASSEMBLY);shape=cq.Shape.cast(tool.GetShape_s(labels['Upper_arm_SO101 v9:1']));px=np.array([.9999619234843208,.008726155048712941,.00007615512646753803]);rows=[]
for i,f in enumerate(shape.Faces()):
 c=np.array(f.Center().toTuple());n=np.array(f.normalAt().toTuple());bb=f.BoundingBox()
 if bb.ymax>58 and bb.ymin<90 and bb.zmax>112 and bb.zmin<134 and abs(n@px)>.999:
  rows.append({'face':i,'center':c.tolist(),'normal':n.tolist(),'area_mm2':f.Area(),'normaldotpx':float(n@px),'bbox':[[bb.xmin,bb.ymin,bb.zmin],[bb.xmax,bb.ymax,bb.zmax]]})
print(json.dumps({'valid':shape.isValid(),'solids':len(shape.Solids()),'rows':rows},indent=2));(ROOT/'outputs/insertion-fix-20261107/final/local-cad/faces.json').write_text(json.dumps(rows,indent=2)+'\n');shape.exportStep(str(ROOT/'outputs/insertion-fix-20261107/final/local-cad/original-upper.step'))
