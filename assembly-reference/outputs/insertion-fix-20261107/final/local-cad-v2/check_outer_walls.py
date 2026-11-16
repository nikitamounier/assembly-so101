from pathlib import Path
import json,numpy as np,cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
out=Path('outputs/insertion-fix-20261107/final/local-cad-v2');a=cq.importers.importStep('outputs/insertion-fix-20261107/final/local-cad/original-upper.step').val().Solids()[0];b=cq.importers.importStep(str(out/'upper_arm_module_printed_0.step')).val().Solids()[0];meta=json.loads((out/'local-relief.json').read_text());px=np.array([.9999619234843208,.008726155048712941,.00007615512646753803]);walls=[]
for side in [-1,1]:
 fs=[]
 for f in a.Faces():
  n=np.array(f.normalAt().toTuple());c=np.array(f.Center().toTuple())
  if BRepAdaptor_Surface(f.wrapped).GetType()==GeomAbs_Plane and n@(side*px)>.999999 and side*c[0]>23 and 60<c[1]<80 and f.Area()>400:fs.append(f)
 if len(fs)!=1:raise RuntimeError('Expected outerwallplane')
 f=fs[0];plane=np.array(f.Center().toTuple())@px;area=f.Area();newarea=sum(g.Area() for g in b.Faces() if BRepAdaptor_Surface(g.wrapped).GetType()==GeomAbs_Plane and np.array(g.normalAt().toTuple())@(side*px)>.999999 and abs(np.array(g.Center().toTuple())@px-plane)<1e-5)
 thickness=[]
 for cut in meta['cuts']:
  n=np.array(cut['normal']);c=np.array(cut['center_mm']);
  if n@(-side*px)>.99:
   thickness.append(abs((c-cut['relief_mm']*n)@px-plane))
 walls.append({'side':side,'outer_plane_projection_mm':float(plane),'source_area_mm2':area,'derived_area_mm2':newarea,'outer_face_area_unchanged':abs(newarea-area)<1e-6,'minimum_remaining_wall_to_matching_relief_mm':min(thickness)})
added=sum(s.Volume() for s in b.cut(a).Solids());removed=sum(s.Volume() for s in a.cut(b).Solids());r={'outer_walls':walls,'minimum_remaining_wall_mm':min(x['minimum_remaining_wall_to_matching_relief_mm'] for x in walls),'added_outside_source_volume_mm3':added,'removed_volume_mm3':removed,'valid':b.isValid(),'one_solid':len(b.Solids())==1};(out/'outer-wall-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
