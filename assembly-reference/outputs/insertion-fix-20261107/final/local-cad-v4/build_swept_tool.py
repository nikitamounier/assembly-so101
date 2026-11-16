from pathlib import Path
import sys,json,numpy as np,cadquery as cq
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
from OCP.BRepOffset import BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Arc
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));from scripts.build_wowrobo_components import load_assembly,leaves,ASSEMBLY
OUT=ROOT/'outputs/insertion-fix-20261107/final/local-cad-v4';D,t,l=load_assembly(ROOT/'assets_local/wowrobo/source'/ASSEMBLY);source=next(s for name,s in leaves(t,l['ST3215 Servo v2:3']) if 'SG-ZIJI_15' in name);solid=source.Solids()[0];off=BRepOffsetAPI_MakeOffsetShape();off.PerformByJoin(solid.wrapped,.05,1e-6,BRepOffset_Skin,False,False,GeomAbs_Arc,True)
if not off.IsDone():raise RuntimeError('Offset50um failed')
expanded=cq.Shape.cast(off.Shape());print('offsetvalid',expanded.isValid(),'solids',len(expanded.Solids()),'faces',len(expanded.Faces()),flush=True)
if not expanded.isValid() or len(expanded.Solids())!=1:raise RuntimeError('Requirevalidsingleoffsetsolid')
expanded.exportStep(str(OUT/'sg15-clearance50um.step'))
