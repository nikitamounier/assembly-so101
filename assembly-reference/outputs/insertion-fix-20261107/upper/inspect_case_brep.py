from pathlib import Path
from scripts.build_wowrobo_components import load_assembly,leaves
import cadquery as cq,json
ROOT=Path(__file__).resolve().parents[3];doc,tool,labels=load_assembly(ROOT/'assets_local/wowrobo/source/STEP/SO101/SO101 Assembly.step');items=leaves(tool,labels['ST3215 Servo v2:2']);rows=[]
for n,s in items:
 if '金属舵盘' in n:continue
 fixed=s.fix();rows.append({'name':n,'valid':s.isValid(),'solids':len(s.Solids()),'volume':s.Volume(),'fixed_valid':fixed.isValid(),'fixed_solids':len(fixed.Solids()),'volume_change':fixed.Volume()-s.Volume()});print(rows[-1],flush=True)
 if not s.isValid():s.exportStep(str(ROOT/'outputs/insertion-fix-20261107/upper/invalid-case-source.step'));fixed.exportStep(str(ROOT/'outputs/insertion-fix-20261107/upper/fixed-case-cutter.step'))
(ROOT/'outputs/insertion-fix-20261107/upper/case-brep-validity.json').write_text(json.dumps(rows,indent=2)+'\n')
