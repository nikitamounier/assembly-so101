from pathlib import Path
import json
from scripts.build_wowrobo_contact_grids import build_one
ROOT=Path.cwd();out=ROOT/'outputs/insertion-fix-20261107/final/local-cad-v3';base=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text());row=dict(base['meshes']['upper_arm_module_printed_0']);row.update(source_file=str(out/'upper_arm_module_printed_0.obj'),grid_file=str(out/'upper-clearance-coarse-grid.bin'),spacing=.0004);r=build_one(row);(out/'upper-clearance-coarse-grid.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['name','shape','elapsed_s','grid_sha256','source_sha256','spacing','sampled_error_max_m','source_watertight']},indent=2),flush=True)
