"""Contact-law-independent source mesh checks at actual candidate trajectory extrema."""
from pathlib import Path
import json,numpy as np,trimesh,igl
ROOT=Path(__file__).resolve().parents[3];out=Path(__file__).parent
source=ROOT/'outputs/linear-insertion-aligned-20261107/connector/audit_alignment.py';ctx={'__file__':str(source)};code=source.read_text();exec(code[:code.index('qa=int')],ctx);alignment=json.loads((source.parent/'alignment.json').read_text());axis=np.array(alignment['axis_world']);report=[]
for file in sorted(out.glob('*.json')):
 r=json.loads(file.read_text())
 if 'final_drive_axial_mm' not in r:continue
 samples=sorted(set([r['all_substep_minimum_axial_mm'],r['all_substep_maximum_axial_mm'],r['final_drive_axial_mm'],r['final_relaxed_axial_mm']]))
 rows=[]
 for travel in samples:
  q=np.array(alignment['start_qpos']);q[20:23]+=axis*(travel/1000);depth,pairs=ctx['audit'](q);pair=max(pairs,key=lambda x:x['max_overlap_m']) if pairs else None;exact=0.
  if pair and depth>0:
   mesh=ctx['meshes'][pair['target']];point=np.array([pair['worst_point_target_source_m']]);distance=trimesh.proximity.closest_point(mesh,point)[1][0];w=igl.winding_number(np.ascontiguousarray(mesh.vertices),np.ascontiguousarray(mesh.faces,dtype=np.int64),point)[0];exact=float(distance if w>.5 else 0)
  rows.append({'axial_mm':travel,'max_sampled_source_sdf_overlap_m':depth,'exact_triangle_witness_overlap_m':exact,'worst_pair':pair})
 report.append({'case':r['case'],'contacts_enabled':r['contacts_enabled'],'final_drive_axial_mm':r['final_drive_axial_mm'],'remaining_to_reference_mm':r['reference_axial_mm']-r['final_drive_axial_mm'],'selected_extrema_source_checks':rows,'selected_source_overlap_guard_pass':all(x['max_sampled_source_sdf_overlap_m']<=.0005 and x['exact_triangle_witness_overlap_m']<=.0005 for x in rows),'qualified_insertion':False})
result={'cases':report,'method':'Original source vertices + all triangle centroids, reciprocal SDF in original CAD frames, exact triangle/winding witness at actual min/max/drive-end/relax-end positions; current fully fixed orientation/shape. Earlier131pose wholepath audit remains separate. Contact-law-independent finite checks, not continuous CCD.','all_contact_enabled_extrema_source_guards_pass':all(x['selected_source_overlap_guard_pass'] for x in report if x['contacts_enabled']),'any_contact_enabled_reference_reached':any(x['remaining_to_reference_mm']<=.05 for x in report if x['contacts_enabled'])};(out/'candidate-source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));print([(x['case'],max(y['exact_triangle_witness_overlap_m'] for y in x['selected_extrema_source_checks'])) for x in report])
