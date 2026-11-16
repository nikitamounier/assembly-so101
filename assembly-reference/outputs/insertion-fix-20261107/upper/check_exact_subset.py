from pathlib import Path
import cadquery as cq,numpy as np,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'combined-clearance-v4';source=cq.importers.importStep(str(ROOT/'assets_local/wowrobo/components-v2/shoulder_connector.step')).val();original=sorted(source.Solids(),key=lambda s:-s.BoundingBox().zlen);rows=[]
for index in range(2):
 revised=cq.importers.importStep(str(OUT/f'shoulder_connector_printed_{index}.step')).val();a=np.load(OUT/f'shoulder_connector_printed_{index}-removed-volume-samples.npz');added=a['added_points'];removed=a['points'];distance=a['source_distance'];selected=removed[np.argsort(-distance)[:100]];result={'constituent':index,'source_valid':original[index].isValid(),'candidate_reimport_valid':revised.isValid(),'added_mesh_witnesses':len(added),'added_witness_classifications':[],'deepest_removed_witness_checks':[]}
 for p in added:
  v=cq.Vector(*(p*1000));result['added_witness_classifications'].append({'point_m':p.tolist(),'original_cad_inside':original[index].isInside(v,1e-7),'candidate_cad_inside':revised.isInside(v,1e-7)})
 for p in selected:
  v=cq.Vector(*(p*1000));result['deepest_removed_witness_checks'].append({'point_m':p.tolist(),'original_cad_inside':original[index].isInside(v,1e-7),'candidate_cad_inside':revised.isInside(v,1e-7)})
 result['true_added_material_witness_count']=sum(not x['original_cad_inside'] and x['candidate_cad_inside'] for x in result['added_witness_classifications']);result['deep_removed_cad_verified_count']=sum(x['original_cad_inside'] and not x['candidate_cad_inside'] for x in result['deepest_removed_witness_checks']);rows.append(result);print(index,result['true_added_material_witness_count'],result['deep_removed_cad_verified_count'],flush=True)
(OUT/'exact-subset-witness-audit.json').write_text(json.dumps({'scope':'IndependentexactOCCpoint-in-solidcrosscheck ofALL62apparentmeshadditionwitnessesand100deepestremovedinteriorwitnesses/part. No separate-reimportcoincidentBooleanused.','rows':rows},indent=2)+'\n')
