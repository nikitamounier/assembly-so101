import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(OUT/'centered-review-candidate.xml'));d=mujoco.MjData(m);audit=NativeBoundaryAudit(m);recipes=json.loads((OUT/'coherent-captured-recipes.json').read_text());results={}
for stage,row in recipes.items():
 d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d);r=audit.audit(d);helper_z=[]
 for g in audit.helper:
  mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];helper_z.append(float((v@d.geom_xmat[g].reshape(3,3)[2]+d.geom_xpos[g,2]).min()))
 results[stage]={'candidate_rejected':r['candidate_rejected'],'reasons':r['candidate_rejection_reasons'],'maximum_material_overlap_m':r['max_sampled_penetration_m'],'helper_floor_minimum_m':min(helper_z),'rejected_pairs':[x for key in ['helper_target','hand_hand','target_target']for x in r.get(key,[])if x.get('rejected')]};print(stage,results[stage],flush=True)
(OUT/'coherent-captured-native-audits.json').write_text(json.dumps(results,indent=2)+'\n')
