import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
out=Path(__file__).resolve().parent
r=WowRoboRandomizer(config='configs/wowrobo_randomization_repaired.json');m=r.model;a=NativeBoundaryAudit(m)
rows=json.loads((out/'lower-candidates.json').read_text());idx=[m.joint('left_'+s).qposadr[0] for s in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']]
for n in [1,5,9,11]:
 row=next(x for x in rows if x['i']==n);d=r.restore('25_right_support');d.qpos[idx]=row['q_left'];mujoco.mj_forward(m,d);v=a.audit(d,target_target=False)
 site=m.site('left_gripperframe').id;frame=d.site_xmat[site].reshape(3,3)
 print(n,'pitch',np.degrees(np.arcsin(frame[2,0])),'why',v['candidate_rejection_reasons'],'table',v['table_minimum'],'handhand',[(p['a']['mesh_name'],p['b']['mesh_name'],p['penetration_m']) for p in v['hand_hand'] if p['rejected']],flush=True)
 (out/f'lower-{n}-audit.json').write_text(json.dumps(v,indent=2))
