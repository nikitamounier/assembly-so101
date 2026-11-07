from pathlib import Path
import json,numpy as np,mujoco
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
from assembly_policy.so101.wowrobo_pin_registration import measure_pin_registration,apply_pin_registration
m=mujoco.MjModel.from_xml_path('assets_local/wowrobo/fit-pin-v2/registration-review.xml');d=mujoco.MjData(m);a=NativeBoundaryAudit(m);out={}
for step in ['connector','upper','final']:
 r=json.loads(Path(f'outputs/linear-insertion-aligned-20261107/{step}/alignment.json').read_text());d.qpos[:]=r['reference_qpos'];mujoco.mj_forward(m,d)
 d.qpos[m.joint('assembly_pin_free').qposadr[0]:m.joint('assembly_pin_free').qposadr[0]+3]=[0,0,1];mujoco.mj_forward(m,d)
 result=a.audit(d);out[step]={'target_pairs':result['target_target'],'other':{k:v for k,v in r.items() if k not in ['start_qpos','reference_qpos','sweep']}}
 print(step,[(x['a'],x['b'],x['penetration_m']) for x in result['target_target']],flush=True)
Path('outputs/native-registration-review-20261107/prior-feature-alignment-audit.json').write_text(json.dumps(out,indent=2)+'\n')
