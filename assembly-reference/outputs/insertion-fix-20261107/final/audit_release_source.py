from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
out=Path('outputs/insertion-fix-20261107/final');load_backend('outputs/linear-insertion-updated-20261107/grids/manifest.json');rows=[]
for mode in ['free_root_locked_shape','free_root_internal_free']:
 m=mujoco.MjModel.from_xml_path(str(out/(mode+'.xml')));d=mujoco.MjData(m);d.qpos[:]=np.load(out/(mode+'-state.npz'))['qpos'];mujoco.mj_forward(m,d);audit=NativeBoundaryAudit(m,max_samples=4096);active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module'];bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(x,y) for x in active for y in other],bounds);peak=max([x['penetration_m'] for x in pairs],default=0);r={'release_mode':mode,'source_max_penetration_mm':peak*1000,'passed_under_0p5mm':peak<=.0005,'pairs':[{'a':x['a']['geom_name'],'b':x['b']['geom_name'],'source_penetration_mm':x['penetration_m']*1000} for x in pairs if x['penetration_m']>.0001]};rows.append(r);print(r,flush=True)
(out/'release-source-audit.json').write_text(json.dumps({'max_samples_per_direction':4096,'rows':rows},indent=2)+'\n')
