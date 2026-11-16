from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
out=Path('outputs/insertion-fix-20261107/final');m=mujoco.MjModel.from_xml_path('outputs/linear-insertion-20261107/source-snapshot/scene.xml');d=mujoco.MjData(m);audit=NativeBoundaryAudit(m,max_samples=4096);active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module'];qa=int(m.joint('forearm_module_free').qposadr[0]);rows=[]
for name,apath in [('candidate',out/'alignment.json'),('negative-wall',Path('outputs/legacy-fit-port-20261107/final/negative-wall-alignment.json'))]:
 a=json.loads(apath.read_text());axis=np.array(a['axis_world']);qstart=np.array(a['start_qpos']);tr=np.load(out/(name+'-trajectory.npz'));samples=[('maximum',float(np.max(tr['qpos']))),('drive',json.loads((out/(name+'-recording.json')).read_text())['drive_mm']/1000),('relaxed',float(tr['qpos'][-1,0]))]
 for phase,travel in samples:
  d.qpos[:]=qstart;d.qpos[qa:qa+3]+=travel*axis;mujoco.mj_forward(m,d);bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(a,b) for a in active for b in other],bounds);peak=max([x['penetration_m'] for x in pairs],default=0);row={'case':name,'phase':phase,'axial_mm':travel*1000,'max_source_penetration_mm':peak*1000,'below_0p5mm':peak<=.0005,'pairs':[{'a':x['a']['geom_name'],'b':x['b']['geom_name'],'mm':x['penetration_m']*1000} for x in pairs if x['penetration_m']>.0001]};rows.append(row);print(row,flush=True)
(out/'recorded-source-audit.json').write_text(json.dumps({'max_samples_per_direction':4096,'rows':rows,'all_under_0p5mm':all(x['below_0p5mm'] for x in rows)},indent=2)+'\n')
