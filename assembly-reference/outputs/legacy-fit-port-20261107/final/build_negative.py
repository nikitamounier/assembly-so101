from pathlib import Path
import copy,json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
p=Path('outputs/legacy-fit-port-20261107/final');src=Path('outputs/linear-insertion-aligned-20261107/final/alignment.json');a=json.loads(src.read_text());m=mujoco.MjModel.from_xml_path('outputs/linear-insertion-20261107/source-snapshot/scene.xml');d=mujoco.MjData(m);d.qpos[:]=a['start_qpos'];mujoco.mj_forward(m,d);RR=d.body('upper_arm_module').xmat.reshape(3,3);normal=RR@np.array(a['feature_frame']['receiver_local_pocket_wall_normal']);offset=.001*normal;qa=int(m.joint('forearm_module_free').qposadr[0]);b=copy.deepcopy(a)
for k in ['start_qpos','initial_qpos','qpos_initial','reference_qpos','end_qpos','qpos_end']:
 q=np.array(b[k]);q[qa:qa+3]+=offset;b[k]=q.tolist()
b['negative_wall_control']={'offset_world_m':offset.tolist(),'offset_m':.001,'direction':'into actual positive receiverlocalX pocketwall','expected':'Blocked before referencedepth or failpenetrationguard; NEVER score negativewalltarget asvalidseat'};b['negative_wall_control']['inherited_main_feature_frame_not_valid_for_offset']=True
(p/'negative-wall-alignment.json').write_text(json.dumps(b,indent=2)+'\n');audit=NativeBoundaryAudit(m,max_samples=4096);active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module'];rows=[]
for name in ['start_qpos','reference_qpos']:
 d.qpos[:]=b[name];mujoco.mj_forward(m,d);bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(x,y) for x in active for y in other],bounds);row={'state':name,'max_sampled_penetration_m':max([x['penetration_m'] for x in pairs],default=0),'pairs':[{'a':x['a']['geom_name'],'b':x['b']['geom_name'],'penetration_m':x['penetration_m']} for x in pairs if x['penetration_m']>.0005]};rows.append(row);print(row)
(p/'negative-wall-source-proof.json').write_text(json.dumps({'normal_world':normal.tolist(),'offset_world_m':offset.tolist(),'max_samples_per_direction':4096,'states':rows},indent=2)+'\n')
