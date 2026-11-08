import json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);old=mujoco.MjModel.from_xml_path(p['model_path']);r=ET.parse(p['model_path']);root=r.getroot()
for servo in [2,3]:
 b=root.find(f".//body[@name='target_servo_{servo}_pivot']");j=b.find('joint');axis=np.fromstring(j.get('axis'),sep=' ');axis/=np.linalg.norm(axis);pos=np.fromstring(b.get('pos','0 0 0'),sep=' ');b.set('pos',' '.join(map(str,pos-.0004*axis)))
xml=OUT/'registration-centered-candidate.xml';r.write(xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(xml));do=mujoco.MjData(old);d=mujoco.MjData(m);res={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','25_right_support','29_final_far_side','30_final_half','31_final_near']:
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/(stage+'_train.npz'))as z:mujoco.mj_setState(old,do,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.qpos[:]=do.qpos
 mujoco.mj_kinematics(old,do)
 # Restore exact world motor cases after centering by shifting their free roots.
 for servo,task in [(2,'upper_arm_module'),(3,'forearm_module')]:
  axis=do.xaxis[old.joint(f'target_servo_{servo}_shaft').id];a=m.joint(task+'_free').qposadr[0];d.qpos[a:a+3]+=.0004*axis
 mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);rows=[]
 for c in d.contact:
  names=[m.geom(int(g)).name for g in c.geom]
  if any(n and(n.startswith('upper_arm_module_printed')or n.startswith('forearm_module_printed'))for n in names):rows.append({'pair':names,'distance_m':float(c.dist)})
 own=[row for row in rows if any(n.startswith('servo_2_case')or n.startswith('servo_3_case')for n in row['pair'])]
 poserr={f'servo{s}':float(np.linalg.norm(d.xpos[m.body(f'target_servo_{s}_pivot').id]-do.xpos[old.body(f'target_servo_{s}_pivot').id]))for s in [2,3]}
 res[stage]={'case_position_errors_m':poserr,'contacts':sorted(rows,key=lambda x:x['distance_m']),'own_contacts':own};print(stage,poserr,'worst',res[stage]['contacts'][:3],'own',own[:8],flush=True)
 np.savez_compressed(OUT/(stage+'-centered-initializer.npz'),qpos=d.qpos.copy(),source_qpos=do.qpos.copy())
(OUT/'centered-registration-contacts.json').write_text(json.dumps(res,indent=2)+'\n')
