import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
res={}
for stage,servo,task in [('21_upper_far_side',2,'upper_arm_module'),('29_final_far_side',3,'forearm_module')]:
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/(stage+'_train.npz'))as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
 q0=d.qpos.copy();h=m.joint(f'target_servo_{servo}_shaft');rows=[]
 for delta in [-.4,-.2,0,.1,.2,.4,.6,.8]:
  d.qpos[:]=q0;d.qpos[h.qposadr[0]]+=delta;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);axis=d.xaxis[h.id];anchor=d.xanchor[h.id];pts=[]
  for c in d.contact:
   names=[m.geom(int(g)).name for g in c.geom]
   if f'servo_{servo}_case__contact'not in names or not any(n.startswith(task+'_printed')for n in names):continue
   v=c.pos-anchor;ax=float(v@axis);rad=float(np.linalg.norm(v-ax*axis));pts.append({'dist':float(c.dist),'ax':ax,'rad':rad,'case_local':((c.pos-d.xpos[m.body(f'target_servo_{servo}_pivot').id])@d.xmat[m.body(f'target_servo_{servo}_pivot').id].reshape(3,3)).tolist(),'normal':c.frame[:3].tolist()})
  row={'delta_rad':delta,'theta_rad':float(d.qpos[h.qposadr[0]]),'count':len(pts),'contacts':pts};rows.append(row);print(stage,delta,len(pts),[(round(x['dist']*1000,3),round(x['rad']*1000,2),round(x['ax']*1000,2))for x in pts[:20]],flush=True)
 res[stage]=rows
(OUT/'own-ranges.json').write_text(json.dumps(res,indent=2)+'\n')
