import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m);res={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','25_right_support','29_final_far_side']:
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/f'{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
 mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
 servo=2 if stage.startswith(('21','22','23','25'))else 3;task='upper_arm_module'if servo==2 else'forearm_module';j=m.joint(f'target_servo_{servo}_shaft').id;anchor=d.xanchor[j];axis=d.xaxis[j];rows=[]
 for c in d.contact:
  names=[m.geom(int(g)).name for g in c.geom]
  if f'servo_{servo}_case__contact'not in names or not any(n.startswith(task+'_printed')for n in names):continue
  delta=c.pos-anchor;z=float(delta@axis);rad=float(np.linalg.norm(delta-axis*z))
  rows.append({'pair':names,'distance_m':float(c.dist),'radial_mm':rad*1000,'axis_mm':z*1000,'point_world':c.pos.tolist()})
 res[stage]=rows
 print(stage,rows[:8],flush=True)
(OUT/'own-contact-points.json').write_text(json.dumps(res,indent=2)+'\n')
