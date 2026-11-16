import json
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));d=mujoco.MjData(m);records=json.loads((OUT/'own-ranges.json').read_text());report={}
for stage,rows in records.items():
 servo=2 if stage.startswith('21')else 3;h=m.joint(f'target_servo_{servo}_shaft');body=m.body(f'target_servo_{servo}_pivot').id
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/(stage+'_train.npz'))as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
 q0=d.qpos.copy();report[stage]=[]
 for row in rows:
  d.qpos[:]=q0;d.qpos[h.qposadr[0]]=row['theta_rad'];mujoco.mj_kinematics(m,d);axis=d.xaxis[h.id];anchor=d.xanchor[h.id];pts=[]
  for p in row['contacts']:
   n=np.array(p['normal']);world=d.xpos[body]+d.xmat[body].reshape(3,3)@np.array(p['case_local']);r=world-anchor
   pp={**p,'normal_axis_dot':float(n@axis),'normal_hinge_lever_m':float(n@np.cross(axis,r))};pts.append(pp)
  report[stage].append({**row,'contacts':pts});print(stage,round(row['theta_rad'],3),[(round(x['rad']*1e3,1),round(x['ax']*1e3,1),round(x['normal_axis_dot'],3),round(x['normal_hinge_lever_m']*1e3,3))for x in pts[:25]],flush=True)
(OUT/'own-normal-levers.json').write_text(json.dumps(report,indent=2)+'\n')
