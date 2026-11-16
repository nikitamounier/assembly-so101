import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_physical_resets import ROOT
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
out=ROOT/'outputs/equilibrium-model-20261107'
p=json.loads((out/'profile/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path'])
z=np.load(ROOT/'outputs/equilibrium-reset-20261107/natural-v1/00_loose.npz');allrows=[]
for label,state in [('start',z['states'][0]),('end',z['replay_end'])]:
 d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,mujoco.mjtState(int(z['state_spec'])));d.ctrl[:]=z['controls'][0];mujoco.mj_forward(m,d);rows=[]
 for i in range(d.ncon):
  c=d.contact[i];names=[m.geom(int(g)).name for g in c.geom]
  if not any('servo_1_'in n for n in names):continue
  force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
  rows.append(dict(names=names,bodies=[m.body(int(m.geom_bodyid[g])).name for g in c.geom],distance=float(c.dist),force=force.tolist(),position=c.pos.tolist()))
 r=dict(label=label,time=float(d.time),shaft1_qpos=float(d.joint('target_servo_1_shaft').qpos[0]),shaft1_qvel=float(d.joint('target_servo_1_shaft').qvel[0]),shaft1_bias=float(d.qfrc_bias[m.joint('target_servo_1_shaft').dofadr[0]]),shaft1_constraint=float(d.qfrc_constraint[m.joint('target_servo_1_shaft').dofadr[0]]),contacts=rows)
 allrows.append(r);print(json.dumps(r,indent=2))
(out/'internal-contacts.json').write_text(json.dumps(allrows,indent=2)+'\n')
