import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'loose-forearm-v1';p=json.load(open(ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json'));load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));d=mujoco.MjData(m);z=np.load(OUT/'latest.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];mujoco.mj_forward(m,d)
rows=[]
for i in range(d.ncon):
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
 rows.append({'geoms':[m.geom(int(g)).name for g in d.contact.geom[i]],'pos':d.contact.pos[i].tolist(),'normal':d.contact.frame[i,:3].tolist(),'dist':float(d.contact.dist[i]),'force':f.tolist()})
com=sum(m.body_mass[i]*d.xipos[i]for i in range(1,m.nbody))/sum(m.body_mass)
r={'contacts':rows,'com_world':com.tolist(),'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'body_inertias':{m.body(i).name:{'mass':float(m.body_mass[i]),'com_world':d.xipos[i].tolist()}for i in range(1,m.nbody)}};print(json.dumps(r,indent=2));(OUT/'contact-review.json').write_text(json.dumps(r,indent=2)+'\n')
