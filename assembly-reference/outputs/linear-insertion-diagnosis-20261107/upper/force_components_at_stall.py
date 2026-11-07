from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend
root=Path(__file__).resolve().parents[3];p=root/'outputs/linear-insertion-aligned-20261107/upper';out=Path(__file__).resolve().parent;load_backend(root/'outputs/linear-insertion-updated-20261107/grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(p/'guide-scene.xml'));d=mujoco.MjData(m);r=json.load(open(p/'receipt.json'));m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.body_contype[:]=0;m.body_conaffinity[:]=0
for key,a,b in [('active_contact_geoms',1,2),('receiver_contact_geoms',2,1)]:
 for name in r[key]:
  gid=m.geom(name).id;bid=m.geom_bodyid[gid];m.geom_contype[gid]=a;m.geom_conaffinity[gid]=b;m.body_contype[bid]|=a;m.body_conaffinity[bid]|=b
state=np.load(p/'final-state.npz');d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.qfrc_applied[0]=.5-10*d.qvel[0];mujoco.mj_forward(m,d);axis=np.array(r['axis_world']);active=set(r['active_contact_geoms']);rows=[]
for i,c in enumerate(d.contact[:d.ncon]):
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);frame=c.frame.reshape(3,3);sign=1 if m.geom(c.geom2).name in active else -1
 normal=sign*f[0]*frame[0];tangent=sign*(f[1]*frame[1]+f[2]*frame[2]);rows.append({'pair':[m.geom(c.geom1).name,m.geom(c.geom2).name],'local_contact_force_6':f.tolist(),'normal_axial_resistance_N':-float(normal@axis),'tangential_axial_resistance_N':-float(tangent@axis),'normal_world':c.frame[:3].tolist()})
baseline_constraint=d.qfrc_constraint.copy();m.geom_condim[:]=1;m.geom_friction[:]=0;mujoco.mj_forward(m,d);frictionless_constraint=d.qfrc_constraint.copy();record={'frictionless_state_intervention':{'condim':1,'geom_friction':[0,0,0],'generalized_contact_reaction_N':frictionless_constraint.tolist(),'delta_vs_baseline_N':(frictionless_constraint-baseline_constraint).tolist()},'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'generalized_drive_with_com_drag_N':d.qfrc_applied.tolist(),'generalized_contact_reaction_N':d.qfrc_constraint.tolist(),'reconstruction':'Frozen final savedstate, constant0.5N drive and10Ns/m COMdrag projected onto exact1DOFaxis. One mj_forward resolves force; not newtimeintegration.','rows':rows};(out/'stall-force-components.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
