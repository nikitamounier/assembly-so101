"""Collision-free gravity-potential initializers; no angle targets in dynamics."""
import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
roots=['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module']
def rootof(b):
 while b:
  n=m.body(b).name
  if n in roots:return n
  b=int(m.body_parentid[b])
 return None
geomroots={g:rootof(int(m.geom_bodyid[g])) for g in range(m.ngeom)}
def update():mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
def pose(b):return d.xpos[m.body(b).id].copy(),d.xmat[m.body(b).id].reshape(3,3).copy()
def setcase(task,servo,position,rotation,hinge):
 d.qpos[m.joint(f'target_servo_{servo}_shaft').qposadr[0]]=hinge
 mujoco.mj_kinematics(m,d);rp,rr=pose(task);cp,cr=pose(f'target_servo_{servo}_pivot');delta=rotation@cr.T
 a=m.joint(task+'_free').qposadr[0];d.qpos[a:a+3]=position+delta@(rp-cp);mujoco.mju_mat2Quat(d.qpos[a+3:a+7],(delta@rr).ravel());update()
results={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','25_right_support','29_final_far_side','30_final_half','31_final_near']:
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/f'{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage.split('_')[0])>=first
 update();q0=d.qpos.copy();task='upper_arm_module' if stage.startswith(('21','22','23','25')) else 'forearm_module';servo=2 if task=='upper_arm_module' else 3
 cp,cr=pose(f'target_servo_{servo}_pivot');h0=float(d.joint(f'target_servo_{servo}_shaft').qpos[0]);rows=[];candidates=[]
 for h in np.linspace(h0-.7,h0+.7,141):
  d.qpos[:]=q0;setcase(task,servo,cp,cr,h)
  energy=sum(float(m.body_mass[b]*9.81*d.xipos[b,2])for b in range(m.nbody)if rootof(b)==task)
  printed=[];own=[]
  for c in d.contact:
   a,b=map(int,c.geom);an,bn=m.geom(a).name,m.geom(b).name;ra,rb=geomroots[a],geomroots[b]
   # Only incoming printed link surfaces. Servo-case pocket contact is unchanged.
   if not any(n and n.startswith(task+'_printed')for n in [an,bn]):continue
   if ra==rb==task:own.append(float(c.dist))
   elif task in [ra,rb]and((ra is not None and rb is not None)or('review_table'in[an,bn])):printed.append(float(c.dist))
  depth=min(printed+own+[0.]);row={'hinge_rad':float(h),'delta_degrees':float(np.rad2deg(h-h0)),'energy_J':energy,'minimum_printed_contact_m':min(printed+[0.]),'minimum_own_contact_m':min(own+[0.]),'feasible':depth>=-.00001}
  rows.append(row)
  if row['feasible']:candidates.append((energy,abs(h-h0),d.qpos.copy(),row))
 if candidates:
  _,_,q,row=min(candidates,key=lambda x:x[:2]);results[stage]={'source_hinge_rad':h0,'candidate':row,'rows':rows};np.savez_compressed(OUT/(stage+'-initializer.npz'),qpos=q,ctrl=d.ctrl,source_qpos=q0)
  print(stage,row,flush=True)
 else:results[stage]={'source_hinge_rad':h0,'candidate':None,'rows':rows};print(stage,'NONE',flush=True)
(OUT/'energy-candidates.json').write_text(json.dumps(results,indent=2)+'\n')
