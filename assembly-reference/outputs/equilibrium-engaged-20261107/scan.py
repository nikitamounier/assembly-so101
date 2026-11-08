"""Static passive-hinge scan at exact held motor-case pose, not equilibrium proof."""
import json,time
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_repaired_resets import RepairedResetSequence,body_pose
from assembly_policy.so101.wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
fitter=object.__new__(RepairedResetSequence);fitter.model=m
receipts={r['id']:r for r in json.loads((ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stage-receipts.json').read_text())}
roots=['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module']
def rootof(b):
 while b:
  n=m.body(b).name
  if n in roots:return n
  b=int(m.body_parentid[b])
 return None
geomroots={g:rootof(int(m.geom_bodyid[g])) for g in range(m.ngeom)}
results={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','25_right_support','29_final_far_side','30_final_half','31_final_near']:
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/f'{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=int(stage.split('_')[0])>=first
 mujoco.mj_forward(m,d);q0=d.qpos.copy()
 task='upper_arm_module' if stage.startswith(('21','22','23','25')) else 'forearm_module';servo=2 if task=='upper_arm_module' else 3
 p0,r0=body_pose(d,f'target_servo_{servo}_pivot'); h0=float(d.joint(f'target_servo_{servo}_shaft').qpos[0])
 rows=[]
 for theta in np.linspace(h0-.5,h0+.5,61):
  d.qpos[:]=q0;fitter._set_case_with_hinge(d,task,p0,r0,float(theta))
  collisions=[]; own=[]
  for c in d.contact:
   a,b=map(int,c.geom);ra,rb=geomroots[a],geomroots[b]
   if ra is None or rb is None:continue
   row={'pair':[m.geom(a).name,m.geom(b).name],'distance_m':float(c.dist)}
   if ra!=rb and task in [ra,rb]:collisions.append(row)
   elif ra==rb==task:own.append(row)
  rows.append({'theta_rad':float(theta),'theta_delta_deg':float(np.rad2deg(theta-h0)), 'worst_active_contact_m':min((x['distance_m']for x in collisions),default=0.),'worst_own_contact_m':min((x['distance_m']for x in own),default=0.),'active_pairs':sorted(collisions,key=lambda x:x['distance_m'])[:3]})
 results[stage]={'task':task,'source_hinge_rad':h0,'rows':rows,'best':sorted(rows,key=lambda r:(min(r['worst_active_contact_m'],r['worst_own_contact_m']),-abs(r['theta_delta_deg'])),reverse=True)[:5]}
 print(stage,'source',np.rad2deg(h0),'best',results[stage]['best'][0],flush=True)
(OUT/'scan.json').write_text(json.dumps(results,indent=2)+'\n')
