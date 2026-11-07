"""Temporary contact-clearance interventions; never edits source geometry/config."""
from pathlib import Path
import json,time
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from scripts.diagnose_wowrobo_insertion import configure,reactions
ROOT=Path.cwd();SOURCE=ROOT/'outputs/linear-insertion-aligned-20261107/cover';OUT=ROOT/'outputs/linear-insertion-diagnosis-20261107/cover'
load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');set_surface_mode(1)
a=json.loads((SOURCE/'alignment.json').read_text());saved=np.load(SOURCE/'final-state.npz');axis=np.array(a['axis_world']);target=a['authored_axial_travel_m'];results=[]
for gap in [0.,.00001,.00005,.0001]:
 start=time.monotonic();m=mujoco.MjModel.from_xml_path(str(SOURCE/'guide-scene.xml'));active=configure(m,'base_side_cover',['base_servo'],True);bid=m.body('base_side_cover').id;assert m.body_parentid[bid]==0;m.body_pos[bid,0]-=gap
 d=mujoco.MjData(m);d.qpos[:]=saved['qpos'];d.qvel[:]=0;mujoco.mj_forward(m,d);zero=reactions(m,d,active,axis);push=BalancedPush(m,d,active,axis,10);push.apply(d,.5);mujoco.mj_forward(m,d);loaded=reactions(m,d,active,axis);peak=0.;reason='duration elapsed'
 for _ in range(4000):
  push.apply(d,.5);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);peak=max(peak,max([-float(c.dist)for c in d.contact],default=0))
  if d.qpos[0]>=target:reason='reference crossed; not seating proof';break
  if peak>.0005:reason='0.5mm penetration guard';break
 push.apply(d,.5);mujoco.mj_forward(m,d);row={'temporary_rear_relief_m':gap,'world_body_translation_m':[-gap,0,0],'friction_unchanged':True,'mu':float(m.geom_friction[m.geom('base_side_cover_printed_0__contact').id,0]),'drive_N':.5,'zero_drive_zero_velocity_reaction':zero,'loaded_initial_reaction':loaded,'final_axial_mm':float(d.qpos[0]*1000),'final_velocity_mps':float(d.qvel[0]),'final_reaction':reactions(m,d,active,axis),'peak_native_penetration_m':peak,'stop':reason,'wall_seconds':time.monotonic()-start};results.append(row);np.savez_compressed(OUT/f'rear-relief-{gap*1e6:.0f}um-state.npz',qpos=d.qpos,qvel=d.qvel);(OUT/f'rear-relief-{gap*1e6:.0f}um.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items()if'reaction'not in k}),flush=True)
(OUT/'rear-relief-controls.json').write_text(json.dumps({'design':'Temporary rigidbodyoffsetonly; .5N,mu.7,sourceSDF+tree/sparse/guideunchanged;zeroinitialvelocityall4controls. Sourcegeometry/alignment/defaultfilesuntouched.','cases':results},indent=2)+'\n')
