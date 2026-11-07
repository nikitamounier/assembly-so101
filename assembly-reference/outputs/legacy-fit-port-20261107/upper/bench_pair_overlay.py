"""Independent upper-only named-pair overlay counterfactual; current profile untouched."""
from pathlib import Path
import json,time,xml.etree.ElementTree as ET
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;FROZEN=ROOT/'outputs/linear-insertion-aligned-20261107/upper';load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json')
r=json.load(open(FROZEN/'receipt.json'));mapping=json.load(open(OUT/'source-guard-mapping.json'));tree=ET.parse(FROZEN/'guide-scene.xml');root=tree.getroot();contact=root.find('contact')
if contact is None:contact=ET.SubElement(root,'contact')
legacy=mapping['legacy_pair']
for i,pair in enumerate(mapping['applicable_pairs']):
 attrs={k:(' '.join(map(str,v)) if isinstance(v,list) else str(v)) for k,v in legacy.items() if k not in ['geom1','geom2','name']};attrs.update(name=f'upper_legacy_fit_{i}',geom1=pair[0],geom2=pair[1]);ET.SubElement(contact,'pair',attrs)
adapted=OUT/'adapted-guide.xml';tree.write(adapted,encoding='unicode')

def build(scene):
 m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.body_contype[:]=0;m.body_conaffinity[:]=0
 for key,a,b in [('active_contact_geoms',1,2),('receiver_contact_geoms',2,1)]:
  for name in r[key]:
   gid=m.geom(name).id;bid=m.geom_bodyid[gid];m.geom_contype[gid]=a;m.geom_conaffinity[gid]=b;m.body_contype[bid]|=a;m.body_conaffinity[bid]|=b
 mujoco.mj_forward(m,d);active={m.body('upper_arm_module').id,m.body('target_servo_2_pivot').id};push=BalancedPush(m,d,active,np.array(r['axis_world']),damping=10.)
 return m,d,push

results=[]
for mode,scene in [('baseline',FROZEN/'guide-scene.xml'),('adapted',adapted)]:
 for force in [.5,5.,20.]:
  m,d,push=build(scene);rows=[];stop=None;peak=0.;t0=time.perf_counter()
  for step in range(8000):
   push.apply(d,force*min(1.,step*m.opt.timestep/.8));mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
   depth=max(0.,-float(np.min(d.contact.dist[:d.ncon]))) if d.ncon else 0.;peak=max(peak,depth)
   if step%100==0:rows.append({'time_s':float(d.time),'axial_mm':float(d.qpos[0]*1000),'velocity':float(d.qvel[0]),'contacts':int(d.ncon),'penetration_m':depth})
   if d.qpos[0]>=.025:stop='reference crossed';break
   if depth>.0005:stop='native overlap >0.5mm';break
  # Retain the real terminal velocity for a one-second force-release test.
  final=float(d.qpos[0]);released=None
  if mode=='adapted':
   for step in range(2000):push.apply(d,0.);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
   released=float(d.qpos[0])
  record={'profile':mode,'drive_N':force,'final_drive_axial_mm':final*1000,'final_release_axial_mm':released*1000 if released is not None else None,'release_change_mm':(released-final)*1000 if released is not None else None,'peak_drive_penetration_m':peak,'stop':stop or '4s duration','wall_s':time.perf_counter()-t0,'explicit_pairs':int(m.npair),'rows':rows};results.append(record);(OUT/f'prebench-{mode}-{force:g}N.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
(OUT/'prebench-summary.json').write_text(json.dumps(results,indent=2)+'\n')
