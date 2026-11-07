from pathlib import Path
import json,mujoco,numpy as np,time
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from scripts.diagnose_wowrobo_insertion import configure
ROOT=Path.cwd();out=ROOT/'outputs/insertion-fix-20261107/final';load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');set_surface_mode(1);a=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/final/alignment.json').read_text());axis=np.array(a['axis_world']);results=[]
for width in [.0003,.0004,.0005]:
 t=time.monotonic();m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/legacy-fit-port-20261107/final/legacy_mapped_20N.xml'));active=configure(m,'forearm_module',['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module'],True);m.pair_solimp[0,2]=width;m.dof_damping[:]=1000;d=mujoco.MjData(m);mujoco.mj_forward(m,d);push=BalancedPush(m,d,active,axis,damping=0);peak=0;reason='elapsed';trace=[]
 for i in range(6000):
  push.apply(d,20);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth)
  if i%100==0:trace.append([d.time,d.qpos[0],d.qvel[0],depth,d.ncon])
  if depth>.0005:reason='penetrationguard';break
  if d.qpos[0]>=.025:reason='crossed';break
 r={'width_m':width,'q_mm':float(d.qpos[0]*1000),'v':float(d.qvel[0]),'time':float(d.time),'peak_mm':peak*1000,'stop':reason,'wall':time.monotonic()-t,'trace':trace};results.append(r);np.savez_compressed(out/f'width-{width:g}-drive.npz',qpos=d.qpos,qvel=d.qvel);(out/f'width-{width:g}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='trace'},flush=True)
(out/'compliance-scan.json').write_text(json.dumps(results,indent=2)+'\n')
