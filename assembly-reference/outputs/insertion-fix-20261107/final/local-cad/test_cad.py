from pathlib import Path
import argparse,json,os,mujoco,numpy as np,time
from assembly_policy.so101.wowrobo_contact_backend import prepare_scene,set_surface_mode
from assembly_policy.so101.wowrobo_final_fit import replace_final_pocket_asset
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from assembly_policy.so101.wowrobo_mating_profile import write_profile_scene
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
from scripts.diagnose_wowrobo_insertion import configure,reactions
ROOT=Path.cwd();p=argparse.ArgumentParser();p.add_argument('--mode',choices=['coarse','fine'],default='coarse');args=p.parse_args();out=ROOT/'outputs/insertion-fix-20261107/final/local-cad'/args.mode;out.mkdir(exist_ok=True);grid=out/'grids';grid.mkdir(exist_ok=True);base=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text());new=json.loads((out.parent/('upper-clearance-coarse-grid.json'if args.mode=='coarse'else'upper-clearance-grid.json')).read_text());base['grid_directory']=str(grid);base['grid_dir']=str(grid)
for name,row in base['meshes'].items():
 if name=='upper_arm_module_printed_0':base['meshes'][name]=dict(new);row=base['meshes'][name]
 target=grid/f"grid_{row['grid_id']}.bin";source=Path(row['grid_file'])
 if not target.exists():os.link(source,target)
 row['grid_file']=str(target)
manifest=grid/'manifest.json';manifest.write_text(json.dumps(base,indent=2)+'\n');replace_final_pocket_asset(ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml',out.parent/'upper_arm_module_printed_0.obj',out/'source-scene.xml');prepare_scene(out/'source-scene.xml',manifest,out/'full-scene.xml');a=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/final/alignment.json').read_text());(out/'alignment.json').write_text(json.dumps(a,indent=2)+'\n');build_guide(out/'full-scene.xml',out/'alignment.json',out/'guide-scene.xml','forearm_module');write_profile_scene(out/'guide-scene.xml',out/'candidate-scene.xml','final');m=mujoco.MjModel.from_xml_path(str(out/'candidate-scene.xml'));active=configure(m,'forearm_module',['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module'],True);m.dof_damping[:]=1000;d=mujoco.MjData(m);mujoco.mj_forward(m,d);axis=np.array(a['axis_world']);push=BalancedPush(m,d,active,axis,damping=0);set_surface_mode(1);trace=[];peak=0;reason='duration';start=time.monotonic()
for i in range(7000):
 push.apply(d,20);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth);trace.append([d.time,d.qpos[0],d.qvel[0],depth,d.ncon,20.])
 if depth>.0005:reason='penetrationguard';break
 if d.qpos[0]>=.025:reason='crossedreference';break
qdrive=float(d.qpos[0]);drivev=float(d.qvel[0]);drivetime=float(d.time);np.savez_compressed(out/'drive-state.npz',qpos=d.qpos,qvel=d.qvel);push.apply(d,20);mujoco.mj_forward(m,d);reaction=reactions(m,d,active,axis)
for i in range(1000):
 push.apply(d,0);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth);trace.append([d.time,d.qpos[0],d.qvel[0],depth,d.ncon,0.])
 if depth>.0005:reason+=';relaxpenetrationguard';break
au=NativeBoundaryAudit(m,max_samples=4096);ag=[g for g in au.targets if au.info[g]['piece']=='forearm_module'];others=[g for g in au.targets if au.info[g]['piece']!='forearm_module'];b=au._bounds(d,au.info);pairs=au._pairs(d,[(x,y) for x in ag for y in others],b);sourcepeak=max([x['penetration_m'] for x in pairs],default=0);r={'route':'ExplicitlocalCADrelief,UNCHANGEDoldcaseforklaw','grid_mode':args.mode,'case_grid_spacing_m':.0004,'receiver_grid_spacing_m':new['spacing'],'drive_mm':qdrive*1000,'drive_v':drivev,'drive_time_s':drivetime,'relaxed_mm':float(d.qpos[0])*1000,'peak_native_mm':peak*1000,'relaxed_source_peak_mm':sourcepeak*1000,'stop':reason,'wall_seconds':time.monotonic()-start,'final_pair_solimp':m.pair_solimp.tolist(),'drive_reaction':reaction,'unified_prerequisites_current':False,'trace_columns':['time','qpos','qvel','depth','contacts','drive_N']};(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');tr=np.array(trace);np.savez_compressed(out/'trajectory.npz',rows=tr,time=tr[:,0],qpos=tr[:,1,None],qvel=tr[:,2,None]);print(json.dumps({k:v for k,v in r.items() if k!='drive_reaction'},indent=2),flush=True)
