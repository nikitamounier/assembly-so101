from pathlib import Path
import json,numpy as np,mujoco,time,hashlib,shutil
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_mating_profile import write_profile_scene
from scripts.diagnose_wowrobo_insertion import configure,reactions
ROOT=Path.cwd();out=Path(__file__).resolve().parent;backend=load_backend(out/'grids/manifest.json');set_surface_mode(1);a=json.loads((out/'alignment-prepared.json').read_text());axis=np.array(a['axis_world']);results=[];(out/'backend-manifest.json').write_text(json.dumps(backend,indent=2)+'\n');shutil.copyfile(out/'alignment-prepared.json',out/'alignment.json')
for name,source in [('candidate',out/'guide-scene.xml'),('negative-wall',out/'negative-guide-scene.xml')]:
 scene=out/(name+'-scene.xml');profile=write_profile_scene(source,scene,'final');m=mujoco.MjModel.from_xml_path(str(scene));assert np.allclose(m.pair_solimp[0],[.1,.99,.0002,.8,2],atol=0,rtol=0)
 active=configure(m,'forearm_module',['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module'],True);m.dof_damping[:]=1000.;d=mujoco.MjData(m);mujoco.mj_forward(m,d);push=BalancedPush(m,d,active,axis,damping=0);trace=[];peak=0;stop='duration';reaction=[];driveq=0
 def append(force,phase):trace.append([float(d.time),float(d.qpos[0]),float(d.qvel[0]),force,phase,float(d.ncon),max(0,-min((float(c.dist) for c in d.contact),default=0))])
 append(20,0)
 for i in range(16000 if name=='candidate' else 6000):
  push.apply(d,20);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth);append(20,0)
  if i%100==0:reaction.append({'time':float(d.time),'q_mm':float(d.qpos[0]*1000),'reaction':reactions(m,d,active,axis)})
  if depth>.0005:stop='penetrationguard';break
  if d.qpos[0]>=.025:stop='referencedepthcrossed';break
 driveq=float(d.qpos[0]);drivev=float(d.qvel[0]);drivetime=float(d.time);np.savez_compressed(out/(name+'-drive-state.npz'),qpos=d.qpos,qvel=d.qvel)
 push.apply(d,20);mujoco.mj_forward(m,d);drive_reaction=reactions(m,d,active,axis)
 for i in range(1000):
  push.apply(d,0);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth);append(0,1)
  if depth>.0005:stop+=';releasepenetrationguard';break
 push.apply(d,0);mujoco.mj_forward(m,d);r={'case':name,'profile':profile,'full_trace':name+'-trajectory.npz','trace_columns':['time_s','qpos_axial_m','qvel_mps','drive_N','phase_0drive_1relax','contacts','native_penetration_m'],'force_N':20,'max_speed_mps':.02,'damping_Ns_per_m':1000,'drive_time_s':drivetime,'drive_mm':driveq*1000,'drive_velocity_mps':drivev,'relaxed_mm':float(d.qpos[0]*1000),'peak_native_penetration_mm':peak*1000,'stop':stop,'drive_final_reaction':drive_reaction,'relaxed_final_reaction':reactions(m,d,active,axis),'reaction_snapshots':reaction,'drive_model_xml':str(scene),'alignment_json':str(out/'alignment.json'),'backend_manifest':str(out/'backend-manifest.json'),'case_geometry_changed':False,'printed_geometry_variant':'FrozenconforminglocalCADclearancev3','contact_law':'Unchangedlegacycaseforkwidth0.2mm','qualification':'Frozen derived CAD; unchanged legacy contact law; fixed receivers and explicit axial guide; hardware calibration separate'};np.savez_compressed(out/(name+'-trajectory.npz'),trace=np.array(trace),time_s=np.array(trace)[:,0],qpos=np.array(trace)[:,1,None],qvel=np.array(trace)[:,2,None],force_N=np.array(trace)[:,3]);(out/(name+'-recording.json')).write_text(json.dumps(r,indent=2)+'\n');results.append({k:v for k,v in r.items() if k!='reaction_snapshots'});print({k:r[k] for k in ['case','drive_mm','relaxed_mm','peak_native_penetration_mm','drive_time_s','stop']},flush=True)
(out/'recordings.json').write_text(json.dumps(results,indent=2)+'\n')
