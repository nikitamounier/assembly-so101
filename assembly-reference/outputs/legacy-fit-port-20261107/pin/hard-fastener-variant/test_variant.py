"""Exactly one legacy fastener normal-law comparison, no geometry changes."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np,mujoco
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from scripts.diagnose_wowrobo_insertion import configure,reactions
ROOT=Path('/Users/nikitamounier/Documents/ChatGPT/robot assembly 2/robotics-assembly-3.0')
OUT=Path(__file__).resolve().parent

def run(speed):
 load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');set_surface_mode(1)
 alignment=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/pin/alignment.json').read_text());axis=np.array(alignment['axis_world']);duration=4 if speed==.02 else 12
 results=[]
 for profile,filename in [('baseline_v1','baseline-v1-guide.xml'),('exact_old_hard_normal','hard-fastener-guide.xml')]:
  begin=time.monotonic();m=mujoco.MjModel.from_xml_path(str(OUT/filename));active=configure(m,'assembly_pin',['base_servo','base_side_cover','shoulder_connector'],True);m.dof_damping[:]=20/speed;d=mujoco.MjData(m);mujoco.mj_forward(m,d);push=BalancedPush(m,d,active,axis,damping=0);trace=[];stop='duration elapsed';guard=.017220118
  for _ in range(round(duration/m.opt.timestep)):
   force=20*min(1,float(d.time)/.8);push.apply(d,force);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
   depth=max(0,-min((c.dist for c in d.contact),default=0));trace.append([float(d.time),float(d.qpos[0]),float(d.qvel[0]),depth,float(d.ncon),force])
   if not np.isfinite(d.qpos).all():stop='nonfinite';break
   if d.qpos[0]>guard:stop='source barrier guard';break
   if depth>.0005:stop='penetration guard';break
   if d.qpos[0]>=.027:stop='reference crossed; invalid fit';break
  push.apply(d,20);mujoco.mj_forward(m,d);t=np.array(trace);label=f'{profile}-{speed*1000:g}mmps';np.savez_compressed(OUT/(label+'-trace.npz'),trace=t,qpos=d.qpos,qvel=d.qvel)
  r={'profile':profile,'speed_mm_s':speed*1000,'force_N':20,'damping_Ns_m':20/speed,'source_scene_sha256':hashlib.sha256((OUT/filename).read_bytes()).hexdigest(),'scene_file':filename,'start':'fresh trueborealigned5mmstand_off','simulated_s':float(d.time),'duration_limit_s':duration,'final_axial_mm':float(d.qpos[0]*1000),'max_axial_all_substeps_mm':float(t[:,1].max()*1000),'min_axial_all_substeps_mm':float(t[:,1].min()*1000),'peak_native_penetration_mm':float(t[:,3].max()*1000),'source_guard_mm':guard*1000,'source_guard_crossed':bool(t[:,1].max()>guard),'stop':stop,'physical_success':False,'trace_rows':len(t),'trace_columns':['time_s','axial_m','velocity_m_s','native_penetration_m','contacts','force_N'],'final_reaction':reactions(m,d,active,axis),'wall_s':time.monotonic()-begin}
  r['negative_guard_pass']=not r['source_guard_crossed']and stop=='duration elapsed'and r['final_axial_mm']<27
  (OUT/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='final_reaction'}),flush=True);results.append(r)
 (OUT/f'comparison-{speed*1000:g}mmps.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--speed',type=float,choices=[.02,.002],required=True);a=p.parse_args();run(a.speed)
