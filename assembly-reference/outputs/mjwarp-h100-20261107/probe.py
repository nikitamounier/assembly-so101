"""Bounded stock-MJWarp compatibility and synthetic H100 control probe."""
import os,sys,json,time,traceback,subprocess,hashlib
from pathlib import Path
import numpy as np
P=Path('/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107')
R=Path('/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4')
result={'scope':'stock MJWarp current WowRobo compatibility; synthetic sphere benchmark is NOT assembly throughput','time':time.time()}
def save():
 (P/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
def event(x):print(json.dumps(x,default=str),flush=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
if apps.strip():raise SystemExit('GPU busy: refusing probe')
# Yield immediately if the retained allocation acquires another GPU workload.
import threading
probe_pid=os.getpid()
def watch_gpu():
 while True:
  time.sleep(1)
  try:
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
   others=[x.strip() for x in apps.splitlines() if x.strip() and x.strip()!=str(probe_pid)]
   if others:
    print('Yielding to other GPU workload: '+str(others),flush=True)
    os._exit(75)
  except subprocess.CalledProcessError:os._exit(76)
threading.Thread(target=watch_gpu,daemon=True).start()
import mujoco,mujoco_warp as mjw,warp as wp,importlib.metadata as md
result['versions']={p:md.version(p) for p in ['mujoco','mujoco-warp','warp-lang']}
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
wp.config.kernel_cache_dir=str(P/'warp-cache');wp.init();wp.set_device('cuda:0');save();event(result)
# Probe exact current task first. Never run contact physics with absent SDF hooks.
os.environ['WOWROBO_CANONICAL_ROOT']=str(R/'robotics-assembly-3.0')
sys.path.insert(0,str(R/'assembly-so101'))
sys.path.append('/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/lib/python3.12/site-packages')
from assembly_policy.wowrobo.runtime import load_runtime
bank=R/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5'
meta=json.loads((bank/'manifest.json').read_text())
env=None
try:
 m,profile,model_signature=load_runtime(meta['profile_path'])
 result['assembly']={'model_signature':model_signature,'bank_signature':meta.get('model_signature'),'nq':m.nq,'nv':m.nv,'ngeom':m.ngeom,'npair':m.npair,'nplugin':m.nplugin,'timestep':m.opt.timestep,'plugins':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_PLUGIN,i) for i in range(m.nplugin)],'sdf_geoms':np.flatnonzero(m.geom_type==int(mujoco.mjtGeom.mjGEOM_SDF)).tolist()}
 save();event(result['assembly'])
 try:
  t=time.perf_counter();gm=mjw.put_model(m);wp.synchronize()
  result['assembly']['stock_import']={'passed':True,'seconds':time.perf_counter()-t,'warning':'Import alone does not port native custom callbacks or establish SDF equivalence'}
 except Exception as e:
  result['assembly']['stock_import']={'passed':False,'error':repr(e),'traceback':traceback.format_exc()}
 result['assembly']['qualification']='NOT QUALIFIED: native custom supplemental contacts and friction hooks are not ported; no assembly GPU stepping or training'
except Exception as e:
 result['assembly_load_error']={'error':repr(e),'traceback':traceback.format_exc()}
finally:
 if env:env.close()
 save();event(result)
# Separate synthetic benchmark checks that installed Warp/CUDA actually executes.
xml='<mujoco><option timestep="0.00025"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .052"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>'
rows=[]
try:
 for n in [1,64,256]:
  m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
  t=time.perf_counter();gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=n,nconmax=16,njmax=64)
  mjw.step(gm,gd);wp.synchronize();compile_s=time.perf_counter()-t
  # Reset after compile. Capture one step; time160 sequential launches per40ms.
  wp.copy(gd.qpos,wp.array(np.tile(d.qpos,(n,1)),dtype=wp.float32));gd.qvel.zero_();gd.time.zero_()
  with wp.ScopedCapture() as capture:mjw.step(gm,gd)
  wp.synchronize();elapsed=[]
  for rep in range(3):
   t=time.perf_counter()
   for k in range(160):wp.capture_launch(capture.graph)
   wp.synchronize();elapsed.append(time.perf_counter()-t)
  native=mujoco.MjData(m);t=time.perf_counter()
  for k in range(480):mujoco.mj_step(m,native)
  native_s=time.perf_counter()-t
  q=gd.qpos.numpy()
  row={'worlds':n,'compile_and_first_step_s':compile_s,'seconds_per_160_substeps':elapsed,'physics_world_control_intervals_per_s':n/float(np.median(elapsed)),'native_one_world_seconds_480_steps':native_s,'finite':bool(np.isfinite(q).all()),'max_abs_qpos_error_vs_native':float(np.max(np.abs(q-native.qpos))),'note':'synthetic sphere on plane; excludes assembly task, resets, observations, reward and PPO'}
  rows.append(row);result['synthetic']=rows;save();event(row)
except Exception as e:
 result['synthetic_error']={'error':repr(e),'traceback':traceback.format_exc()};save();event(result['synthetic_error'])
save();event({'finished':True})
