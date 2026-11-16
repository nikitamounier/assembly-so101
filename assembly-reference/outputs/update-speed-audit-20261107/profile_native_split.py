"""Isolated zero-action single-world timings; never attaches to the live learner."""
from pathlib import Path
import ctypes,json,time,sys,hashlib
import mujoco,numpy as np
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
ROOT=Path(sys.argv[1]);bank=ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5';meta=json.loads((bank/'manifest.json').read_text())
lib=ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))));clock=ctypes.CDLL('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/artifacts_local/chimera/profiler-clock.so');slot=ctypes.c_void_p.in_dll(lib,'mjcb_time');previous=slot.value;slot.value=ctypes.cast(clock.assembly_profile_time,ctypes.c_void_p).value
names=[mujoco.mjtTimer(i).name for i in range(int(mujoco.mjtTimer.mjNTIMER))]
env=BatchedAssemblyEnv(meta['task_names'],np.zeros(1,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=1));engine=env.engine
backend=sys.modules['_wowrobo_training_native_backend'];native=backend.surface_library()
native.so101_surface_stats.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int]

measure={}
class Engine:
 def rollout(self,*args,**kwargs):
  stats=np.zeros(7);native.so101_surface_stats(stats,1)
  t=time.perf_counter();r=engine.rollout(*args,**kwargs);measure['rollout_s']=time.perf_counter()-t
  native.so101_surface_stats(stats,1);measure['sdf_search_s']=float(stats[1]);measure['surface_supplement_s']=float(stats[2]);measure['sdf_pair_calls']=int(stats[0]);return r
 def close(self):return engine.close()
env.engine=Engine();refresh=env._refresh

def forward():
 t=time.perf_counter();r=refresh();measure['refresh_s']=time.perf_counter()-t;return r
env._refresh=forward
cases=[('base_side_cover','00_loose',False),('base_side_cover','05_cover_near',False),('upper_arm_module','21_upper_far_side',False),('upper_arm_module','21_upper_far_side',True)]
rows=[]
try:
 for task,stage,variation in cases:
  z=np.load(bank/(task+'_train.npz'));ids=np.flatnonzero(z['stage']==stage)
  matching=[i for i in ids if bool(np.any(z['states'][i,-3:]!=1))==variation];assert matching
  state=z['states'][matching[0]].copy();ti=meta['task_names'].index(task);env.set_states(np.array([0]),state[None],np.array([ti]))
  for d in env.work+env.data:
   for timer in d.timer:timer.duration=0;timer.number=0
  measure.clear();t=time.perf_counter();reward,done,info=env.step(np.zeros((1,12)));elapsed=time.perf_counter()-t
  work={name:dict(seconds=float(env.work[0].timer[i].duration),calls=int(env.work[0].timer[i].number)) for i,name in enumerate(names)}
  row=dict(task=task,stage=stage,variation=variation,friction=state[-3:].tolist(),wall_s=elapsed,**measure,other_s=elapsed-measure['rollout_s']-measure['refresh_s'],native_timers=work,ncon=int(env.work[0].ncon),solver_niter=env.work[0].solver_niter.tolist(),safety=bool(info['safety'][0]));rows.append(row);print(json.dumps(dict(case=stage,variation=variation,wall_s=elapsed,rollout_s=measure['rollout_s'])),flush=True)
 result=dict(scope='Four isolated one-world zero-action control intervals, not production-wide percentages',rows=rows,model_signature=meta['model_signature'],timestep=float(env.m.opt.timestep),control_dt=env.ecfg.control_dt,substeps=env.substeps,nbody=env.m.nbody,nv=env.m.nv,ngeom=env.m.ngeom,npair=env.m.npair,options={k:float(getattr(env.m.opt,k)) for k in ['iterations','ls_iterations','tolerance','sdf_initpoints','sdf_iterations']})
 Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
finally:slot.value=previous;env.close()
