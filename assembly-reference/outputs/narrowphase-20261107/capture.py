import os,sys,json,time,ctypes,hashlib,socket
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')=='3604284'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a');sys.path.insert(0,str(ROOT))
OUT=ROOT.parent/'narrowphase-20261107'
import mujoco
mode=sys.argv[1]
native=None
if mode=='stages':
 native=ctypes.CDLL(str(OUT/'stage.so'),mode=ctypes.RTLD_GLOBAL)
 assert native.so101_install_contact_search()==4
 import assembly_policy.so101.workcell as wc
 wc.load_contact_search=lambda:native
elif mode not in ('prepare','stock','timed'):
 native=ctypes.CDLL(str(OUT/'diagnostic.so'),mode=ctypes.RTLD_GLOBAL)
 native.np_engine();assert native.so101_install_contact_search()==4
 import assembly_policy.so101.workcell as wc
 wc.load_contact_search=lambda:native
 original_load=mujoco.mj_loadPluginLibrary
 mujoco.mj_loadPluginLibrary=lambda path: None if Path(path).name in ('libassemblenist_sdf.so','libso101_driver.so') else original_load(path)
 native.np_start.argtypes=[ctypes.c_void_p];native.np_dump.argtypes=[ctypes.c_char_p];native.np_pairs.argtypes=[ctypes.c_char_p];native.np_base.restype=ctypes.c_uint64
from assembly_policy.sim.env import STATE_SPEC
if mode=='prepare':
 from assembly_policy.train import Trainer,parse
 args=parse(['--suite','so101','--tasks','all','--bank-dir',str(ROOT/'data/so101/banks_shared_v13'),'--run-dir',str(OUT/'prepare-run'),'--init-policy',str(ROOT/'runs/chimera-v13-8192-20261106/checkpoint.pt'),'--device','cpu','--envs','44','--steps','8','--nthread','2','--no-eval','--so101-max-contact-depth','.0004','--so101-stability','pose','--so101-safety-penalty','1','--eps-q','.01','--seed','20261105'])
 tr=Trainer(args);env=tr.env.env;states=[];ctrls=[];tasks=[];step=env.step
 def record(action):
  before=env.state.copy();result=step(action);states.append(before);ctrls.append(env.last_ctrl.copy());tasks.append(env.task.copy());return result
 env.step=record;batch,stats=tr.collect()
 np.savez_compressed(OUT/'replay.npz',states=np.concatenate(states),ctrls=np.concatenate(ctrls),tasks=np.concatenate(tasks))
 (OUT/'workload.json').write_text(json.dumps(dict(envs=44,control_steps=8,valid=int(batch['valid'].sum()),model_signature=tr.args.get('model_signature'),configuration=__import__('assembly_policy.so101.workcell',fromlist=['configuration']).configuration()),indent=2))
 tr.env.close();print('PREPARE PASS',flush=True);sys.exit()
from assembly_policy.so101.scene import model
m=model(contact_monitor=True);d=mujoco.MjData(m)
if mode=='timed':
 native=ctypes.CDLL(str(OUT/'observer.so'),mode=ctypes.RTLD_GLOBAL)
 native.np_start.argtypes=[ctypes.c_void_p];native.np_dump.argtypes=[ctypes.c_char_p];native.np_pairs.argtypes=[ctypes.c_char_p];native.np_base.restype=ctypes.c_uint64
clocklib=ctypes.CDLL(str(ROOT/'artifacts_local/chimera/profiler-clock.so'))
mjlib=ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))))
ctypes.c_void_p.in_dll(mjlib,'mjcb_time').value=ctypes.cast(clocklib.assembly_profile_time,ctypes.c_void_p).value
replay=np.load(OUT/'replay.npz');states=replay['states'];ctrls=replay['ctrls']
results=[];counts=[];contacthash=hashlib.sha256();narrow=0.;wall=time.perf_counter()
if mode=='stages':native.stage_reset()
elif native:
 native.np_reset()
 if mode in ('traced','timed'):
  native.np_stride(8 if mode=='traced' else 64);native.np_start(m._address)
for idx,(state,ctrl) in enumerate(zip(states,ctrls)):
 mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,state,STATE_SPEC);d.ctrl[:]=ctrl
 # Forward reconstruction is included in this diagnostic narrow-phase workload.
 mujoco.mj_forward(m,d)
 for j in range(8):
  mujoco.mj_step(m,d)
  counts.append(d.ncon)
  for c in d.contact:
   contacthash.update(np.asarray([c.geom1,c.geom2],dtype=np.int64).tobytes());contacthash.update(np.array([c.dist,*c.pos,*c.frame]).tobytes())
 result=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,result,STATE_SPEC);results.append(result)
 narrow+=float(d.timer[mujoco.mjtTimer.mjTIMER_COL_NARROW].duration)
elapsed=time.perf_counter()-wall
np.savez_compressed(OUT/(mode+'-states.npz'),states=np.array(results),contacts=np.array(counts))
if native and mode in ('traced','timed'):
 native.np_stop();native.np_dump(str(OUT/('nodes.csv' if mode=='traced' else 'timed-nodes.csv')).encode());native.np_pairs(str(OUT/('pairs.csv' if mode=='traced' else 'timed-pairs.csv')).encode())
if mode=='stages':
 native.stage_dump(str(OUT/'stages.csv').encode());native.stage_events(str(OUT/'surface-events.txt').encode())
metadata=dict(mode=mode,wall_s=elapsed,narrow_thread_s=narrow,contact_sha256=contacthash.hexdigest(),contact_count=int(sum(counts)),frames=len(states),substeps_per_frame=8,includes_initial_forward=True,host=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),geom_names=[m.geom(i).name for i in range(m.ngeom)],base=native.np_base() if native and mode!='stages' else None)
(OUT/(mode+'.json')).write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
