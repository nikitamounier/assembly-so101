from common import *
import ctypes
import mujoco
import numpy as np

variant=sys.argv[1];mod=backend(variant)
timed='--native-timers' in sys.argv
if timed:
    lib=ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))))
    clock=ctypes.CDLL(str(OUT/'clock.so'));slot=ctypes.c_void_p.in_dll(lib,'mjcb_time');previous_clock=slot.value
    slot.value=ctypes.cast(clock.assembly_profile_time,ctypes.c_void_p).value
if variant=='instrumented':
    mod.build_native=lambda build_dir: Path(build_dir)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=CANON/'assets_local/wowrobo/training-accepted-v6';meta=json.loads((bank/'manifest.json').read_text())
assert meta['model_signature']==SIGNATURE
env=BatchedAssemblyEnv(meta['task_names'],np.zeros(1,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=1))
assert env.m.opt.timestep==.00025 and env.substeps==160
native=mod.surface_library();pair=np.zeros((256*256,14))
if variant=='instrumented':native.wow_profile_pairs.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int]
def stats():
    native.wow_profile_pairs(pair.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),1)
def role(a,b):
    if any('finger' in x or 'jaw' in x or 'helper' in x or 'grasp' in x for x in [a,b]):return 'grasp/helper'
    if any('bracket' in x for x in [a,b]) and any('motor' in x or 'servo' in x for x in [a,b]):return 'bracket/hardware (inspect body relationship)'
    return 'source-SDF assembly/hardware'
engine=env.engine;timing={};saved={};rows=[]
class Capture:
    def rollout(self,*a,**kw):
        if variant=='instrumented':mod.surface_stats(reset=True);stats()
        if timed:
            for d in env.work:
                for timer in d.timer:timer.duration=0;timer.number=0
        start=time.perf_counter();tr,se=engine.rollout(*a,**kw);timing['rollout_s']=time.perf_counter()-start
        if timed:timing['native_timers']={mujoco.mjtTimer(i).name:dict(seconds=float(t.duration),calls=int(t.number)) for i,t in enumerate(env.work[0].timer)}
        timing['trajectory']=tr.copy();timing['sensors']=se.copy()
        if variant=='instrumented':stats();timing['pairs']=pair.copy()
        return tr,se
    def close(self):engine.close()
env.engine=Capture();original_refresh=env._refresh
def refresh():
    t=time.perf_counter();r=original_refresh();timing['refresh_s']=time.perf_counter()-t;return r
env._refresh=refresh
cases=[('base_side_cover','00_loose',False),('base_side_cover','05_cover_near',False),('upper_arm_module','21_upper_far_side',False),('upper_arm_module','21_upper_far_side',True)]
try:
    for ci,(task,stage,variation) in enumerate(cases):
        z=np.load(bank/(task+'_train.npz'));ids=np.flatnonzero(z['stage']==stage)
        ids=[i for i in ids if bool(np.any(z['states'][i,-3:]!=1))==variation];assert ids
        state=z['states'][ids[0]].copy();ti=meta['task_names'].index(task);env.set_states(np.array([0]),state[None],np.array([ti]))
        timing.clear();start=time.perf_counter();reward,done,info=env.step(np.zeros((1,12)));wall=time.perf_counter()-start
        d=env.data[0];contacts=np.array([np.r_[c.dist,c.pos,c.frame,c.geom,c.friction] for c in d.contact]);forces=np.zeros((d.ncon,6))
        for k in range(d.ncon):mujoco.mj_contactForce(env.m,d,k,forces[k])
        arrays=dict(trajectory=timing['trajectory'],sensors=timing['sensors'],state=env.state.copy(),qpos=d.qpos.copy(),qvel=d.qvel.copy(),warmstart=d.qacc_warmstart.copy(),observation=env.hist.copy(),reward=reward,done=done,contacts=contacts,forces=forces,**{'info_'+k:v for k,v in info.items()})
        for k,v in arrays.items():saved[f'case{ci}_{k}']=v
        pairs=[]
        if variant=='instrumented':
            for ix in np.flatnonzero(timing['pairs'][:,0]):
                g1,g2=divmod(int(ix),256);names=[mujoco.mj_id2name(env.m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in [g1,g2]]
                bodies=[mujoco.mj_id2name(env.m,mujoco.mjtObj.mjOBJ_BODY,int(env.m.geom_bodyid[g])) for g in [g1,g2]]
                pairs.append(dict(geoms=[g1,g2],names=names,bodies=bodies,role=role(*names),stats=timing['pairs'][ix].tolist()))
            pairs.sort(key=lambda p:sum(p['stats'][1:3]),reverse=True)
        row=dict(case=ci,task=task,stage=stage,variation=variation,bank_row=int(ids[0]),friction=state[-3:].tolist(),wall_s=wall,rollout_s=timing['rollout_s'],refresh_s=timing['refresh_s'],pairs=pairs)
        if timed:row['native_timers']=timing['native_timers']
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='pairs'}),flush=True)
    tag=variant+('-timed' if timed else '')
    np.savez_compressed(OUT/(tag+'-parity.npz'),**saved)
    write(OUT/(tag+'-probe.json'),dict(rows=rows,hardware=hardware(),model_signature=SIGNATURE,timestep=env.m.opt.timestep,substeps=env.substeps,ngeom=env.m.ngeom,npair=env.m.npair,source_tree_sha256=sha(ROOT/'assembly-so101/plugin/so101_surface_tree.h'),scope='Four exact single-world zero-action intervals; pair timers cover native rollout only, excluding refresh/reset. Roles are lexical hints; body and geom names are authoritative.'))
finally:
    env.close()
    if timed:slot.value=previous_clock
