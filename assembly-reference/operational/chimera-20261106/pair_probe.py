"""Auxiliary 44-env/eight-step CPU probe; NOT the captured PPO iteration."""
import csv,ctypes,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a');sys.path.insert(0,str(ROOT))
assert os.environ.get('SLURM_JOB_ID')=='3598717'
OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106/pair-probe';OUT.mkdir(exist_ok=True)
assert not (OUT/'result.json').exists(),'Reconcile existing probe first'
mj=Path(mujoco.__file__).parent;library=next(mj.glob('libmujoco.so.*'))
subprocess.run(['cc','-shared','-fPIC','-O2','-I'+str(mj/'include'),str(ROOT/'operational/chimera-20261106/pair_profile.c'),str(library),'-Wl,-rpath,'+str(mj),'-o',str(OUT/'pair-profiler.so')],check=True)
from assembly_policy.train import Trainer,parse
from assembly_policy.sim.env import STATE_SPEC
args=parse(['--suite','so101','--tasks','all','--bank-dir',str(ROOT/'data/so101/banks_shared_v13'),'--run-dir',str(OUT/'probe-run'),'--init-policy',str(ROOT/'runs/chimera-v13-8192-20261106/checkpoint.pt'),'--device','cpu','--envs','44','--steps','8','--nthread','2','--no-eval','--so101-max-contact-depth','.0004','--so101-stability','pose','--so101-safety-penalty','1','--eps-q','.01','--seed','20261105'])
tr=Trainer(args);env=tr.env.env;m=env.m
prof=ctypes.CDLL(str(OUT/'pair-profiler.so'));prof.pair_start.argtypes=[ctypes.c_int];prof.pair_dump.argtypes=[ctypes.c_char_p]
# Verify a delegating collider wrapper preserves the physics state bit for bit.
def trial():
 d=mujoco.MjData(m);mujoco.mj_setState(m,d,env.state[0],STATE_SPEC);mujoco.mj_forward(m,d)
 for _ in range(8):mujoco.mj_step(m,d)
 state=np.empty(env.nstate);mujoco.mj_getState(m,d,state,STATE_SPEC);return state
before=trial();assert prof.pair_start(m.ngeom)==0;after=trial();assert np.array_equal(before,after),'Delegating profiler changed physics'
prof.pair_reset();start=time.perf_counter();batch,stats=tr.collect();elapsed=time.perf_counter()-start
assert prof.pair_dump(str(OUT/'pairs.csv').encode())==0
rows=[]
for r in csv.DictReader((OUT/'pairs.csv').open()):
 a,b=int(r['geom1']),int(r['geom2']);n=int(r['calls']);seconds=int(r['nanoseconds'])/1e9
 rows.append(dict(a=m.geom(a).name,b=m.geom(b).name,body_a=m.body(int(m.geom_bodyid[a])).name,body_b=m.body(int(m.geom_bodyid[b])).name,calls=n,seconds=seconds,contacts=int(r['contacts']),us_per_call=seconds*1e6/n))
rows.sort(key=lambda r:-r['seconds']);total=sum(r['seconds'] for r in rows)
result=dict(scope='Auxiliary eight-control-step, 44-env probe with checkpoint weights; not the uninterrupted PPO timing capture',envs=44,steps=8,valid_transitions=int(batch['valid'].sum()),wall_s=elapsed,sim_s=stats['sim'],pair_sum_thread_seconds=total,parity_bitwise=True,model_signature=tr.args.get('model_signature'),cpu_affinity=sorted(os.sched_getaffinity(0)),pairs=rows,search_config={k:__import__('assembly_policy.so101.workcell',fromlist=['configuration']).configuration()[k] for k in ['contact_search','motor_base_search','grasp_search','horn_search','assembly_search','solver']})
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');tr.env.close();print(json.dumps(dict(wall_s=elapsed,pair_sum_thread_seconds=total,top=rows[:10])),flush=True)
