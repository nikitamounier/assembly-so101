"""All existing reset rows: initial observations/rewards/grips and contact parity."""
from pathlib import Path
import concurrent.futures,hashlib,json,sys,threading,time
import numpy as np
from assembly_policy.wowrobo import env as native
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v14'
variant=sys.argv[1];assert variant in ('baseline','candidate')
base=ROOT/('native-v13' if variant=='baseline' else 'native-v14')
bank=base/('training-accepted-v13' if variant=='baseline' else 'training-accepted-v14')
meta=json.loads((bank/'manifest.json').read_text());tasks=meta['task_names']
m,profile,identity=native.load_runtime(meta['profile_path'])
local=threading.local();environments=[];lock=threading.Lock()
def digest(a):
    a=np.ascontiguousarray(a);assert np.isfinite(a).all()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
arrays={name:digest(getattr(m,name)) for name in dir(m) if isinstance(getattr(m,name,None),np.ndarray) and np.isfinite(getattr(m,name)).all()}
entries=[]
for task in tasks:
    for split in ('train','eval'):
        p=bank/(task+'_'+split+'.npz')
        if not p.exists():continue
        with np.load(p) as z:
            for i,state in enumerate(z['states']):entries.append((p.name,i,tasks.index(task),state.copy()))
def run(row):
    filename,index,task,state=row
    if not hasattr(local,'env'):
        local.env=native.BatchedAssemblyEnv(tasks,np.zeros(1,np.int64),native.EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=1))
        with lock:environments.append(local.env)
    e=local.env;e.set_states(np.array([0]),state[None],np.array([task]));d=e.data[0]
    obs={k:digest(v) for k,v in e.observe().items()}
    contacts=[dict(geoms=list(map(int,c.geom)),distance=float(c.dist),pos=c.pos.tolist(),frame=c.frame.tolist()) for c in d.contact]
    grasps=sorted({tuple(sorted(m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom)) for c in d.contact if c.dist<=.0001 and any('gripper' in m.body(int(m.geom_bodyid[int(g)])).name or 'moving_jaw' in m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom)})
    return dict(file=filename,index=index,obs=obs,reward=e.reward.snapshot(d,dict(task=tasks[task])),grasp=grasps,contacts=contacts,maxdepth=max(0,-min(d.contact.dist,default=0)),warning=bool(e.native_warning[0]))
start=time.perf_counter()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(run,entries))
finally:
    for e in environments:e.close()
x=dict(status='CAPTURED',variant=variant,model_signature=identity,rows=rows,arrays=arrays,seconds=time.perf_counter()-start,count=len(rows),threads=6)
(REL/('bank-'+variant+'.json')).write_text(json.dumps(x,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in x.items() if k not in ('rows','arrays')}),flush=True)
