"""Full compatible-bank initial admission under the actual active native runtime."""
from pathlib import Path
import concurrent.futures,hashlib,json,threading,time
import numpy as np
from assembly_policy.wowrobo import env as native
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'bank-19200-admission';BANK=OUT/'training-bank-v14'
meta=json.loads((BANK/'manifest.json').read_text());tasks=meta['task_names']
m,profile,signature=native.load_runtime(meta['profile_path'])
assert signature==meta['model_signature']
local=threading.local();environments=[];lock=threading.Lock();entries=[]
for ti,task in enumerate(tasks):
    for split in ('train','eval'):
        with np.load(BANK/(task+'_'+split+'.npz')) as z:
            for i,state in enumerate(z['states']):entries.append((task,split,i,ti,state.copy()))
def inspect(row):
    task,split,index,ti,state=row
    if not hasattr(local,'env'):
        local.env=native.BatchedAssemblyEnv(tasks,np.zeros(1,np.int64),native.EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=1))
        with lock:environments.append(local.env)
    e=local.env;e.set_states(np.array([0]),state[None],np.array([ti]));d=e.data[0]
    obs=e.observe();assert all(np.isfinite(a).all() for a in obs.values())
    depth=float(max(0,-min(d.contact.dist,default=0)))
    return dict(task=task,split=split,index=index,depth_m=depth,warning=bool(e.native_warning[0]),finite=True,
                obs_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(obs[k]).tobytes() for k in sorted(obs))).hexdigest())
t=time.perf_counter()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(inspect,entries))
finally:
    for e in environments:e.close()
bad=[r for r in rows if r['depth_m']>.0005+1e-12 or r['warning']]
result=dict(status='PASS' if not bad else 'FAIL',physics_signature=signature,rows=len(rows),train_rows=sum(r['split']=='train' for r in rows),eval_rows=sum(r['split']=='eval' for r in rows),bad=bad,max_depth_m=max(r['depth_m'] for r in rows),all_observations_finite=True,seconds=time.perf_counter()-t,scope='Every converted and preserved reset row, actual native0.5ms/hinge/friction forward; source prior dynamic admission retained, full-episode convergence not certified.')
(OUT/'initial-admission.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'initial-admission-rows.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
print(json.dumps(result),flush=True)
assert not bad,bad[:10]
