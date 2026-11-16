"""Read-only source profiling of versioned native releases in fresh processes."""
import argparse,ctypes,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import mujoco
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('version',choices=['v10','v12','v13']);p.add_argument('mode',choices=['panel','ppo']);a=p.parse_args()
sys.path=[x for x in sys.path if Path(x).resolve()!=a.out.resolve()]
rel=a.root/('native-'+a.version);out=a.out/(a.version+'-repeat');out.mkdir(parents=True,exist_ok=True)
def write(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(a.root/'robotics-assembly-3.0')
backend_root=rel if a.version!='v10' else a.root
b=load('_wowrobo_training_native_backend',backend_root/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');original=b.load_backend;b.load_backend=lambda manifest:original(manifest,out/'native')
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/('training-accepted-'+a.version);meta=json.loads((bank/'manifest.json').read_text())
import socket
hardware=dict(host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),affinity=sorted(os.sched_getaffinity(0)))
sources={str(p.relative_to(rel)):sha(p) for p in rel.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cc','.c','.h','.json']}
write(out/(a.mode+'-identity.json'),dict(hardware=hardware,sources=sources,manifest_sha256=sha(bank/'manifest.json')))
import threading
class Sampler:
    def __init__(self,path):
        self.path=Path(path);self.ident=threading.get_ident();self.stop=threading.Event();self.rows=[]
    def start(self):
        self.begin=time.perf_counter();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        previous=self.begin
        while not self.stop.wait(.02):
            at=time.perf_counter();f=sys._current_frames().get(self.ident);stack=[]
            while f:
                stack.append(f'{f.f_code.co_name} ({f.f_code.co_filename}:{f.f_code.co_firstlineno})');f=f.f_back
            self.rows.append((stack[::-1],at-previous));previous=at
    def finish(self):
        self.end=time.perf_counter();self.stop.set();self.thread.join()
        if self.rows:self.rows[-1]=(self.rows[-1][0],self.rows[-1][1]+self.end-self.begin-sum(x[1] for x in self.rows))
        frames=[];index={};samples=[];weights=[]
        for stack,w in self.rows:
            ids=[]
            for frame in stack:
                if frame not in index:index[frame]=len(frames);frames.append({'name':frame})
                ids.append(index[frame])
            samples.append(ids);weights.append(w)
        write(self.path,{'$schema':'https://www.speedscope.app/file-format-schema.json','shared':{'frames':frames},'profiles':[{'type':'sampled','name':self.path.stem+' Python wall stacks, 20 ms target; GIL scheduling bias possible','unit':'seconds','startValue':0,'endValue':self.end-self.begin,'samples':samples,'weights':weights}]})

if a.mode=='panel':
 e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2));assert e.substeps==80 and e.m.opt.timestep==.0005
 engine=e.engine;spans={}
 class Capture:
  def rollout(self,*args,**kw):
   start=time.perf_counter();result=engine.rollout(*args,**kw);spans['rollout']=time.perf_counter()-start;return result
  def close(self):engine.close()
 e.engine=Capture();old=e._refresh
 def refresh():
  t=time.perf_counter();r=old();spans['refresh']=time.perf_counter()-t;return r
 e._refresh=refresh
 rows=[]
 try:
  cases=[]
  for ti,task in enumerate(meta['task_names']):
   with np.load(bank/(task+'_train.npz')) as z:
    stages=list(np.unique(z['stage']))
    selected=stages if task=='base_side_cover' else [stages[-1]]
    if task=='upper_arm_module':selected=[s for s in stages if 'far_side' in s] or selected
    for stage in selected:
     ids=np.flatnonzero(z['stage']==stage);friction=z['states'][ids,159:162];nom=np.flatnonzero(np.all(friction==1,axis=1));var=np.flatnonzero(np.any(friction!=1,axis=1));ix=[int(ids[nom[0] if len(nom) else 0]),int(ids[var[0] if len(var) else 0])];cases.append((ti,task,str(stage),ix,z['states'][ix].copy()))
  for ti,task,stage,ix,state in cases:
   for rep in range(4):
    t=time.perf_counter();e.set_states(np.arange(2),state,np.full(2,ti,np.int64));reset=time.perf_counter()-t
    action=np.zeros((2,12)) if rep%2==0 else .005*np.sin(np.arange(24).reshape(2,12))
    spans.clear();t=time.perf_counter();cpu=time.process_time();reward,done,info=e.step(action);wall=time.perf_counter()-t;cpu=time.process_time()-cpu
    assert np.isfinite(e.state).all() and np.isfinite(reward).all()
    warnings=[sum(int(w.number) for w in d.warning) for d in e.data];assert not any(warnings),warnings
    row=dict(task=task,stage=stage,bank_rows=ix,rep=rep,warmup=rep==0,action='zero' if rep%2==0 else 'deterministic-small',reset_s=reset,wall_s=wall,cpu_s=cpu,spans=spans.copy(),contacts=[d.ncon for d in e.data],warnings=warnings,unstable=info['unstable'].tolist(),safety=info['safety'].tolist(),friction=state[:,159:162].tolist());rows.append(row);print(json.dumps(row),flush=True)
  write(out/'panel.json',dict(rows=rows,hardware=hardware,timestep=e.m.opt.timestep,substeps=e.substeps,worlds=2,threads=2,npair=e.m.npair,nstate=e.nstate,signature=e.model_signature))
 finally:e.close()
else:
 import torch
 from assembly_policy import train
 torch.set_num_threads(1)
 args=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(bank),'--run-dir',str(out/'ppo'),'--envs','2','--steps','32','--nthread','2','--device','cpu','--iters','1','--micro-batch','1024','--no-eval','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01','--seed','20261107'])
 t=time.perf_counter();tr=train.Trainer(args);init=time.perf_counter()-t
 # Load same finite archived policy and Adam for realistic actions. Diagnostic only:
 # reset SGS/counters under identical seed; never write or resume the source run.
 cp=a.root/'run-v10/checkpoint.before-state-build-repause.pt';d=torch.load(cp,map_location='cpu',weights_only=False);tr.model.load_state_dict(d['model']);tr.opt.load_state_dict(d['opt']);tr.lr=d['lr'];tr.it=d['it'];tr.transitions=d['transitions'];tr.a.iters=tr.it+1;spans=[]
 for name in ['collect','update','save']:
  orig=getattr(tr,name)
  def wrapped(*args,_f=orig,_n=name,**kw):
   t=time.perf_counter();r=_f(*args,**kw);spans.append(dict(name=_n,seconds=time.perf_counter()-t));return r
  setattr(tr,name,wrapped)
 from assembly_policy.wowrobo import env
 realstep=env.BatchedAssemblyEnv.step;steps=[]
 def step(self,action):
  t=time.perf_counter();cpu=time.process_time();r=realstep(self,action);steps.append(dict(seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,valid=len(r[0]),unstable=int(r[2]['unstable'].sum())));return r
 env.BatchedAssemblyEnv.step=step
 try:
  sampler=Sampler(out/'learner.speedscope.json');sampler.start();t=time.perf_counter();cpu=time.process_time();tr.train();elapsed=time.perf_counter()-t;process_cpu=time.process_time()-cpu;sampler.finish()
 finally:tr.env.close()
 rec=json.loads((out/'ppo/log.jsonl').read_text().splitlines()[-1]);write(out/'ppo-result.json',dict(elapsed_s=elapsed,process_cpu_s=process_cpu,initialization_s=init,record=rec,spans=spans,steps=steps,hardware=hardware,source_checkpoint_sha256=sha(cp),source_checkpoint_it=d['it'],source_checkpoint_transitions=d['transitions'],scope='2world/2thread local actual32-step PPO update. Same archived policy+Adam, fresh seeded diagnostic SGS/resets; checkpoint counters retained, CPU learner. Not production fleet or exact continuation.'))
