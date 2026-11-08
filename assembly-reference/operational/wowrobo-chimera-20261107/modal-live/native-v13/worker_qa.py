"""Production-sized10env/10thread actual worker parity and peakRSS capture."""
from pathlib import Path
import json,sys,time,threading,subprocess,hashlib
import numpy as np
import torch,io
from assembly_policy.rl.nets import ActorCritic
from assembly_policy.wowrobo.obs import PointTables
torch.set_num_threads(1)
from assembly_policy.rl.remote import RemoteVecEnv
from assembly_policy.wowrobo.env import EnvConfig
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v13';variant=sys.argv[1];assert variant in ('baseline','candidate');RUN=REL/('worker-qa-'+variant);RUN.mkdir(exist_ok=False);tasks=['base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module'];bank=ROOT/('native-v10/training-accepted-v10' if variant=='baseline' else 'native-v13/training-accepted-v13');meta=json.loads((bank/'manifest.json').read_text());S=[];T=[]
for ti,task in enumerate(tasks):
 z=np.load(bank/(task+'_train.npz'))
 for family in (2,4):
  ix=next(i for i in np.flatnonzero(z['family']==family) if np.all(z['states'][i,159:162]==1));s=z['states'][ix].copy();s[159:162]=[.9,1.1,1.] if family==2 else [1.1,.9,1.];S.append(s);T.append(ti)
S=np.array(S);T=np.array(T);cursor=0

def provider(k,finished=None,outcomes=None):
 global cursor
 ix=np.arange(cursor,cursor+k)%10;cursor+=k;return ix.copy(),S[ix].copy(),T[ix].copy()
worker=[]
def launch():
 deadline=time.time()+180
 while not (RUN/'server.json').exists():
  assert time.time()<deadline;time.sleep(.2)
 log=(RUN/'worker.log').open('w');p=subprocess.Popen([sys.executable,'-u',str(REL/'worker_qa_entry.py'),str(RUN)],stdout=log,stderr=subprocess.STDOUT);worker.append(p);log.close()
threading.Thread(target=launch,daemon=True).start();sig=json.loads(((ROOT/'native-v10' if variant=='baseline' else REL)/'prepared.json').read_text())['physics_signature'];cfg=EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=10)
env=RemoteVecEnv(RUN,tasks,10,cfg,provider,1,np.zeros(10,dtype=np.int64),accept_timeout=10,min_frac=1,init_wait=120,model_signature=sig);assert env.alive.all();env._learn_cost=lambda got:None
captures={};wall=time.perf_counter();step_times=[]
if variant=='baseline':
 raw=(ROOT/'run-v10/checkpoint.pt').read_bytes();checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert checkpoint['args']['physics_signature']=='34d224d4edff266369b3f1322baf80013c0daf09599a9eeb8751c6270a88ff92'
 tables=PointTables(tasks,review_model=meta['review_model']);assert tables.signature==checkpoint['args']['observation_signature'];a=checkpoint['args'];policy=ActorCritic(448,proprio_dim=250,act_dim=12,init_std=a['init_std'],max_std=a['max_std'],min_std=a['min_std']);policy.load_state_dict(checkpoint['model']);policy.eval();torch.manual_seed(20261107);commands=[]
else:commands=json.loads((REL/'worker-qa-baseline/commands.json').read_text())

def add(prefix,obj):
 if isinstance(obj,dict):
  for k,v in obj.items():add(prefix+'.'+k,v)
 elif obj is not None:
  a=np.asarray(obj);assert a.dtype.kind!='O';assert np.isfinite(a).all();captures[prefix]=a.copy()
try:
 add('initial',env.observe())
 for k in range(3):
  if variant=='baseline':
   o=env.observe()
   with torch.no_grad():
    pts=tables(torch.tensor(o['task'],dtype=torch.long),torch.from_numpy(o['tcp_p']),torch.from_numpy(o['tcp_R']),torch.from_numpy(o['part_p']),torch.from_numpy(o['part_q']),torch.from_numpy(o['finger']));action=policy.dist(pts,torch.from_numpy(o['proprio'])).sample().numpy()
   commands.append(action.tolist())
  else:action=np.asarray(commands[k],dtype=np.float32)
  start=time.perf_counter();r,done,info,final,finished=env.step(action);step_times.append(time.perf_counter()-start);assert info['valid'].all() and not info['unstable'].any();add(str(k)+'.reward',r);add(str(k)+'.done',done);add(str(k)+'.info',info);add(str(k)+'.obs',env.observe());add(str(k)+'.final',final);add(str(k)+'.finished',finished)
finally:
 env.close()
 deadline=time.time()+30
 while not (RUN/'worker-rss.json').exists() and time.time()<deadline:time.sleep(.2)
 for p in worker:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
if variant=='baseline':(RUN/'commands.json').write_text(json.dumps(commands)+'\n')
rss=json.loads((RUN/'worker-rss.json').read_text());assert rss['envs']==rss['threads']==10 and rss['steps']==3 and rss['model_signature']==sig
np.savez_compressed(RUN/'physical-payloads.npz',**captures);hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in captures.items()};x=dict(status='PASS',variant=variant,arrays=hashes,model_signature=sig,worker=rss,wall_s=time.perf_counter()-wall,step_wall_s=sum(step_times),steps_s=step_times,physics_threads=10,envs=10,controls=3,friction_variation=True,actions='frozen current-policy Gaussian samples replayed exactly in candidate');(RUN/'result.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='arrays'}),flush=True)
