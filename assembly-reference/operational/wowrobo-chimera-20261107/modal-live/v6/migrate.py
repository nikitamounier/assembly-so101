"""Exact policy/Adam continuation with SGS histories mapped by unchanged state bytes."""
from pathlib import Path
import io,json,hashlib,copy
import torch,numpy as np
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parent.parent;parent=ROOT/'run-v5';RUN=ROOT/'run-v6'
assert (parent/'handoff-closed-v6.json').exists()
assert json.loads((ROOT/'reset-validation-v6.json').read_text())['status']=='PASS'
raw=(parent/'checkpoint.pt').read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
cmd=json.loads((parent/'command.json').read_text());cmd=cmd[cmd.index('--suite'):]
def option(k,v):
 if k in cmd:cmd[cmd.index(k)+1]=str(v)
 else:cmd.extend([k,str(v)])
for k,v in {'--bank-dir':ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6','--run-dir':RUN,'--device':'cpu','--bank-per-family':0}.items():option(k,v)
args=parse(cmd);assert args.envs==5090 and args.steps==32 and args.epochs==5 and args.minibatches==4
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
def bank_keys(bank,tasks):
 keys=[]
 for task in tasks:
  with np.load(bank/(task+'_train.npz')) as z:
   keys.extend((task,int(f),hashlib.sha256(s.tobytes()).hexdigest()) for s,f in zip(z['states'],z['family']))
 assert len(keys)==len(set(keys));return keys
class Done(Exception):pass
class Migration(Trainer):
 def _maybe_resume(self):
  assert not (self.run/'checkpoint.pt').exists()
  for k in ('model_signature','observation_signature','reward_signature','physics_signature'):assert self.args[k]==old['args'][k],k
  self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
  assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt'])
  oldkeys=bank_keys(ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5',self.tasks);newkeys=bank_keys(Path(args.bank_dir),self.tasks);index={k:i for i,k in enumerate(oldkeys)}
  assert len(oldkeys)==len(old['sampler']['cnt']);mapped=0
  for i,k in enumerate(newkeys):
   if k not in index:continue
   j=index[k];mapped+=1
   for f in ('buf','ptr','cnt','sum'):getattr(self.sampler,f)[i]=old['sampler'][f][j]
  self.sampler.rng.bit_generator.state=copy.deepcopy(old['sampler']['rng']);self.sampler._cdf=None
  self.args['parent_checkpoint']=dict(path=str(parent/'checkpoint.pt'),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_rows_preserved=mapped,sampler_new_rows=len(newkeys)-mapped,reason='Latest accepted TRAIN state union and bounded passive episode recorder; EVAL excluded; same physical model and PPO hyperparameters.')
  self.save();(self.run/'parent-checkpoint.pt').write_bytes(raw)
  result=dict(status='PASS',parent_it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,learning_rate_preserved=True,sampler_rows_preserved=mapped,new_sampler_entries=len(newkeys)-mapped,total_entries=len(newkeys),envs=5090,target_batch=162880,checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest())
  (ROOT/'migration-v6.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise Done
try:Migration(args)
except Done:pass
