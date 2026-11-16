"""Explicit policy/Adam continuation into the larger admitted bank and rollout batch."""
from pathlib import Path
import io,json,hashlib
import torch,numpy as np
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parents[4];parent=Path(json.loads((ROOT/'parent-run.json').read_text())['run']);RUN=ROOT/'run-v5'
assert (parent/'handoff-closed.json').exists()
assert json.loads((ROOT/'reset-validation-v5.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'gpu-shape-v4.json').read_text())['status']=='PASS'
raw=(parent/'checkpoint.pt').read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
cmd=json.loads((parent/'command.json').read_text());cmd=[s.replace(str(parent.parent),str(ROOT)) for s in cmd]
def option(name,value):
 if name in cmd:cmd[cmd.index(name)+1]=str(value)
 else:cmd.extend([name,str(value)])
for k,v in {'--bank-dir':ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5','--run-dir':RUN,'--device':'cpu','--envs':5090,'--remote-workers':509,'--micro-batch':1024}.items():option(k,v)
args=parse(cmd[3:]);assert args.steps==32 and args.epochs==5 and args.minibatches==4
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
class Done(Exception):pass
class Migration(Trainer):
 def _maybe_resume(self):
  assert not (self.run/'checkpoint.pt').exists()
  self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
  assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt'])
  self.args['parent_checkpoint']=dict(path=str(parent/'checkpoint.pt'),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_reset=True,reason='User requests accepted scaling states and4092ModalCPU plus retained/greedy Chimera. Expanded admitted bank and per-episode friction are versioned; new sampling history required.')
  self.save();(self.run/'parent-checkpoint.pt').write_bytes(raw)
  result=dict(status='PASS',parent_run=str(parent),new_run=str(RUN),parent_it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,learning_rate_preserved=True,new_sampler_entries=len(self.bank_S),new_model=self.args['model_signature'],new_envs=5090,steps=32,target_batch=162880,ppo_epochs=5,ppo_minibatches=4,micro_batch=1024,checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest())
  (ROOT/'migration-v5.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise Done
try:Migration(args)
except Done:pass
