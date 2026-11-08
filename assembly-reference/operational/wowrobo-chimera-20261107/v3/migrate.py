"""Explicit versioned continuation: retain policy/optimizer, reset SGS for new entries."""
from pathlib import Path
import io,json,hashlib,time
import torch,numpy as np
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent;parent=Path(json.loads((ROOT/'parent-run.json').read_text())['run']);RUN=ROOT/'run-v3'
assert (parent/'handoff-closed.json').exists(),'Require positive old learner closure'
raw=(parent/'checkpoint.pt').read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert old['it']>=1
cmd=json.loads((parent/'command.json').read_text());cmd=[s.replace(str(parent.parents[0]),str(ROOT)) for s in cmd];cmd[cmd.index('--bank-dir')+1]=str(ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-minimum-24-v1');cmd[cmd.index('--run-dir')+1]=str(RUN);cmd[cmd.index('--device')+1]='cpu';args=parse(cmd[3:]);assert not (RUN/'checkpoint.pt').exists()
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
class Done(Exception):pass
class Migration(Trainer):
 def _maybe_resume(self):
  assert not (self.run/'checkpoint.pt').exists()
  assert self.tasks==list(self.backend.TASK_NAMES)
  self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
  assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt'])
  sampler=self.sampler.state_dict()
  self.args['parent_checkpoint']=dict(path=str(parent/'checkpoint.pt'),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_reset=True,reason='Human requests24passed minimum-valid states with matched centered-housing physical model. New bank identities; old outcome histories do not map to new entries.')
  self.save();(self.run/'config.migration.json').write_text(json.dumps(self.args,indent=2)+'\n');(self.run/'parent-checkpoint.pt').write_bytes(raw)
  result=dict(status='PASS',old_run=str(parent),new_run=str(RUN),parent_it=old['it'],transitions=old['transitions'],old_model=old['args']['model_signature'],new_model=self.args['model_signature'],old_bank_entries=None,new_bank_entries=len(self.bank_S),policy_preserved_exact=True,optimizer_preserved_exact=True,learning_rate_preserved=True,new_sampler_fresh=True,observation_signature_changed=old['args']['observation_signature']!=self.args['observation_signature'],reward_definition='same0.5mm/1degree/0.4sstable seating, centered reference frames',checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest())
  (ROOT/'migration-v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise Done
try:Migration(args)
except Done:pass
