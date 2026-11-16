"""Explicit native→approximate physics continuation; no claimed native parity."""
from pathlib import Path
import io,json,hashlib,copy
import numpy as np,torch
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parent.parent;parent=ROOT/'run-v6';RUN=ROOT/'run-v8';REL=ROOT/'bare-v8'
assert json.loads((REL/'distributed-smoke.json').read_text())['status']=='PASS'
assert json.loads((parent/'handoff-closed-v8.json').read_text())['status']=='PASS'
archive=parent/'checkpoint.before-bare-v8.pt';raw=archive.read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert not (RUN/'checkpoint.pt').exists()
cmd=json.loads((parent/'command.json').read_text());cmd=cmd[cmd.index('--suite'):]
for k,v in {'--run-dir':RUN,'--device':'cpu','--envs':9600,'--remote-workers':960,'--min-conn-frac':.025}.items():
 if k in cmd:cmd[cmd.index(k)+1]=str(v)
 else:cmd.extend([k,str(v)])
args=parse(cmd);assert args.steps==32 and args.epochs==5 and args.minibatches==4 and args.micro_batch==1024
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
class Done(Exception):pass
class Migration(Trainer):
 def _maybe_resume(self):
  for k in ('bank_signature','bank_directory_signature','observation_signature','env'):assert self.args[k]==old['args'][k],k
  assert self.args['physics_backend']=='bare-mujoco-boxes-v8'
  assert self.args['model_signature']==json.loads((REL/'scene.json').read_text())['scene_sha256']
  assert old['args']['model_signature']=='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879'
  self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.sampler.load_state_dict(copy.deepcopy(old['sampler']));self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
  assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt']) and same(self.sampler.state_dict(),old['sampler'])
  self.args.update(penetration_termination=False,velocity_termination=False,inter_part_collision=False,robot_self_collision=False,physics_approximate=True)
  self.args['parent_checkpoint']=dict(path=str(archive),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],physics_backend=old['args']['physics_backend'],physics_changed=True,policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_preserved_exact=True,reason='Human explicitly requests fast bare MuJoCo same assembly task; versioned approximate physics continuation, not native-contact certification.')
  self.save()
  result=dict(status='PASS',run=str(RUN),physics_changed=True,physics_backend=self.args['physics_backend'],physics_signature=self.args['physics_signature'],parent_it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_and_rng_preserved_exact=True,new_envs=9600,worker_slots=960,target_batch=307200,bank_entries=len(self.bank_S),checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest());(REL/'migration.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise Done
try:Migration(args)
except Done:pass
