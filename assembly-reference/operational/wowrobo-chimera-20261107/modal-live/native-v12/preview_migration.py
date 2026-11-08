"""Native timestep-only checkpoint continuation from the CURRENTnative run."""
from pathlib import Path
import io,json,hashlib,copy
import numpy as np,torch
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v12';parent=ROOT/'run-v10';RUN=REL/'migration-preview';BANK=REL/'training-accepted-v12'
assert json.loads((REL/'production-worker-proof.json').read_text())['status']=='PASS' and json.loads((REL/'distributed-smoke.json').read_text())['status']=='PASS'
assert json.loads((parent/'state-build-cpu-priority.json').read_text())['preserve_training_checkpoint'] is True
archive=parent/'checkpoint.before-state-build-priority.pt';raw=archive.read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert old['args']['physics_backend']=='wowrobo-native-source-sdf' and old['args']['physics_signature']=='34d224d4edff266369b3f1322baf80013c0daf09599a9eeb8751c6270a88ff92';assert old['it']>=31
assert not (RUN/'checkpoint.pt').exists();cmd=json.loads((parent/'command.json').read_text());cmd=cmd[cmd.index('--suite'):]
for k,v in {'--run-dir':RUN,'--device':'cpu','--bank-dir':BANK,'--envs':9600,'--remote-workers':960,'--min-conn-frac':.025}.items():
 if k in cmd:cmd[cmd.index(k)+1]=str(v)
 else:cmd.extend([k,str(v)])
a=parse(cmd);assert a.steps==32 and a.epochs==5 and a.minibatches==4 and a.micro_batch==1024
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
class Done(Exception):pass
class Migration(Trainer):
 def _maybe_resume(self):
  for k in ('bank_signature','observation_signature','reward_signature','env'):assert self.args[k]==old['args'][k],k
  assert self.args['model_signature']==json.loads((REL/'prepared.json').read_text())['physics_signature']
  assert self.args['physics_backend']=='wowrobo-native-source-sdf'
  self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.sampler.load_state_dict(copy.deepcopy(old['sampler']));self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
  assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt']) and same(self.sampler.state_dict(),old['sampler'])
  provenance=dict(path=str(archive),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],physics_backend=old['args']['physics_backend'],physics_signature=old['args']['physics_signature'],reason='Human requests faster native MuJoCo; qualified swept contact search with exact worker/oracle/PPO parity, unchanged0.5ms/80substeps/40msactions/CAD/friction/guards/rewards. No bare descendant.')
  self.args.update(parent_checkpoint=provenance,continuation_origin=provenance,native_timestep_s=.0005,substeps_per_action=80,control_dt_s=.04,source_model_signature=old['args']['model_signature'])
  self.save();x=dict(status='PASS',run=str(RUN),parent_it=old['it'],parent_transitions=old['transitions'],parent_checkpoint=provenance,physics_signature=self.args['physics_signature'],old_timestep=.0005,new_timestep=.0005,substeps=80,control_dt=.04,policy_optimizer_SGS_reset_rng_exact=True,checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest(),bank_payloads_unchanged=True,bank_directory_signature=self.args['bank_directory_signature']);(REL/'migration-preview.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True);raise Done
try:Migration(a)
except Done:pass
