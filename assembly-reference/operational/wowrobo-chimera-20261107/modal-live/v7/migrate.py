"""Capacity-only exact checkpoint continuation, including every SGS/RNG bit."""
from pathlib import Path
import io,json,hashlib,copy
import numpy as np,torch
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parent.parent;parent=ROOT/'run-v6';RUN=parent
assert (parent/'handoff-closed-v7.json').exists()
assert json.loads((ROOT/'gpu-shape-v7.json').read_text())['status']=='PASS'
archive=parent/'checkpoint.before-capacity-v7.pt';raw=archive.read_bytes();assert (parent/'checkpoint.pt').read_bytes()==raw;old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
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
        assert (RUN/'checkpoint.pt').read_bytes()==raw
        for k in ('model_signature','bank_signature','bank_directory_signature','reward_signature','observation_signature','physics_signature','env'):assert self.args[k]==old['args'][k],k
        self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.sampler.load_state_dict(copy.deepcopy(old['sampler']));self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
        assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt']) and same(self.sampler.state_dict(),old['sampler'])
        self.args['parent_checkpoint']=dict(path=str(archive),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_preserved_exact=True,reason='Human adds4092SAILCPUs alongside4092Modal+retained/greedyChimera; capacity-only5090to9600envs. Same bank/physics/reward/observations,epochs/minibatches/microbatch.')
        self.save()
        result=dict(status='PASS',same_run_continuation=True,capacity_phase=7,parent_it=old['it'],transitions=old['transitions'],policy_preserved_exact=True,optimizer_preserved_exact=True,sampler_and_rng_preserved_exact=True,lr_preserved=True,new_envs=9600,worker_slots=960,target_batch=307200,bank_entries=len(self.bank_S),checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest())
        (ROOT/'migration-v7.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise Done
try:Migration(args)
except Done:pass
