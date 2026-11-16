"""Validate a frozen checkpoint and real Trainer resume without constructing a second environment."""
from pathlib import Path
import io,json,hashlib,sys,time
import numpy as np,torch
from assembly_policy.train import Trainer,parse
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'run-v2'
raw=(RUN/'checkpoint.pt').read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert saved['it']>=1 and saved['transitions']>0
assert saved['args']['run_status']=='DIAGNOSTIC_UNQUALIFIED_RESETS'
assert all(torch.isfinite(t).all() for t in saved['model'].values())
assert any(float(v.get('step',0))>0 and bool(v['exp_avg'].abs().max()>0) for v in saved['opt']['state'].values())
snapshot=ROOT/('resume-validation-'+str(time.time_ns()));snapshot.mkdir();(snapshot/'checkpoint.pt').write_bytes(raw)
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
class Done(Exception):pass
holder=[]
class Probe(Trainer):
 def _maybe_resume(self):
  original=self.run;self.run=snapshot
  try:super()._maybe_resume()
  finally:self.run=original
  holder.append(self);raise Done
args=parse(json.loads((RUN/'command.json').read_text())[3:]);args.device='cpu'
try:Probe(args)
except Done:pass
probe=holder[0]
assert probe.it==saved['it'] and probe.transitions==saved['transitions']
assert same(probe.model.state_dict(),saved['model'])
assert same(probe.opt.state_dict(),saved['opt'])
assert same(probe.sampler.state_dict(),saved['sampler'])
result=dict(status='PASS',it=probe.it,transitions=probe.transitions,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_bytes=len(raw),finite_model=True,nonzero_optimizer_gradients=True,model_restore_exact=True,optimizer_restore_exact=True,sampler_restore_exact=True,actual_trainer_resume_checks_passed=True,stopped_before_environment_construction=True,snapshot=str(snapshot),model_signature=saved['args']['model_signature'])
(snapshot/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'checkpoint-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
