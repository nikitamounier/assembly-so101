"""Real SO101 PPO/CUDA smoke test, with learning-state and resume assertions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.workcell import default_bank_directory
from assembly_policy.train import Trainer,parse
from assembly_policy.rl.sampler import ResetSampler


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


class AuditedTrainer(Trainer):
    def _maybe_resume(self):
        super()._maybe_resume()
        self.restored=copy.deepcopy(dict(model=self.model.state_dict(),opt=self.opt.state_dict(),sampler=self.sampler.state_dict(),it=self.it,transitions=self.transitions,lr=self.lr))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank-dir',type=Path,default=default_bank_directory())
    p.add_argument('--device',default='cuda')
    p.add_argument('--envs',type=int,default=8)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,scope='Learner software/restore smoke test; not assembly success or exact simulator continuation',checks={},runtime=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda))
    active=None;hooks=[];start=time.monotonic()
    def write():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; install the pinned CUDA wheel/driver or explicitly select --device cpu')
        if a.device.startswith('cuda'):report['runtime']['gpu']=torch.cuda.get_device_name()
        args=['--suite','so101','--tasks','all','--bank-dir',str(a.bank_dir),'--run-dir',str(out/'run'),'--device',a.device,'--envs',str(a.envs),'--steps',str(a.steps),'--nthread',str(a.threads),'--epochs','2','--minibatches','4','--iters','2','--save-every','1','--keep-every','1000000','--no-eval','--so101-max-contact-depth',str(PENETRATION_LIMIT_M),'--so101-stability','pose','--so101-safety-penalty','1.0','--seed','31']
        # Same explicit small-bank sampler floor as the reviewed pilot.
        count=sum(len(np.load(a.bank_dir/f'{name}_train.npz')['states']) for name in ('motor','horn','screw','retainer','yoke'))
        args+=['--eps-q',str(1.6/count)]
        active=AuditedTrainer(parse(args));active.stop_requested=False
        before={k:v.detach().clone() for k,v in active.model.named_parameters()}
        grads={'actor':0,'critic':0,'nonfinite':0}
        def capture(name):
            def observe(gradient):
                if not torch.isfinite(gradient).all():grads['nonfinite']+=1
                if gradient.abs().max()>0:
                    for branch in ('actor','critic'):
                        if name.startswith(branch+'.'):grads[branch]+=1
                return gradient
            return observe
        for name,param in active.model.named_parameters():hooks.append(param.register_hook(capture(name)))
        active.train()
        rows=[json.loads(line) for line in (out/'run/log.jsonl').read_text().splitlines()]
        report['rows']=rows
        checks=report['checks']
        checks['finite_losses']=all(all(math.isfinite(row[k]) for k in ('pi','vf','ent','kl','std','lr','fps')) for row in rows)
        checks['expected_transitions']=active.transitions==2*a.envs*a.steps
        checks['actor_and_critic_gradients']=grads['actor']>0 and grads['critic']>0 and grads['nonfinite']==0
        checks['actor_and_critic_weights_changed']=all(any(not torch.equal(before[name],param) for name,param in active.model.named_parameters() if name.startswith(branch+'.')) for branch in ('actor','critic'))
        checks['normalization_updated']=float(active.model.norm.count)>1.
        checks['optimizer_state_populated']=bool(active.opt.state)
        initial=torch.load(out/'run/initial_policy.pt',map_location=a.device,weights_only=False)
        checks['baseline_weights_unupdated']=all(torch.equal(initial['model'][name],value) for name,value in before.items())
        checks['baseline_normalization_fitted']=float(initial['model']['norm.count'])>1.
        sampler=ResetSampler(3,eps=.01,history=3,seed=21);sampler.update([0,0,1],[0,1,1]);state=copy.deepcopy(sampler.state_dict());draw=sampler.sample(100);sampler.load_state_dict(state)
        checks['sgs_outcomes_weights_rng']=np.allclose(sampler.p_hat(),[.5,1.,0.]) and sampler.weights()[0]>sampler.weights()[1] and np.array_equal(draw,sampler.sample(100))
        saved=torch.load(out/'run/checkpoint.pt',map_location=a.device,weights_only=False)
        checks['atomic_checkpoint_contains_sgs']='sampler' in saved
        active.env.close()
        for handle in hooks:handle.remove()
        hooks.clear();active=None
        # Deliberately remove the legacy sidecar: an atomic checkpoint must suffice.
        (out/'run/sampler.pt').rename(out/'run/sampler.legacy.pt')
        active=AuditedTrainer(parse(args));active.stop_requested=False
        checks['exact_saved_learning_state_restored']=all(same(active.restored[k],saved[k]) for k in ('model','opt','sampler','it','transitions','lr'))
        active.a.iters=3;active.train()
        checks['resume_advances_counters']=active.it==3 and active.transitions==3*a.envs*a.steps
        checks['resumed_weights_finite']=all(bool(torch.isfinite(p).all()) for p in active.model.parameters())
        report['task_outcomes']=dict(meaning='Diagnostics only; zero assembly successes is allowed for this software check',episodes=sum(r['episodes'] for r in rows),safety_rates=[r['safety'] for r in rows],success_rates=[r['success'] for r in rows])
        report['passed']=all(checks.values())
        if not report['passed']:raise AssertionError('Failed learner checks: '+', '.join(k for k,v in checks.items() if not v))
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if active is not None:active.env.close()
        for handle in hooks:handle.remove()
        report['elapsed_s']=time.monotonic()-start
        report['sources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('assembly_policy/train.py','assembly_policy/rl/nets.py','assembly_policy/rl/sampler.py','scripts/cloud/healthcheck.py')}
        write()
    print(json.dumps(dict(passed=report['passed'],report=str(out/'report.json'),checks=report['checks'])),flush=True)


if __name__=='__main__':main()
