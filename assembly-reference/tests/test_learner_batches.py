"""Exercise the real PPO update's handling of partial/lost-worker batches."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from assembly_policy.train import Trainer,OBS
from assembly_policy.rl.nets import ActorCritic


def case(valid,minibatches):
    torch.manual_seed(17)
    n=len(valid);tr=Trainer.__new__(Trainer)
    tr.a=SimpleNamespace(minibatches=minibatches,epochs=1,micro_batch=4,clip=.2,vf_coef=1.,ent=.005,desired_kl=.01,max_grad=1.)
    tr.dev=torch.device('cpu');tr.lr=1e-4
    tr.model=ActorCritic(2,proprio_dim=4,act_dim=2)
    tr.opt=torch.optim.Adam(tr.model.parameters(),lr=tr.lr)
    seen=[]
    def points(ob):
        seen.extend(ob['task'].tolist())
        return torch.zeros(len(ob['task']),6)
    tr._points=points
    ob=dict(prop=torch.zeros(1,n,4),task=torch.arange(n).reshape(1,n),tcp_p=torch.zeros(1,n,3),tcp_R=torch.zeros(1,n,3,3),part_p=torch.zeros(1,n,3),part_q=torch.zeros(1,n,4),finger=torch.zeros(1,n,2))
    with torch.no_grad():
        dist=tr.model.dist(torch.zeros(n,6),ob['prop'][0]);act=dist.sample();logp=dist.log_prob(act).sum(-1);v=tr.model.value(torch.zeros(n,6),ob['prop'][0])
    batch=dict(ob,act=act[None],logp=logp[None],val=v[None],mu=dist.mean[None],rew=torch.zeros(1,n),ret=(v+.2)[None],adv=torch.linspace(-1,1,n)[None],valid=torch.tensor([valid]))
    return tr,batch,seen


def test_every_valid_transition_is_used_once_with_a_minibatch_remainder():
    tr,batch,seen=case([True]*7,4)
    result=tr.update(batch)
    assert sorted(seen)==list(range(7))
    assert all(np.isfinite(v) for v in result.values())


def test_lost_transitions_are_excluded_without_dropping_valid_remainders():
    tr,batch,seen=case([True,False,True,True,False,True,True],4)
    tr.update(batch)
    assert sorted(seen)==[0,2,3,5,6]


def test_single_valid_transition_is_finite_even_with_more_requested_minibatches():
    tr,batch,seen=case([False,True,False],4)
    result=tr.update(batch)
    assert seen==[1]
    assert all(np.isfinite(v) for v in result.values())
    assert all(torch.isfinite(p).all() for p in tr.model.parameters())


def test_all_lost_transitions_fail_before_an_optimizer_step():
    tr,batch,seen=case([False,False],4)
    before={k:v.clone() for k,v in tr.model.state_dict().items()}
    with pytest.raises(ValueError,match='No valid transitions'):tr.update(batch)
    assert not seen
    assert all(torch.equal(before[k],v) for k,v in tr.model.state_dict().items())
