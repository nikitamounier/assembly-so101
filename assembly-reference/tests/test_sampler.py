import numpy as np
from assembly_policy.rl.sampler import ResetSampler


def test_success_history_wrap_and_rng_restore():
    s=ResetSampler(2,eps=.01,history=3,seed=9)
    s.update([0,0,0,0,0],[0,1,0,1,1])
    assert s.cnt[0]==5 and s.sum[0]==2 and s.p_hat()[0]==2/3
    t=ResetSampler(2,eps=.01,history=3,seed=8);t.load_state_dict(s.state_dict())
    assert np.array_equal(s.sample(100),t.sample(100))


def test_unvisited_entries_keep_positive_sampling_probability():
    s=ResetSampler(5,eps=.01,history=3,seed=5);s.update([0,0],[0,1])
    assert np.all(s.weights()>0)
    assert set(s.sample(10000))==set(range(5))
