"""One-time transfer milestone, with native substep timing and no reset farming."""
import hashlib
import mujoco
import numpy as np
import pytest
from assembly_policy.cube_trial import core
from assembly_policy.cube_trial.env import BatchedCubeEnv, EnvConfig


def contacts(left=(0., 0.), right=(1., 1.), ground=False):
    return dict(pad_loads_n=np.array([*left, *right]), right_held=min(right)>.01, table_touching=ground)


@pytest.fixture
def env():
    e=BatchedCubeEnv(['cube_handoff'],[0],EnvConfig(nthread=1,cost_coef=0))
    yield e
    e.close()


def prime(e):
    e.carry_started[0]=True
    e._handoff_sample(0,contacts(left=(1.,1.),right=(0.,0.)))


def hold(e,c,n=60):
    for _ in range(n): e._handoff_sample(0,c)


def test_bonus_requires_full_120ms_and_is_once_per_episode(env):
    assert core.HANDOFF_REWARD==.5
    prime(env)
    hold(env,contacts(),59)
    assert not env.handoff_pending[0] and not env.handoff_paid[0]
    hold(env,contacts(),1)
    assert env.handoff_pending[0] and env.handoff_paid[0]
    env.handoff_pending[:]=False
    hold(env,contacts(),100)
    hold(env,contacts(left=(2.,2.)),10)
    hold(env,contacts(),100)
    assert not env.handoff_pending[0]


@pytest.mark.parametrize('invalid',[contacts(right=(1.,0.)), contacts(left=(.02,0)), contacts(ground=True)])
def test_incomplete_grasp_left_support_or_table_contact_cannot_earn(env,invalid):
    prime(env)
    hold(env,invalid,100)
    assert not env.handoff_paid[0]
    hold(env,contacts(),59)
    assert not env.handoff_paid[0]
    hold(env,invalid,1)  # even a single native-step interruption restarts duration
    hold(env,contacts(),59)
    assert not env.handoff_paid[0]


def test_requires_prior_left_grasp_and_valid_carry(env):
    env.carry_started[0]=True
    hold(env,contacts())
    assert not env.handoff_paid[0]
    prime(env)
    env.off_target_table_contact[0]=True
    hold(env,contacts())
    assert not env.handoff_paid[0]


@pytest.mark.parametrize('family,consumed', [('pre_handoff',False),('dual_grasp',False),('post_handoff',True),('right_transport',True),('near_target',True),('placed_held',True)])
def test_native_reset_consumption_and_no_bonus_for_starting_after_transfer(env,family,consumed):
    d,_=core.generate(family,0,env.m)
    state=np.empty(env.nstate);mujoco.mj_getState(env.m,d,state,core.STATE_SPEC)
    # Production reset banks supply authoritative family identity by state hash.
    env.reset_handoff_consumed[hashlib.sha256(state.tobytes()).hexdigest()]=consumed
    env.handoff_pending[0]=True;env.handoff_hold_s[0]=1
    env.set_states([0],state[None])
    assert bool(env.handoff_paid[0])==consumed
    assert not env.handoff_pending[0] and env.handoff_hold_s[0]==0
    if consumed:
        reward,_,_=env.step(np.zeros((1,12)))
        assert reward[0]==0


def test_real_native_right_grasp_can_earn_half_without_ending_episode(env):
    d,_=core.generate('post_handoff',0,env.m)
    state=np.empty(env.nstate);mujoco.mj_getState(env.m,d,state,core.STATE_SPEC)
    env.set_states([0],state[None])
    # Reconstruct an already-seen left grasp before a real right-only hold.
    env.handoff_paid[0]=False;prime(env)
    rewards=[]
    for _ in range(12):
        r,done,_=env.step(np.zeros((1,12)));rewards.append(float(r[0]))
        assert not done[0]
    assert sum(rewards)==.5 and rewards.count(.5)==1


def test_actual_bank_families_mark_only_post_transfer_starts_consumed():
    from pathlib import Path
    directory=Path(__file__).resolve().parents[1]/'data/cube_trial/banks_handoff_bridge_v8'
    if not directory.exists():
        pytest.skip('Qualified bridge bank is not installed')
    e=BatchedCubeEnv(['cube_handoff'],[0],EnvConfig(nthread=1,reset_context_path=str(directory/'reset-context.npz')))
    try:
        for split in ('train','eval'):
            with np.load(directory/f'cube_handoff_{split}.npz') as b:
                for state,family in zip(b['states'],b['family']):
                    expected=core.FAMILIES[int(family)] in ('post_handoff','right_transport','near_target','placed_held')
                    assert e.reset_handoff_consumed[hashlib.sha256(state.tobytes()).hexdigest()]==expected
    finally:
        e.close()
