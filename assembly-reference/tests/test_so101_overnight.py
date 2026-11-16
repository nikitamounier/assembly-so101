"""Low-cost launch guard tests; no MuJoCo scene or learner starts."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('overnight',ROOT/'scripts/so101_overnight.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def bank_layout(folder,count=1):
    (folder/'family_eval').mkdir()
    for task in ('motor','horn','screw','retainer','yoke'):
        for split in ('train','eval'):
            path=folder/('family_eval' if split=='eval' else '')/f'{task}_{split}.npz'
            np.savez(path,states=np.zeros((5*count,2)),family=np.repeat(np.arange(5),count))


def test_bank_layout_requires_every_cell_in_both_splits(tmp_path):
    bank_layout(tmp_path)
    launcher.qualify_bank_layout(tmp_path,1,1)
    bad=tmp_path/'family_eval/yoke_eval.npz'
    np.savez(bad,states=np.zeros((5,2)),family=np.array([0,0,1,2,3]))
    with pytest.raises(ValueError,match='unbalanced'):
        launcher.qualify_bank_layout(tmp_path,1,1)


def test_default_full_bank_cannot_accept_pilot(tmp_path):
    bank_layout(tmp_path)
    with pytest.raises(ValueError,match='unbalanced'):
        launcher.qualify_bank_layout(tmp_path,256,64)


@pytest.mark.parametrize('value',['2026-11-07T12:00:00','2026-11-07T12:00:00+01:00'])
def test_explicit_utc_required(value):
    with pytest.raises(ValueError,match='UTC'):
        launcher.parse_utc(value)


def test_evaluation_checks_loose_and_every_family_with_one_shared_budget(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher,'run_bounded',lambda command,environment,log,deadline,stop_file:calls.append((command,deadline)))
    args=argparse.Namespace(bank_dir=tmp_path,eval_timeout=20,eval_trials=2)
    launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',tmp_path/'eval',args,{})
    assert [cmd[cmd.index('--families')+1] for cmd,_ in calls]==['loose','all']
    assert all(cmd[cmd.index('--bank-dir')+1]==str(tmp_path) for cmd,_ in calls)
    assert all(cmd[cmd.index('--workers')+1]=='1' for cmd,_ in calls)
    assert all(cmd[cmd.index('--nthread')+1]=='1' for cmd,_ in calls)
    assert all(launcher.time.time()<deadline<=launcher.time.time()+20 for _,deadline in calls)


def test_evaluation_failure_is_persisted_and_raised(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise subprocess.TimeoutExpired('evaluate',1)
    monkeypatch.setattr(launcher,'run_bounded',fail)
    settings=argparse.Namespace(bank_dir=tmp_path,eval_timeout=1,eval_trials=1)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',tmp_path/'eval',settings,{})
    assert 'TimeoutExpired' in json.loads((tmp_path/'eval/error.json').read_text())['error']


def test_local_profile_keeps_approved_gates_and_bounded_resources():
    cfg=json.loads((ROOT/'configs/so101_local.json').read_text())
    assert (cfg['physics'],cfg['device'])==('native','cuda')
    assert cfg['contact_depth_m']==0.0005 and cfg['safety_penalty']==1
    assert cfg['envs']==8 and cfg['threads']==8
    assert cfg['iterations']==20 and cfg['max_hours']==2
    assert cfg['allow_task_mechanics_failure'] is False
    assert cfg['review_seconds']==7200


def test_run_bounded_rejects_stop_and_expired_deadline_before_spawn(tmp_path,monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('Must not spawn')
    monkeypatch.setattr(launcher.subprocess,'Popen',unexpected)
    stop=tmp_path/'STOP';stop.touch()
    with pytest.raises(InterruptedError):
        launcher.run_bounded(['unused'],{},None,launcher.time.time()+10,stop)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.run_bounded(['unused'],{},None,launcher.time.time()-1)


def test_evaluation_hard_end_limits_its_own_timeout(tmp_path,monkeypatch):
    deadlines=[]
    monkeypatch.setattr(launcher,'run_bounded',lambda command,environment,log,deadline,stop_file:deadlines.append(deadline))
    hard_end=launcher.time.time()+5
    settings=argparse.Namespace(bank_dir=tmp_path,eval_timeout=1800,eval_trials=1,deadline=hard_end)
    launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',tmp_path/'eval',settings,{})
    assert deadlines==[hard_end,hard_end]


@pytest.mark.parametrize('periodic_review',[False,True])
def test_supervisor_clean_completion_does_not_restart_and_evaluates_both_ends(tmp_path,monkeypatch,periodic_review):
    from datetime import datetime,timedelta,timezone
    import sys
    import types
    bank=tmp_path/'bank';bank.mkdir();bank_layout(bank)
    preflight=tmp_path/'preflight.json'
    preflight.write_text(json.dumps(dict(model_signature='model',bank_directory_signature='bank',
        max_contact_depth=0.0005,stability_metric='pose',safety_penalty=1.,
        physics_backend='native',physics_signature='model',checks={'thread_tracks_pitch':True})))
    resets=types.ModuleType('assembly_policy.so101.resets')
    resets.signature=lambda:'model'
    resets.check_bank=lambda path:None
    resets.directory_signature=lambda path:'bank'
    monkeypatch.setitem(sys.modules,'assembly_policy.so101.resets',resets)
    monkeypatch.setattr(launcher,'require_headroom',lambda run:None)
    monkeypatch.setattr(launcher,'snapshot',lambda run:dict(latest=None,available_memory_gb=8,disk_free_gb=4))
    monkeypatch.setattr(launcher.time,'sleep',lambda seconds:None)
    spawned=[];initialized=[];evaluated=[]
    class Process:
        pid=12345
        returncode=0
        def poll(self):return 0
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda command,**kwargs:spawned.append(command) or Process())
    def initialize(command,*args):
        initialized.append(command)
        run=Path(command[command.index('--run-dir')+1]);(run/'checkpoint.pt').touch()
        (run/'baseline_policy.pt').touch()
        (run/'baseline_calibration.json').write_text(json.dumps({'optimizer_updates':0,'calibration_transitions':256}))
    monkeypatch.setattr(launcher,'run_bounded',initialize)
    monkeypatch.setattr(launcher,'evaluate_checkpoint',lambda checkpoint,folder,*args:evaluated.append((folder.name,checkpoint.name)))
    end=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    run=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['supervisor','--run-dir',str(run),'--bank-dir',str(bank),
        '--bank-signature','bank','--preflight',str(preflight),'--end-utc',end,
        '--train-per-family','1','--eval-per-family','1'])
    if periodic_review:
        sys.argv+=['--first-review-utc',(datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat()]
    launcher.main()
    assert len(spawned)==(2 if periodic_review else 1) and len(initialized)==1
    if periodic_review:
        worker=spawned[1]
        assert worker[worker.index('--evaluation-workers')+1]=='8'
        assert '--evaluate-snapshot' in worker
    assert evaluated==[('initial','baseline_policy.pt'),('final','checkpoint.pt')]
    assert spawned[0][spawned[0].index('--tasks')+1]=='all'
    assert spawned[0][spawned[0].index('--iters')+1]=='20'
    assert json.loads((run/'supervisor.json').read_text())['state']=='completed'


@pytest.mark.parametrize('fault',[None,'weights','nonfinite','unstable'])
def test_baseline_calibrates_before_save_without_optimizer_and_retains_policy(tmp_path,fault):
    import torch
    from types import SimpleNamespace
    from assembly_policy.rl.nets import RunningNorm
    calls=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Linear(2,1);self.norm=RunningNorm(2)
    model=Model()
    before={k:v.clone() for k,v in model.actor.state_dict().items()}
    class FakeTrainer:
        def __init__(self,args):
            self.run=tmp_path;self.it=0;self.transitions=0;self.model=model
            self.args={'bank_directory_signature':'bank'}
            self.opt=SimpleNamespace(state={})
            self.env=SimpleNamespace(close=lambda:calls.append('close'))
        def collect(self):
            calls.append('collect');self.model.norm.update(torch.ones((2,2)));self.transitions+=2
            if fault=='weights':
                with torch.no_grad():self.model.actor.weight.add_(1)
            return {'rew':torch.tensor([float('nan') if fault=='nonfinite' else 0.])},{'unstable':int(fault=='unstable')}
        def save(self):
            calls.append('save')
            assert float(self.model.norm.count)>2
            torch.save({'it':self.it,'transitions':self.transitions},self.run/'checkpoint.pt')
    if fault:
        with pytest.raises((ValueError,FloatingPointError)):
            launcher.initialize_baseline(['--run-dir',str(tmp_path)],FakeTrainer)
        assert 'save' not in calls and calls[-1]=='close'
        assert not (tmp_path/'baseline_policy.pt').exists()
        return
    report=launcher.initialize_baseline(['--run-dir',str(tmp_path)],FakeTrainer)
    assert calls==['collect','save','close']
    assert report['optimizer_updates']==0 and report['calibration_transitions']==2
    policy_path=tmp_path/'baseline_policy.pt';policy_bytes=policy_path.read_bytes()
    policy=torch.load(policy_path,weights_only=False)
    assert policy['it']==0 and policy['transitions']==2
    assert float(policy['model']['norm.count'])>2
    assert all(torch.equal(before[k],policy['model']['actor.'+k]) for k in before)
    (tmp_path/'checkpoint.pt').write_bytes(b'rolling checkpoint overwritten')
    assert policy_path.read_bytes()==policy_bytes


@pytest.mark.parametrize('envs,threads,expected',[(8,8,8),(16,16,8),(8,3,3),(2,8,2)])
def test_evaluation_worker_count_respects_configured_limits(tmp_path,monkeypatch,envs,threads,expected):
    commands=[]
    monkeypatch.setattr(launcher,'run_bounded',lambda command,*args:commands.append(command))
    settings=argparse.Namespace(bank_dir=tmp_path,eval_timeout=20,eval_trials=2,envs=envs,threads=threads)
    launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',tmp_path/'eval',settings,{})
    assert len(commands)==2
    assert all(command[command.index('--workers')+1]==str(expected) for command in commands)
    assert all(command[command.index('--nthread')+1]=='1' for command in commands)


def identical_evaluation_bank(folder):
    (folder/'family_eval').mkdir()
    for task in ('motor','horn','screw','retainer','yoke'):
        states=np.arange(12,dtype=float).reshape(6,2)
        np.savez(folder/f'{task}_eval.npz',states=states[:2],family=np.zeros(2,int),seed=[100,101])
        np.savez(folder/'family_eval'/f'{task}_eval.npz',states=states,
                 family=[0,0,1,2,3,4],seed=[100,101,102,103,104,105])


def test_identical_loose_episodes_reuse_full_family_results(tmp_path,monkeypatch):
    identical_evaluation_bank(tmp_path)
    commands=[]
    tasks={task:dict(trials=6,successes=1,episodes=[
        dict(entry=0,family='loose',success=True,safety=False,timeout=False,final_error_m=.001),
        dict(entry=1,family='loose',success=False,safety=False,timeout=True,final_error_m=.02),
        dict(entry=2,family='pickup',success=False,safety=True,timeout=False,final_error_m=.03)])
        for task in ('motor','horn','screw','retainer','yoke')}
    all_result=dict(policy='checkpoint.pt',seed=501,action_mode='sample',
                    families=['loose','pickup','inhand','partial','near'],tasks=tasks)
    def evaluate(command,*args):
        commands.append(command)
        Path(command[command.index('--out')+1]).write_text(json.dumps(all_result))
    monkeypatch.setattr(launcher,'run_bounded',evaluate)
    settings=argparse.Namespace(bank_dir=tmp_path,eval_timeout=20,eval_trials=2,envs=8,threads=8)
    folder=tmp_path/'evaluation'
    launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',folder,settings,{})
    assert len(commands)==1 and commands[0][commands[0].index('--families')+1]=='all'
    assert commands[0][commands[0].index('--workers')+1]=='8'
    result=json.loads((folder/'loose.json').read_text())
    assert result['families']==['loose'] and result['seed']==501 and result['action_mode']=='sample'
    assert result['derivation']=='derived from all-family evaluation; identical states and per-entry seeds'
    assert result['derived_from']==str((folder/'all.json').resolve())
    for task,values in result['tasks'].items():
        assert values['episodes']==tasks[task]['episodes'][:2]
        assert values['trials']==2 and values['successes']==1
    assert json.loads((folder/'all.json').read_text())==all_result


@pytest.mark.parametrize('mismatch',['states','indices','seeds'])
def test_different_loose_episodes_keep_two_evaluations(tmp_path,monkeypatch,mismatch):
    identical_evaluation_bank(tmp_path)
    path=tmp_path/'family_eval'/'yoke_eval.npz'
    with np.load(path) as saved:bank={key:saved[key].copy() for key in saved.files}
    if mismatch=='states':bank['states'][0,0]+=.001
    elif mismatch=='seeds':bank['seed'][0]+=1
    else:
        # Same loose states and stored seeds, but shifted bank indices change
        # the episode-private action RNG seed and therefore forbid reuse.
        for key in bank:bank[key]=bank[key][[2,0,1,3,4,5]]
    np.savez(path,**bank)
    commands=[]
    monkeypatch.setattr(launcher,'run_bounded',lambda command,*args:commands.append(command))
    settings=argparse.Namespace(bank_dir=tmp_path,eval_timeout=20,eval_trials=2)
    launcher.evaluate_checkpoint(tmp_path/'checkpoint.pt',tmp_path/'evaluation',settings,{})
    assert [command[command.index('--families')+1] for command in commands]==['loose','all']
