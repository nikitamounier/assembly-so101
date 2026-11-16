"""Opt-in learner bootstrap; original trainer/worker/physics source stays unchanged.

Stage this file with telemetry.py beside it, then invoke with the exact normal
assembly_policy.train arguments in a checkpoint-safe continuation.
"""
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


def entry_metadata(trainer):
    entries=[]
    for task_index,task in enumerate(trainer.tasks):
        with np.load(Path(trainer.a.bank_dir)/(task+'_train.npz'),allow_pickle=False) as archive:
            family=archive['family'];stages=archive['stage'];keep=np.ones(len(family),bool)
            if trainer.a.bank_per_family>0:
                keep[:]=False
                for f in np.unique(family):keep[np.flatnonzero(family==f)[:trainer.a.bank_per_family]]=True
            for i in np.flatnonzero(keep):
                entry=len(entries)
                assert int(trainer.bank_T[entry])==task_index
                assert int(trainer.bank_F[entry])==int(family[i])
                entries.append(dict(task=task,stage=str(stages[i]),family=int(family[i])))
    assert len(entries)==len(trainer.bank_S)
    return entries


def recorder_type():
    path=Path(__file__).with_name('telemetry.py')
    # In the local checkout use the canonical module; releases stage it beside this bootstrap.
    if not path.exists():path=Path(__file__).resolve().parent.parent/'assembly-so101/assembly_policy/wowrobo/telemetry.py'
    spec=importlib.util.spec_from_file_location('_observational_training_replays',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.EpisodeRecorder


def install(training_module, recorder_class=None):
    base=training_module.Trainer
    recorder_class=recorder_class or recorder_type()

    class RecordedTrainer(base):
        def __init__(self,args):
            super().__init__(args)
            self.replay_recorder=None
            self.replay_disabled_reason=None
            if args.suite!='wowrobo':return
            try:
                identity={k:self.args.get(k) for k in ('model_signature','bank_signature','reward_signature','observation_signature')}
                identity.update(run=str(self.run),review_model=str(self.ecfg.review_model))
                self.replay_recorder=recorder_class(self.run,entry_metadata(self),self.pts.names,args.envs,identity,
                    horizon=round(self.ecfg.horizon_s/self.ecfg.control_dt),dt=self.ecfg.control_dt)
                close=self.env.close
                def observed_close():
                    self.replay_recorder.close()
                    return close()
                self.env.close=observed_close
                print(f'[replay] bounded observer enabled: {len(self.replay_recorder.keys)} task/start groups; no worker protocol or physics changes',flush=True)
            except Exception as exc:
                self.replay_disabled_reason=repr(exc)
                print('[replay] observer initialization disabled: '+repr(exc),flush=True)

        def collect(self):
            recorder=self.replay_recorder
            if recorder is None or self.replay_disabled_reason:return super().collect()
            observe,step=self.env.observe,self.env.step
            latest=[None]
            def observed():
                value=observe();latest[0]=value;return value
            def stepped(action):
                try:
                    if not self.replay_disabled_reason:
                        recorder.before_step(latest[0],self.env.entry,self.env.alive_mask())
                except Exception as exc:
                    self.replay_disabled_reason=repr(exc)
                    print('[replay] observer disabled: '+repr(exc),flush=True)
                result=step(action)
                try:
                    if not self.replay_disabled_reason:
                        reward,done,info,final_obs,_=result
                        recorder.after_step(reward,done,info,final_obs,self.it+1)
                except Exception as exc:
                    self.replay_disabled_reason=repr(exc)
                    print('[replay] observer disabled: '+repr(exc),flush=True)
                return result
            self.env.observe,self.env.step=observed,stepped
            try:return super().collect()
            finally:self.env.observe,self.env.step=observe,step

    training_module.Trainer=RecordedTrainer
    return RecordedTrainer


def main():
    from assembly_policy import train
    install(train)
    # The original CLI, resume guards, optimizer, signals and training loop remain authoritative.
    train.main()


if __name__=='__main__':main()
