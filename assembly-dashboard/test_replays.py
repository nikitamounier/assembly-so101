import importlib.util
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
import types

import numpy as np

MODULE=Path(__file__).resolve().parent.parent/'assembly-so101/assembly_policy/wowrobo/telemetry.py'
spec=importlib.util.spec_from_file_location('replay_tests_module',MODULE)
telemetry=importlib.util.module_from_spec(spec);spec.loader.exec_module(telemetry)


class Sink:
    def __init__(self):self.clips=[]
    def submit(self,clip):self.clips.append(clip);return True
    def close(self):pass


def observation(marker,n=1,bodies=2):
    return dict(part_p=np.full((n,bodies,3),marker,dtype=np.float32),
                part_q=np.tile(np.array([1,0,0,0],np.float32),(n,bodies,1)))


def info(n=1,success=False,valid=True):
    return dict(valid=np.full(n,valid),success=np.full(n,success),
                safety=np.full(n,not success),timeout=np.zeros(n,bool))


class ReplayTests(unittest.TestCase):
    def test_bootstrap_preserves_observations_results_rng_and_restores_methods(self):
        import record_training
        class Recorder:
            keys=['cover/01_pickup']
            def __init__(self,*a,**kw):self.before=0;self.after=0
            def before_step(self,*a):self.before+=1
            def after_step(self,*a):self.after+=1
            def close(self):pass
        class Env:
            entry=np.array([0])
            def observe(self):return observation(1)
            def alive_mask(self):return np.ones(1,bool)
            def step(self,action):return (action.copy(),np.ones(1,bool),info(),observation(9),np.array([0]))
            def close(self):pass
        class Trainer:
            def __init__(self,args):
                self.a=args;self.args={};self.tasks=['cover'];self.bank_T=np.array([0]);self.bank_F=np.array([1]);self.bank_S=np.zeros((1,2));self.run=Path(args.bank_dir);self.pts=types.SimpleNamespace(names=['robot','part']);self.ecfg=types.SimpleNamespace(review_model='review.xml',horizon_s=1,control_dt=.04);self.env=Env();self.it=8
            def collect(self):
                output=[]
                for _ in range(2):
                    self.env.observe();output.append(self.env.step(np.random.normal(size=2))[0])
                self.env.observe()  # Final value bootstrap observation is not a simulated step.
                return np.stack(output)
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp)/'cover_train.npz',family=np.array([1]),stage=np.array(['01_pickup']))
            args=types.SimpleNamespace(bank_dir=temp,bank_per_family=640,envs=1,suite='wowrobo')
            module=types.SimpleNamespace(Trainer=Trainer)
            wrapped=record_training.install(module,Recorder)
            np.random.seed(42);expected=Trainer(args).collect();state=np.random.get_state()
            np.random.seed(42);trainer=wrapped(args);before=trainer.env.observe
            actual=trainer.collect()
            np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(np.random.get_state()[1],state[1])
            self.assertEqual(trainer.env.observe,before)
            self.assertEqual((trainer.replay_recorder.before,trainer.replay_recorder.after),(2,2))

    def recorder(self,sink,n=1,entries=None):
        return telemetry.EpisodeRecorder('/fixture/run',entries or [dict(task='cover',stage='01_pickup')],
                    ['robot','part'],n,dict(model_signature='physical'),horizon=8,dt=.04,frame_stride=1,
                    refresh_seconds=0,writer=sink)

    def test_partial_start_is_skipped_and_terminal_frame_precedes_reset(self):
        sink=Sink();rec=self.recorder(sink)
        rec.before_step(observation(10),np.array([0]),np.ones(1,bool),now=1)
        rec.after_step(np.array([0.]),np.ones(1,bool),info(),observation(11),1,now=1)
        self.assertEqual(len(sink.clips),0)
        rec.before_step(observation(20),np.array([0]),np.ones(1,bool),now=2)
        rec.after_step(np.array([-.2]),np.zeros(1,bool),info(),None,1,now=2)
        rec.before_step(observation(21),np.array([0]),np.ones(1,bool),now=3)
        rec.after_step(np.array([-.3]),np.ones(1,bool),info(),observation(99),1,now=3)
        clip=sink.clips[0]
        self.assertTrue(clip['complete_episode']);self.assertFalse(clip['success'])
        self.assertAlmostEqual(clip['episode_return'],-.5)
        np.testing.assert_array_equal(clip['frames'][:,0,0],[20,21,99])
        np.testing.assert_allclose(clip['times'],[0,.04,.08])
        self.assertEqual(clip['source'],'actual_training_episode')

    def test_first_success_replaces_failure_priority_and_later_failures_do_not(self):
        sink=Sink();rec=self.recorder(sink)
        for step,success in enumerate([False,False,True,False],1):
            rec.before_step(observation(step),np.array([0]),np.ones(1,bool),now=step)
            rec.after_step(np.array([int(success)]),np.ones(1,bool),info(success=success),observation(step+10),step,now=step)
        self.assertEqual([clip['success'] for clip in sink.clips],[False,True])

    def test_worker_loss_discards_fragment_and_buffer_count_is_per_start(self):
        sink=Sink();rec=self.recorder(sink,n=5090)
        self.assertEqual(rec.frames.shape[0],1)
        rec.fresh[0]=True
        entries=np.zeros(5090,int);alive=np.ones(5090,bool)
        rec.before_step(observation(1,5090),entries,alive,now=1)
        data=info(5090);data['valid'][0]=False
        rec.after_step(np.zeros(5090),np.zeros(5090,bool),data,None,1,now=1)
        rec.before_step(observation(2,5090),entries,alive,now=2)
        rec.after_step(np.zeros(5090),np.ones(5090,bool),info(5090),observation(3,5090),1,now=2)
        self.assertEqual(len(sink.clips),0)

    def test_writer_artifacts_are_bounded_and_queue_never_waits(self):
        writer=telemetry.ReplayWriter.__new__(telemetry.ReplayWriter)
        writer.pending=queue.Queue(maxsize=1);writer.dropped=0;writer.error=None
        self.assertTrue(writer.submit({}))
        start=time.perf_counter();self.assertFalse(writer.submit({}))
        self.assertLess(time.perf_counter()-start,.05)
        with tempfile.TemporaryDirectory() as folder:
            real=telemetry.ReplayWriter(folder,dict(identity={'model':'fixture'}))
            for i,success in enumerate([False,True,True]):
                clip=dict(id=str(i),task='cover',stage='01_pickup',success=success,complete_episode=True,
                          frames=np.full((3,2,7),i,np.float32),times=np.arange(3,dtype=np.float32))
                self.assertTrue(real.submit(clip));real.pending.join()
            index=json.loads((Path(folder)/'training-replays/index.json').read_text())
            self.assertEqual(len(list((Path(folder)/'training-replays').glob('*.npz'))),2)
            self.assertEqual(index['clips']['cover/01_pickup']['success']['id'],'2')
            self.assertEqual(index['clips']['cover/01_pickup']['failure']['id'],'0')
            real.close()


if __name__=='__main__':unittest.main()
