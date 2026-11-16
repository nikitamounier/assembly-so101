import json
import tempfile
import unittest
from pathlib import Path
from export import collect,starting_states


class ExportTests(unittest.TestCase):
    def test_rollback_excludes_rejected_branch_and_post_checkpoint_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for version, iterations in [('v6', [24, 25, 26]), ('v8', [26, 544]), ('v9', [26])]:
                run = root / ('run-' + version); run.mkdir()
                (run / 'command.json').write_text('[]')
                (run / 'learner-process.json').write_text('{}')
                (run / 'log.jsonl').write_text(''.join(json.dumps(dict(it=it)) + '\n' for it in iterations))
            origin = dict(path=str(root / 'run-v6/checkpoint.before-bare-v8.pt'), it=25)
            (root / 'run-v9/config.json').write_text(json.dumps(dict(rollback_origin=origin)))
            result = collect([dict(root=str(root), discover=True)], scheduler=False)
            self.assertEqual(result['rollback_origin'], origin)
            self.assertEqual([(r['version'], r['it']) for r in result['previous_history']], [('v6', 24), ('v6', 25)])
            self.assertEqual([(r['version'], r['it']) for r in result['history']], [('v9', 26)])

    def test_version_discovery_follows_activated_successor_and_its_operators(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for version in ['v5','v6','v7']:
                run=root/('run-'+version);run.mkdir()
                (run/'checkpoint.pt').write_bytes(b'not deserialized')
                if version!='v7':
                    (run/'command.json').write_text('[]');(run/'learner-process.json').write_text('{}')
            ops=root/'v6';ops.mkdir();(ops/'cluster.json').write_text(json.dumps(dict(gpu_job='approved-fallback',cpu_jobs=[])))
            result=collect([dict(root=str(root),discover=True)],scheduler=False)
            self.assertEqual(result['version'],'v6')
            self.assertEqual(result['gpu_job'],'approved-fallback')
    def test_sampler_alignment_respects_task_family_cap_and_history_window(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bank=root/'bank';bank.mkdir();run=root/'run';run.mkdir()
            np.savez(bank/'cover_train.npz',family=np.array([0,1,0]),stage=np.array(['00_loose','01_pickup','00_loose']))
            torch.save(dict(cnt=np.array([3,10]),sum=np.array([1,3]),buf=np.zeros((2,5),np.uint8)),run/'sampler.pt')
            meta=dict(task_names=['cover'],records=[dict(task='cover',kind='prototype'),dict(task='cover',kind='variation'),dict(task='cover',kind='variation')])
            result=starting_states(run,bank,meta,['--bank-per-family','1'],True,8)
            loose,pickup=result['groups']
            self.assertEqual(loose['entries'],1)
            self.assertEqual(loose['success'],1/3)
            self.assertEqual(pickup['sampled_episodes'],10)
            self.assertEqual(pickup['history_episodes'],5)
            self.assertEqual(pickup['success'],3/5)
            self.assertEqual(pickup['variations'],1)
            self.assertAlmostEqual(sum(r['sampling_probability'] for r in result['groups']),1)

    def test_prepared_checkpoint_is_not_an_activated_learner(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'old',Path(temp)/'new'
            for root,name in [(a,'run-v3'),(b,'run-v4')]:
                run=root/name;run.mkdir(parents=True)
                (run/'checkpoint.pt').write_bytes(b'not loaded')
            (a/'run-v3/command.json').write_text('[]')
            (a/'run-v3/learner-process.json').write_text('{}')
            result=collect([dict(root=str(a),run='run-v3',version='v3'),dict(root=str(b),run='run-v4',version='v4')],scheduler=False)
            self.assertEqual(result['version'],'v3')

    def test_new_run_history_stays_separate_and_partial_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            candidates=[]
            for version,it in [('v3',8),('v4',9)]:
                root=Path(temp)/version;run=root/('run-'+version);run.mkdir(parents=True)
                (run/'command.json').write_text('[]');(run/'learner-process.json').write_text('{}')
                (run/'log.jsonl').write_text(json.dumps(dict(it=it,task_success={'base_side_cover':.5}))+'\n{"unfinished":')
                candidates.append(dict(root=str(root),run='run-'+version,version=version))
            result=collect(candidates,scheduler=False)
            self.assertEqual([r['it'] for r in result['history']],[9])
            self.assertEqual([r['it'] for r in result['previous_history']],[8])
            self.assertFalse(result['available_metrics']['task_return'])
            self.assertNotIn('task_return',result['history'][0])


if __name__=='__main__':unittest.main()
