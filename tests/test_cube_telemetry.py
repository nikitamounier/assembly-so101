import json

import numpy as np

from assembly_policy.cube_trial.telemetry import EpisodeRecorder


def obs(n, x=0):
    return dict(proprio=np.zeros((n, 250)), part_p=np.full((n, 2, 3), x, float),
                part_q=np.tile([1., 0, 0, 0], (n, 2, 1)))


def test_complete_success_trace_and_raw_returns(tmp_path):
    rec = EpisodeRecorder(tmp_path, 1, ['loose'])
    info = dict(valid=np.array([True]), success=np.array([False]))
    rec.begin(obs(1), np.array([3]))
    rec.step(np.array([-.1]), np.array([True]), info, obs(1), np.array([0]), 1)
    assert rec.totals.sum() == 0  # startup episode has an unknown prefix
    rec.begin(obs(1, 1), np.array([4]))
    rec.step(np.array([-.2]), np.array([False]), info, None, np.array([0]), 2)
    rec.begin(obs(1, 2), np.array([4]))
    info['success'][0] = True
    rec.step(np.array([.9]), np.array([True]), info, obs(1, 3), np.array([0]), 2)
    result = json.loads((tmp_path / 'latest-success.json').read_text())
    assert result['bank_entry'] == 4 and result['complete_episode']
    assert len(result['frames']) == 3
    assert result['frames'][0][12] == 1 and result['frames'][-1][12] == 3
    np.testing.assert_allclose(result['episode_return'], .7)
    metrics = rec.flush(2, 3)
    assert metrics['fam_completed']['loose'] == 1
    np.testing.assert_allclose(metrics['fam_return']['loose'], .7)
    np.testing.assert_allclose(metrics['fam_reward']['loose'], .2)


def test_worker_loss_excludes_partial_episodes(tmp_path):
    rec = EpisodeRecorder(tmp_path, 1, ['loose'])
    info = dict(valid=np.array([True]), success=np.array([False]))
    rec.begin(obs(1), np.array([3]))
    rec.step(np.array([0.]), np.array([True]), info, obs(1), np.array([0]), 1)
    rec.begin(obs(1), np.array([4]))
    info['valid'][0] = False
    rec.step(np.array([0.]), np.array([False]), info, None, np.array([0]), 1)
    rec.begin(obs(1), np.array([4]))
    info['valid'][0] = True
    info['success'][0] = True
    rec.step(np.array([1.]), np.array([True]), info, obs(1), np.array([0]), 1)
    assert not (tmp_path / 'latest-success.json').exists()
    assert rec.flush(1, 2)['fam_completed']['loose'] == 0


def test_latest_success_is_retained_independently_for_each_family(tmp_path):
    rec = EpisodeRecorder(tmp_path, 2, ['loose', 'placed_held'])
    rec.complete[:] = True  # known complete reset starts for this isolated test
    rec.begin(obs(2), np.array([3, 4]))
    info = dict(valid=np.array([True, True]), success=np.array([True, True]))
    rec.step(np.array([.9, .8]), np.array([True, True]), info, obs(2), np.array([0, 1]), 1)
    left = json.loads((tmp_path/'latest-success-loose.json').read_text())
    placed = json.loads((tmp_path/'latest-success-placed_held.json').read_text())
    assert left['family'] == 'loose' and left['bank_entry'] == 3
    assert placed['family'] == 'placed_held' and placed['bank_entry'] == 4
    assert json.loads((tmp_path/'latest-success.json').read_text())['id'] == placed['id']
    rec.begin(obs(2), np.array([5, 6]))
    info['success'][1] = False
    rec.step(np.array([.7, 0]), np.array([True, False]), info, obs(1), np.array([0, 1]), 2)
    assert json.loads((tmp_path/'latest-success-loose.json').read_text())['id'] != left['id']
    assert json.loads((tmp_path/'latest-success-placed_held.json').read_text())['id'] == placed['id']
