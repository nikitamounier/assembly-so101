import assert from 'node:assert/strict';
import test from 'node:test';
import { latestTaskReplay, replayForSelection } from './task-replays.mjs';

const clip = (stage, time, success, extra = {}) => ({
  id: `${stage}-${time}`, task: 'cover', stage, recorded_at: time, success,
  duration_s: 1, complete_episode: true, source: 'actual_training_episode',
  identity: { run: 'current', model_signature: 'native' }, ...extra,
});
const pick = clips => latestTaskReplay(clips, 'cover', 'current', 'native');

test('prefers latest success across starting states over any newer failure', () => {
  const chosen = clip('entry', 20, true);
  assert.equal(pick({ 'cover/near': clip('near', 10, true), 'cover/entry': chosen,
    'cover/pickup': clip('pickup', 30, false) }), chosen);
});
test('uses latest failure across starts when no success is available', () => {
  const chosen = clip('pickup', 30, false);
  assert.equal(pick({ 'cover/entry': clip('entry', 20, false), 'cover/pickup': chosen }), chosen);
});
test('does not substitute another run, model, task or incomplete recording', () => {
  assert.equal(pick({
    'cover/old': clip('old', 99, true, { identity: { run: 'old', model_signature: 'native' } }),
    'cover/model': clip('model', 99, true, { identity: { run: 'current', model_signature: 'other' } }),
    'cover/partial': clip('partial', 99, true, { complete_episode: false }),
    'cover/reset': clip('reset', 99, true, { source: 'reset_preview' }),
    'other/near': clip('near', 99, true, { task: 'other' }),
    'cover/wrong-key': clip('near', 99, true),
  }), null);
  assert.equal(pick({}), null);
});

test('explicit starting state stays pinned instead of switching to another success', () => {
  const selected = clip('left_support', 10, false);
  const replays = { 'cover/left_support': selected, 'cover/right_support': clip('right_support', 99, true) };
  assert.equal(replayForSelection(replays, 'cover', 'left_support', 'current', 'native'), selected);
  assert.equal(replayForSelection(replays, 'cover', 'unrecorded', 'current', 'native'), null);
  assert.equal(replayForSelection(replays, 'cover', null, 'current', 'native'), replays['cover/right_support']);
});
