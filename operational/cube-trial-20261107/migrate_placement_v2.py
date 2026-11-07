"""Explicit audited reward-only checkpoint migration after a clean learner stop."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from assembly_policy.cube_trial.resets import directory_signature, signature

ROOT = Path(__file__).resolve().parents[2]
OLD_ENV_SHA = 'c27dcf36652e8f155a64e2a701a29c02fee98dd1154eff0ce8a190bd1f3c25a2'
NEW_ENV_SHA = '224c2fd7070045c82a017313e14bbbc221ad29f41aa838ae001e973772d970e8'


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Migration must be allocated'
    cfg = json.loads((ROOT/'operational/cube-trial-20261107/cluster.json').read_text())
    receipt = json.loads((ROOT/'artifacts_local/chimera/controller-adoption.json').read_text())
    assert receipt['learner'] is None and receipt['workers'] == {}
    old = Path('/proc')/str(receipt['old_controller_pid'])/'stat'
    assert old.read_text().split()[2] == 'T'
    path = ROOT/cfg['run']/'checkpoint.pt'
    archive = path.with_name('checkpoint.before-placement-v2.pt')
    assert not archive.exists(), 'Reward migration already has an archive; reconcile instead of repeating'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    assert checkpoint['it'] == receipt['graceful_checkpoint_iteration']
    assert args['environment_signature'] == OLD_ENV_SHA
    assert hashlib.sha256((ROOT/'assembly_policy/cube_trial/env.py').read_bytes()).hexdigest() == NEW_ENV_SHA
    assert args['model_signature'] == signature() == args['physics_signature']
    assert args['bank_directory_signature'] == directory_signature(ROOT/cfg['bank'])
    changed = copy.deepcopy(checkpoint)
    changed['args']['environment_signature'] = NEW_ENV_SHA
    changed['args']['reward_contract_version'] = 'placement_v2'
    changed['reward_migration'] = dict(from_environment_sha256=OLD_ENV_SHA, to_environment_sha256=NEW_ENV_SHA,
                                     reason='Explicit user placement-contact/lookback/release/speed reward revision')
    shutil.copy2(path, archive)
    torch.save(changed, path.with_suffix('.pt.reward-tmp'))
    restored = torch.load(path.with_suffix('.pt.reward-tmp'), map_location='cpu', weights_only=False)
    for key in ('model', 'opt', 'sampler', 'lr', 'it', 'transitions'):
        assert equal(checkpoint[key], restored[key]), 'Checkpoint training state changed: '+key
    os.replace(path.with_suffix('.pt.reward-tmp'), path)
    latest = path.with_name('latest-success.json')
    if latest.exists():
        latest.rename(latest.with_name('latest-success.before-placement-v2.json'))
    audit = dict(iteration=checkpoint['it'], transitions=checkpoint['transitions'],
                 training_state_exactly_preserved=True, archive=str(archive),
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 new_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 **changed['reward_migration'])
    (ROOT/'artifacts_local/chimera/reward-placement-v2-migration.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__ == '__main__':
    main()
