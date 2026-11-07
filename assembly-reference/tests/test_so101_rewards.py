"""Actual native completion rewards and rejection after previously earned success."""
import json
import os
from pathlib import Path
import pytest
from assembly_policy.so101.env import EnvConfig,TASK_NAMES
from scripts.check_so101_rewards import audit_rewards


@pytest.fixture(scope='module')
def audit():
    report=audit_rewards()
    # Optional receipt from the same executions; no second qualification run.
    output=os.environ.get('SO101_REWARD_AUDIT_OUTPUT')
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    return report


@pytest.mark.parametrize('task',TASK_NAMES)
@pytest.mark.parametrize('profile',['default','guarded'])
def test_completed_part_earns_positive_terminal_reward(audit,task,profile):
    row=next(x for x in audit['cases'] if x['case']=='completed' and x['task']==task and x['profile']==profile)
    assert row['passed'],row
    assert all(not x['success'] and x['reward']<=0 for x in row['history'][:-1])
    assert row['history'][-1]['time_s']>=.44-1e-9


@pytest.mark.parametrize('case',['misplaced_part','misplaced_predecessor','loose_reset',
                                'wrong_orientation_geometry','translated_geometry'])
def test_wrong_parts_reject_success_and_targets_follow_the_base(audit,case):
    rows=[x for x in audit['cases'] if x['case']==case]
    assert len(rows)==(len(TASK_NAMES)-1 if case=='misplaced_predecessor' else len(TASK_NAMES))
    assert all(x['passed'] for x in rows),rows


def test_direct_training_defaults_match_the_qualified_pose_reward(tmp_path):
    from assembly_policy.train import parse
    assert EnvConfig().stability_metric=='pose'
    base=['--suite','so101','--run-dir',str(tmp_path)]
    assert parse(base).so101_stability=='pose'
    assert parse([*base,'--so101-stability','velocity']).so101_stability=='velocity'
