"""Read-only, allocated export for the local live dashboard; never reads server.json."""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--known-episode-id', default='')
    parser.add_argument('--known-family-episodes', default='{}')
    args = parser.parse_args()
    known_families = json.loads(args.known_family_episodes)
    assert os.environ.get('SLURM_JOB_ID'), 'Dashboard cluster reads must be allocated'
    cfg = json.loads((ROOT / 'operational/cube-trial-20261107/cluster.json').read_text())
    run = ROOT / cfg['run']
    history = []
    if (run / 'log.jsonl').exists():
        with (run / 'log.jsonl').open() as stream:
            for line in deque(stream, maxlen=1200):
                try:
                    row = json.loads(line)
                    history.append({k: row.get(k) for k in ('it', 'transitions', 'fps', 'timestamp', 'success', 'fam_success', 'fam_reward', 'fam_return', 'fam_completed')})
                except json.JSONDecodeError:
                    pass
    summary = json.loads((run / 'dashboard-summary.json').read_text()) if (run / 'dashboard-summary.json').exists() else None
    state = json.loads((ROOT / 'artifacts_local/chimera/controller.json').read_text())
    episode = json.loads((run / 'latest-success.json').read_text()) if (run / 'latest-success.json').exists() else None
    episode_id = episode['id'] if episode else None
    FAMILIES = tuple((summary or {}).get("families", {}))
    family_clips = {}
    for family in FAMILIES:
        path = run / ('latest-success-' + family + '.json')
        if path.exists():
            clip = json.loads(path.read_text())
            if clip['family'] == family:
                family_clips[family] = clip
    if episode:
        family = episode['family']
        if family not in family_clips or episode['recorded_at'] > family_clips[family]['recorded_at']:
            family_clips[family] = episode
    episode_ids = {family: clip['id'] for family, clip in family_clips.items()}
    changed_clips = {family: clip for family, clip in family_clips.items() if known_families.get(family) != clip['id']}
    if episode_id == args.known_episode_id:

        episode = None
    print(json.dumps(dict(observed_at=time.time(), run=cfg['run'], history=history, summary=summary,
          episode=episode, episode_id=episode_id, episode_ids=episode_ids, episodes=changed_clips, phase=state.get('phase') if state.get('run') == str(run) else 'starting_new_run',
          controller_age_s=time.time()-state.get('observed_utc', 0),
          training_age_s=time.time()-(run/'log.jsonl').stat().st_mtime if history else None,
          target_envs=cfg['envs'], target_cpu_jobs=len(cfg['cpu_jobs']),
          running_cpu_jobs=sum(state.get('jobs', {}).get(j, {}).get('JobState')=='RUNNING' for j in cfg['cpu_jobs']),
          errors=state.get('errors', [])), separators=(',', ':')))


if __name__ == '__main__':
    main()
