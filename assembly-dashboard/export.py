"""Small read-only allocated export. No checkpoint deserialization or credential files."""
import json
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path


def starting_states(run, bank_path, bank, command, include_sampler, iteration):
    import numpy as np
    def option(key, default):
        return command[command.index(key)+1] if key in command else default
    tasks=bank.get('task_names',[])
    task_arg=option('--tasks','all')
    if task_arg not in ('all','wowrobo'):tasks=task_arg.split('+')
    limit=int(option('--bank-per-family',640))
    entries=[];groups={}
    for task in tasks:
        path=bank_path/(task+'_train.npz')
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            family=z['family'];stages=z['stage'] if 'stage' in z else np.array(['unknown']*len(family))
            keep=np.ones(len(family),bool)
            if limit>0:
                keep[:]=False
                for f in np.unique(family):keep[np.flatnonzero(family==f)[:limit]]=True
            records=[r for r in bank.get('records',[]) if r.get('task')==task]
            for index in np.flatnonzero(keep):
                stage=str(stages[index]);key=task+'/'+stage
                group=groups.setdefault(key,dict(task=task,stage=stage,entries=0,prototypes=0,variations=0,eval_entries=None))
                group['entries']+=1
                kind=records[index].get('kind','prototype') if len(records)==len(family) else None
                if kind=='prototype':group['prototypes']+=1
                elif kind=='variation':group['variations']+=1
                entries.append(key)
    result=dict(groups=list(groups.values()),sampler_available=False)
    if not include_sampler:return result
    sampler_path=run/'sampler.pt'
    if not sampler_path.exists():return result
    # This is the tiny sampler artifact, never policy/Adam checkpoint weights.
    import io,torch
    torch.set_num_threads(1)
    before=sampler_path.stat().st_mtime_ns
    raw=sampler_path.read_bytes()
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if sampler_path.stat().st_mtime_ns!=before:
        result['sampler_error']='Sampler changed during observation';return result
    if len(entries)!=len(state['cnt']):
        result['sampler_error']='Sampler and selected bank entry counts differ';return result
    horizon=int(state['buf'].shape[1]);cnt=np.asarray(state['cnt']);sums=np.asarray(state['sum']);n=np.minimum(cnt,horizon)
    eps=float(option('--eps-q',.0001));p=np.divide(sums,np.maximum(n,1));weights=eps+np.sqrt(np.maximum(p*(1-p),0))
    if option('--sampler','beta')=='uniform':weights[:]=1
    weights=weights/weights.sum()
    for key,group in groups.items():
        ids=np.array([i for i,k in enumerate(entries) if k==key],dtype=int)
        count=int(n[ids].sum());wins=int(sums[ids].sum())
        group.update(sampled_episodes=int(cnt[ids].sum()),history_episodes=count,history_wins=wins,
                     success=wins/count if count else None,visited_entries=int((cnt[ids]>0).sum()),sampling_probability=float(weights[ids].sum()))
    # The log is flushed before sampler saving. Skip an in-flight newer log row.
    log=run/'log.jsonl'
    aligned=not log.exists() or before>=log.stat().st_mtime_ns
    result.update(sampler_available=True,sampler_saved_at=before/1e9,sampler_identity=str(before),history_per_entry=horizon,
                  iteration=iteration if aligned else None,observation_unix=time.time())
    return result


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def history(run, version):
    path = run / 'log.jsonl'
    if not path.exists():
        return []
    rows = []
    fields = ('it', 'transitions', 'fps', 'success', 'safety', 'episodes', 'ep_len',
              'valid_transitions', 'target_transitions', 'fam_success', 'task_success',
              'task_episodes', 'fam_return', 'task_return', 'lr', 'pi', 'vf', 'kl', 'ent')
    with path.open() as stream:
        for line in deque(stream, maxlen=1200):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            clean = {k: row[k] for k in fields if k in row}
            clean['version'] = version
            clean['workers'] = {k: row.get('workers', {}).get(k) for k in
                                ('active_envs', 'rt_med', 'rt_p90', 'ct_mean')}
            rows.append(clean)
    return rows


def collect(candidates, scheduler=True, include_sampler=False):
    expanded=[]
    for candidate in candidates:
        if candidate.get('discover'):
            runs=sorted((p for p in Path(candidate['root']).glob('run-v*') if re.fullmatch(r'run-v\d+',p.name)),key=lambda p:int(p.name[5:]))
            expanded.extend(dict(root=candidate['root'],run=p.name,version=p.name[4:]) for p in runs[-20:])
        else:expanded.append(candidate)
    activated = []
    for candidate in expanded:
        root, version = Path(candidate['root']), candidate['version']
        run = root / candidate['run']
        if (run / 'command.json').exists() and (run / 'learner-process.json').exists():
            activated.append((root, run, version))
    if not activated:
        raise RuntimeError('No activated assembly learner found')
    root, run, version = activated[-1]
    canonical_ops = root / 'robotics-assembly-3.0/operational/wowrobo-chimera-20261107'
    ops=root/version if (root/version/'cluster.json').exists() else canonical_ops
    cfg = read_json(ops / 'cluster.json', {})
    controller = read_json(run / 'controller.json', {})
    training_config = read_json(run / 'config.json', {})
    command = read_json(run / 'command.json', [])
    def option(key, default=None):
        return command[command.index(key)+1] if key in command else default
    bank_path = Path(option('--bank-dir', str(root / 'absent')))
    bank = read_json(bank_path / 'manifest.json', {})
    current = history(run, version)
    previous = [row for _, other, name in activated[:-1] for row in history(other, name)]
    rollback = training_config.get('rollback_origin')
    origin = training_config.get('parent_checkpoint') or training_config.get('continuation_origin') or rollback
    if origin:
        ancestors = {}
        for _ in range(20):
            origin_run = Path(origin['path']).parent
            if origin_run in ancestors or not origin_run.is_relative_to(root):
                break
            ancestors[origin_run] = int(origin['it'])
            parent_config = read_json(origin_run / 'config.json', {})
            origin = parent_config.get('parent_checkpoint') or parent_config.get('continuation_origin') or parent_config.get('rollback_origin')
            if not origin:
                break
        previous = [row for _, other, name in activated[:-1] if other in ancestors
                    for row in history(other, name) if row['it'] <= ancestors[other]]
    queue = []
    jobs = cfg.get('cpu_jobs', [])
    ledger=read_json(root/('autoscale-'+version)/'submissions.json',{})
    jobs=list(dict.fromkeys([*jobs,*[str(j) for j in ledger.get('jobs',[]) if str(j).isdigit()]]))
    if scheduler and jobs:
        query = subprocess.run(['squeue', '--noheader', '--jobs='+','.join(jobs),
                                '--format=%i|%T|%C|%N'], capture_output=True, text=True, timeout=15)
        if query.returncode == 0:
            for line in query.stdout.splitlines():
                fields = line.split('|')
                if len(fields) == 4:
                    queue.append(dict(id=fields[0], state=fields[1], cpus=int(fields[2]), host=fields[3]))
    migration = read_json(root / ('migration-'+version+'.json')) or read_json(run/'migration.json')
    stage_data=starting_states(run,bank_path,bank,command,include_sampler,current[-1]['it'] if current else (migration or {}).get('parent_it'))
    bank_version=re.search(r'v\d+$',bank_path.name)
    frozen=None
    if bank_version:
        for path in [root/bank_version[0]/('accepted-states-'+bank_version[0])/'manifest.json',
                     ops/('accepted-states-'+bank_version[0])/'manifest.json',
                     canonical_ops/'modal-live'/('accepted-states-'+bank_version[0])/'manifest.json']:
            frozen=read_json(path)
            if frozen is not None:break
    if frozen is None and bank_version and bank_version[0]=='v4':
        frozen=read_json(canonical_ops/'modal-live/accepted-states/manifest.json')
    if frozen:
        for group in stage_data['groups']:
            group['eval_entries']=sum(r.get('split')=='eval' and r.get('stage')==group['stage'] for r in frozen.get('records',[]))
    now = time.time()
    ck = run / 'checkpoint.pt'
    errors = [key+': exit '+str(v.get('exit')) for key, v in controller.get('workers', {}).items()
              if v.get('exit') not in (None, 0) and v.get('scheduler', [None])[0] == 'RUNNING']
    # Future native replay producers must use this explicit, complete-episode contract.
    clips = []
    for path in run.glob('latest-success*.json'):
        clip = read_json(path, {})
        if (clip.get('success') is True and clip.get('complete_episode') is True
                and clip.get('run') == str(run) and clip.get('frames')):
            clips.append(clip)
    recording=read_json(run/'training-replays/manifest.json',{})
    replay_index=read_json(run/'training-replays/index.json',{})
    config=read_json(run/'config.json',{})
    recording_valid=bool(recording and recording.get('identity',{}).get('run')==str(run)
                      and all(recording['identity'].get(k)==config.get(k) for k in
                              ('model_signature','bank_signature','reward_signature','observation_signature')))
    replay_valid=recording_valid and replay_index.get('identity')==recording.get('identity')
    preferred={}
    if replay_valid:
        for key,versions in replay_index.get('clips',{}).items():
            for outcome in ('success','failure'):
                clip=versions.get(outcome,{})
                if clip.get('complete_episode') and clip.get('source')=='actual_training_episode' and clip.get('identity')==recording['identity']:
                    preferred[key]=clip;break
    return dict(observed_at=now, root=str(root), run=str(run), version=version,
                history=current, previous_history=previous, migration=migration,
                target_envs=int(option('--envs', 0)), steps=int(option('--steps', 32)),
                worker_slots=int(option('--remote-workers', 0)),
                controller_age_s=now-controller['time'] if controller.get('time') else None,
                training_age_s=now-(run/'log.jsonl').stat().st_mtime if current else None,
                checkpoint_bytes=ck.stat().st_size if ck.exists() else 0,
                checkpoint_saved_at=ck.stat().st_mtime if ck.exists() else None,
                cpu_jobs=queue, configured_cpu_jobs=len(jobs), gpu_job=controller.get('gpu_job',cfg.get('gpu_job')),
                training_rows=len(bank.get('records', [])), prototypes=bank.get('passed_source_stages'),
                qualification=('Approximate bare MuJoCo; native-contact qualification does not apply' if training_config.get('physics_backend')=='bare-mujoco-boxes-v8' else bank.get('qualification')),
                model_signature=training_config.get('physics_signature',bank.get('model_signature')),
                source_model_signature=bank.get('model_signature'), physics_backend=training_config.get('physics_backend'),
                rollback_origin=rollback,
                bank=str(bank_path), bank_schema=bank.get('schema'), errors=errors,
                phase=controller.get('phase', 'running' if current else 'starting'),
                learner_closed=any(run.glob('handoff-closed*.json')),
                episodes=clips,
                starting_states=stage_data,
                replays=preferred,recording=recording if recording_valid else None,
                available_metrics=dict(task_return=any('task_return' in r for r in current),
                                       fam_return=any('fam_return' in r for r in current)))


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Cluster dashboard reads must use an allocation'
    print(json.dumps(collect(CANDIDATES,include_sampler=INCLUDE_SAMPLER), separators=(',', ':')))
