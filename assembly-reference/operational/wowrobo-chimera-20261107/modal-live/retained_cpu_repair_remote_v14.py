"""Restore only timed-out retained leases during the explicit state-build pause."""
from pathlib import Path
import argparse,fcntl,hashlib,json,re,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--active-run',choices=['run-v14']);a=p.parse_args()
ROOT=Path(__file__).resolve().parent.parent
folder=ROOT/'resource-repair';folder.mkdir(exist_ok=True)
lock=(folder/'repair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
priority=json.loads((ROOT/'run-v10/state-build-cpu-priority.json').read_text())
active=priority.get('preserve_training_checkpoint') is True and priority.get('state_root','').endswith('/reset-bank-19200-20261107-v1')
if a.active_run:
    live=json.loads((ROOT/a.active_run/'initial-restored.json').read_text())
    assert live['status']=='PASS' and live['physics_signature']=='c7dd67ba82f88fa6b71cc1d23d25fea997996e13870bffe8bee14df3076b0f17'
    assert {str(j) for j in range(3602571,3602611)}<=set(json.loads((ROOT/'ops-v14/cluster.json').read_text())['cpu_jobs'])
    active=True
if not active:
    print(json.dumps(dict(action='keep_monitor',reason='state_build_priority_not_active')));raise SystemExit
jobs=json.loads((ROOT/'ops-v10.json').read_text())['config']['cpu_jobs']
assert len(jobs)==40 and set(jobs)=={str(j) for j in range(3602571,3602611)}
assert hashlib.sha256((ROOT/'run-v10/checkpoint.before-state-build-priority.pt').read_bytes()).hexdigest()=='1ffa1031de2429c7f7263680b5f07662cff65213f2c7e1227340c145ebc1b4d6'
pause=json.loads((folder/'native-pause-proof.json').read_text());assert pause['status']=='PASS' and pause['policy_optimizer_finite'] and pause['original31_preserved']
assert pause['it']==32 and pause['transitions']==1046323 and pause['source_model']=='34d224d4edff266369b3f1322baf80013c0daf09599a9eeb8751c6270a88ff92'
assert hashlib.sha256((ROOT/'run-v10/checkpoint.pt').read_bytes()).hexdigest()==pause['checkpoint_sha256']=='64a8d77312dfc5aaea84896d4e5cebeb7143d4cd175c937ccf4dd74ea303a237'
path=folder/'terminal-requeue-ledger.json'
ledger=json.loads(path.read_text()) if path.exists() else {}
def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(path)
def inspect(job):
    p=subprocess.run(['scontrol','show','job',job,'-o'],text=True,capture_output=True,check=True,timeout=15)
    return dict(re.findall(r'(\w+)=(\S*)',p.stdout))
result=dict(action='keep_monitor',active_run=a.active_run,apply=a.apply,timed_out=[],attempts=[],uncertain_or_rejected=[])
for job in jobs:
    row=inspect(job)
    if row.get('JobState')!='TIMEOUT':continue
    result['timed_out'].append(job)
    expected='/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/operational/chimera-20261106/node_workers.sbatch'
    assert row['JobId']==job and row['UserId']=='nikitamounier(10258)' and row['Partition']=='cpu_preemptible'
    assert row['NumCPUs']=='10' and row['Requeue']=='1' and row['Command']==expected
    assert row['JobName']=='so101-cpu10-'+str(int(job)-3602571)
    key=job+':'+row['Restarts']+':'+row['EndTime']
    if key in ledger:
        if ledger[key]['status']!='accepted':result['uncertain_or_rejected'].append(key)
        continue
    if not a.apply:continue
    current=inspect(job)
    if any(current.get(k)!=row.get(k) for k in ('JobState','Restarts','EndTime','Command','UserId')):continue
    entry=dict(status='prepared',job=job,restarts_before=row['Restarts'],end_time=row['EndTime'],time=time.time())
    ledger[key]=entry;save()  # Prepared intent is a permanent fence if execution becomes unknown.
    try:
        done=subprocess.run(['scontrol','requeue',job],text=True,capture_output=True,timeout=20)
        entry.update(status='accepted' if done.returncode==0 else 'rejected',returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
    except subprocess.TimeoutExpired:
        entry.update(status='uncertain',reason='scontrol_timeout_do_not_retry')
    save();result['attempts'].append(entry);result['action']='requeued_terminal_lease' if entry['status']=='accepted' else 'ACTION inspect_terminal_requeue'
    break  # One positively identified terminal lease per cycle; never interrupt running jobs.
if result['uncertain_or_rejected']:result['action']='ACTION inspect_terminal_requeue'
(folder/'latest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
