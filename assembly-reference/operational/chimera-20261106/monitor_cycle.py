"""One allocated observation with durable raw receipts and deduplicated alert keys."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
lock=open(out/'cycle.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
series_path=out/'series.json';series=json.loads(series_path.read_text()) if series_path.exists() else {'failures':0,'conditions':{}}
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 /large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv/bin/python /large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/operational/chimera-20261106/observe.py']
try:
 r=subprocess.run(command,capture_output=True,text=True,timeout=90);raw=dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
except subprocess.TimeoutExpired as e:
 raw=dict(command=command,returncode=None,stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout,stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr,error='observation_timeout; original invocation terminated by subprocess.run; do not retry this cycle')
raw_path=out/(stamp+'.json');raw_path.write_text(json.dumps(raw,indent=2)+'\n');raw_sha=hashlib.sha256(raw_path.read_bytes()).hexdigest()
summary=None;conditions=[];now=time.time()
try:
 if raw['returncode']!=0:raise ValueError('Observation command failed')
 summary=json.loads(raw['stdout']);assert summary['allocated_monitor_job']=='3077705'
 series['failures']=0
 if not summary['source_valid']:conditions.append('source_integrity')
 conditions.extend(summary['errors'])
 phase=(summary.get('controller') or {}).get('phase')
 if phase=='complete':conditions.append('training_complete')
 if phase=='waiting_allocations':
  series.setdefault('capacity_wait_since',now)
  if now-series['capacity_wait_since']>=1800:conditions.append('capacity_wait_30min')
 else:series.pop('capacity_wait_since',None)
 if phase in ('running','learner_starting'):
  series.setdefault('startup_since',now)
  stale=summary['progress_age_s']
  if stale is not None and stale>1200:
   series['stale_count']=series.get('stale_count',0)+1
   if series['stale_count']>=3:conditions.append('progress_stall')
  else:series['stale_count']=0
  if summary['last_update'] is None and now-series['startup_since']>1800:conditions.append('startup_30min')
 else:series.pop('startup_since',None);series['stale_count']=0
except (ValueError,AssertionError,KeyError,TypeError) as e:
 series['failures']+=1
 if series['failures']>=3:conditions.append('observation_failure_3cycles')
 raw['classification_error']=str(e)
active={}
events=[]
for condition in sorted(set(conditions)):
 key=series['conditions'].get(condition,condition+':'+stamp)
 active[condition]=key
 events.append(dict(key=key,condition=condition,receipt=str(raw_path),receipt_sha256=raw_sha))
series['conditions']=active;series['last_receipt']=str(raw_path)
tmp=series_path.with_suffix('.tmp');tmp.write_text(json.dumps(series,indent=2)+'\n');tmp.replace(series_path)
print(json.dumps(dict(receipt=str(raw_path),receipt_sha256=raw_sha,summary=summary,observation_failure_streak=series['failures'],events=events,terminal=any(e['condition']=='training_complete' for e in events)),sort_keys=True),flush=True)
