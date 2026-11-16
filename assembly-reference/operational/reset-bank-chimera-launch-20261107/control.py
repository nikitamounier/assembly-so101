"""Inspect/pause only the exact reset coordinator, never any training process."""
import argparse,json,os,signal,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('action',choices=['status','pause','resume','stop-preflight']);a=p.parse_args()
r=Path('/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4')
d=json.loads((r/'allocated.json').read_text());assert d['job_id']==os.environ['SLURM_JOB_ID']
pid=d['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();assert b'launch/driver.py' in cmd and str(r).encode() in cmd
if a.action=='pause':os.kill(pid,signal.SIGSTOP)
if a.action=='resume':os.kill(pid,signal.SIGCONT)
if a.action=='stop-preflight':
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            status=(item/'stat').read_text().rsplit(')',1)[1].split()
            childcmd=(item/'cmdline').read_bytes()
            if int(status[1])==pid and b'launch/full_preflight.py' in childcmd and str(r).encode() in childcmd:
                os.kill(int(item.name),signal.SIGTERM)
                print('Stopped redundant offline-authoring preflight child',item.name)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for filename in ['run-status.json','runtime-smoke.json']:
 path=r/filename
 if path.exists():
  v=json.loads(path.read_text());print(filename,{k:v[k] for k in ['status','stage','physical_admission_pass','reasons'] if k in v})
lines=subprocess.check_output(['ps','-eo','pid,ppid,rss,etime,args'],text=True).splitlines()
rows=[line for line in lines if str(r) in line and ('launch/driver.py' in line or 'full_preflight.py' in line or 'build_wowrobo_randomized_bank.py' in line)]
print('matching_processes',len(rows));print('\n'.join(rows[:8]))
