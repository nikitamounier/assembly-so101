"""Bounded restartable allocated partition supervisor for one reset job."""
import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/reset-bank-farm-20261107';out.mkdir(exist_ok=True)
remote='/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4'
py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
cmd=['ssh','-o','BatchMode=yes','chimera-login',f'srun --jobid=3620194 --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=cores {py} {remote}/launch/farm_control.py supervise-partitions --root {remote} --timeout 5400']
def status(v):
 p=out/'supervisor-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
end=time.monotonic()+10800
for attempt in range(30):
 if time.monotonic()>end:break
 status({'pid':os.getpid(),'job':3620194,'status':'supervising_partitions','attempt':attempt,'unix':time.time()})
 code=subprocess.run(cmd).returncode
 if code==0:
  status({'pid':os.getpid(),'job':3620194,'status':'partitions_merged_coordinator_resumed','unix':time.time()});raise SystemExit(0)
 status({'pid':os.getpid(),'job':3620194,'status':'retrying_allocated_supervisor','last_exit':code,'unix':time.time()});time.sleep(20)
status({'pid':os.getpid(),'job':3620194,'status':'bounded_supervisor_wait_exhausted','unix':time.time()});raise SystemExit(2)
