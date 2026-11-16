"""Checkpoint-safe close of the old learner; preserve CPU and GPU leases."""
from pathlib import Path
import json,subprocess,shlex,os,time,signal,sys,hashlib
HERE=Path(__file__).parent;old=json.loads((HERE.parent/'target.json').read_text());r=old['remote'];py=old['python']
relay=f'''from pathlib import Path
import os,signal,json,time
root=Path({r!r});script=root/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107/cluster.py';stopped=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if str(script).encode() in a and b'control' in a:os.kill(int(p.name),signal.SIGSTOP);stopped.append(dict(pid=int(p.name),start=(p/'stat').read_text().split()[21]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
(root/'run-v2/handoff-controller.json').write_text(json.dumps(dict(stopped=stopped,time=time.time()),indent=2)+'\\n');print(json.dumps(dict(stopped_controllers=stopped)))
'''
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+py+' -c '+shlex.quote(relay)],check=True)
code=f'''from pathlib import Path
import os,signal,json,time
run=Path({r!r})/'run-v2';row=json.loads((run/'learner-process.json').read_text());p=Path('/proc')/str(row['pid']);a=(p/'cmdline').read_bytes().split(b'\\0');assert b'assembly_policy.train' in a and str(run).encode() in a;start=(p/'stat').read_text().split()[21];os.kill(row['pid'],signal.SIGTERM);(run/'handoff-request.json').write_text(json.dumps(dict(pid=row['pid'],start=start,job=os.environ['SLURM_JOB_ID'],host=row['host'],time=time.time(),reason='Human requests matched24minimum-valid reset update'),indent=2)+'\\n');print('Requested checkpoint-safe stop; no lease cancellation')
'''
# Obtain actual current GPU config; no stale job/host guesses.
fetch='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 cat '+r+'/robotics-assembly-3.0/operational/wowrobo-chimera-20261107/cluster.json'
cfg=json.loads(subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',fetch],check=True,capture_output=True,text=True).stdout)
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid='+cfg['gpu_job']+' --overlap --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=none '+py+' -c '+shlex.quote(code)],check=True)
(HERE/'handoff-target.json').write_text(json.dumps(cfg,indent=2)+'\n')
