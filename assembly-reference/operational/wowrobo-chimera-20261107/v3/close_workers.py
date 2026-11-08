from pathlib import Path
import os,json,subprocess,shlex,time
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).parent;old=json.loads((HERE.parent/'target.json').read_text());r=old['remote'];cfg=json.loads((HERE/'handoff-target.json').read_text())
code=f'''from pathlib import Path
import os,signal,json,time
run=Path({r!r})/'run-v2';assert (run/'handoff-closed.json').exists();job=os.environ['SLURM_JOB_ID'];closed=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if len(a)<3 or a[1:3]!=[b'-m',b'assembly_policy.rl.remote'] or str(run).encode() not in a:continue
  if f'SLURM_JOB_ID={{job}}'.encode() not in (p/'environ').read_bytes().split(b'\\0'):continue
  start=(p/'stat').read_text().split()[21];assert (p/'stat').read_text().split()[21]==start;os.kill(int(p.name),signal.SIGKILL);closed.append(dict(pid=int(p.name),start=start))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
time.sleep(.1);print(json.dumps(dict(job=job,closed=closed,leases_preserved=True)))
'''
# One allocated relay invocation dispatches steps; no SSH storm.
relay=f'''import subprocess,json,os
from concurrent.futures import ThreadPoolExecutor
code={code!r};jobs={cfg['cpu_jobs']!r};py={old['python']!r}
def go(job):
 p=subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--ntasks=1','--cpus-per-task=1','--cpu-bind=none',py,'-c',code],capture_output=True,text=True,timeout=40,env={{k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}});return dict(job=job,rc=p.returncode,stdout=p.stdout,stderr=p.stderr[-500:])
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(go,jobs))
from pathlib import Path
run=Path({r!r})/'run-v2';(run/'worker-handoff-closure.json').write_text(json.dumps(rows,indent=2)+'\\n');print(json.dumps(rows))
'''
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+old['python']+' -c '+shlex.quote(relay)],check=True,capture_output=True,text=True);(HERE/'worker-closures.json').write_text(p.stdout);print(p.stdout)
