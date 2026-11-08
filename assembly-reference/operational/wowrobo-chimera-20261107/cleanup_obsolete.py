"""Retire only workers from the superseded five-process diagnostic launch."""
import json,subprocess,shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());jobs=json.loads((h/'cluster.json').read_text())['cpu_jobs'];r=c['remote']
code=f'''from pathlib import Path
import os,signal,json
run={r+'/run'!r}.encode();job=os.environ['SLURM_JOB_ID'];gone=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().split(b'\\0');env=(p/'environ').read_bytes().split(b'\\0')
  if len(argv)<3 or argv[1:3]!=[b'-m',b'assembly_policy.rl.remote'] or run not in argv:continue
  if f'SLURM_JOB_ID={{job}}'.encode() not in env:continue
  i=argv.index(b'--nthread')
  if argv[i+1]!=b'2':continue
  start=(p/'stat').read_text().split()[21]
  if (p/'stat').read_text().split()[21]!=start:continue
  os.kill(int(p.name),signal.SIGKILL);gone.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(dict(job=job,obsolete_owned_workers_killed=gone)))
'''
def go(job):
 cmd='srun --jobid='+job+' --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)
 p=subprocess.run(['ssh','chimera-login',cmd],text=True,capture_output=True,timeout=30)
 return dict(job=job,rc=p.returncode,out=p.stdout,err=p.stderr[-400:])
with ThreadPoolExecutor(max_workers=8) as pool:
 rows=list(pool.map(go,jobs))
(h/'obsolete-worker-cleanup.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
