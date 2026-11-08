from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
code=f'''from pathlib import Path
import os,fcntl,json
root=Path({r!r});run=root/'run-v2';job=os.environ['SLURM_JOB_ID'];module=b'assembly_policy.train' if job=='3606264' else b'assembly_policy.rl.remote';matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().split(b'\\0')
  if len(argv)>2 and argv[1:3]==[b'-m',module] and str(run).encode() in argv and f'SLURM_JOB_ID={{job}}'.encode() in (p/'environ').read_bytes().split(b'\\0'):
   matches.append(dict(pid=int(p.name),start=(p/'stat').read_text().split()[21],lock_descriptors=[str(f.resolve()) for f in (p/'fd').iterdir() if str(f.resolve()).endswith('.lock')]))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(matches)==1,matches
path=run/('learner.lock' if module==b'assembly_policy.train' else f'worker-{{job}}.lock');f=path.open('a');blocked=False
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:blocked=True
assert blocked,'Ownership lock did not survive exec'
result=dict(job=job,exact_process_count=len(matches),processes=matches,duplicate_lock_rejected=blocked);(root/f'ownership-proof-{{job}}.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
'''
for job in ['3606264','3602598']:
 subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid='+job+' --overlap --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=none '+c['python']+' -c '+shlex.quote(code)],check=True)
