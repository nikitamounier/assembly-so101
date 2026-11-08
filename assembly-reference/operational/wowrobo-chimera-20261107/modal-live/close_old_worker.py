from pathlib import Path
import os,signal,json,time
RUN=RUN_TOKEN
def owned():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().split(b'\0')
   if len(args)<3 or args[1:3]!=[b'-m',b'assembly_policy.rl.remote'] or RUN.encode() not in args:continue
   if ('SLURM_JOB_ID='+os.environ['SLURM_JOB_ID']).encode() not in (p/'environ').read_bytes().split(b'\0'):continue
   result.append(dict(pid=int(p.name),start=(p/'stat').read_text().split()[21]))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return result
closed=owned()
for row in closed:
 p=Path('/proc')/str(row['pid'])
 try:
  assert (p/'stat').read_text().split()[21]==row['start'];os.kill(row['pid'],signal.SIGKILL)
 except ProcessLookupError:pass
time.sleep(.2);remaining=owned();assert not remaining,remaining
print(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],closed=closed,remaining=remaining,leases_preserved=True)))
