"""Positively close all processes from the invalid startup, preserve every lease."""
from pathlib import Path
import os,sys,json,signal,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;OLD=ROOT/'run';PY='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
def matching(modules,run):
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if len(argv)<3 or argv[1]!=b'-m' or argv[2] not in modules or str(run).encode() not in argv:continue
   if f'SLURM_JOB_ID={os.environ["SLURM_JOB_ID"]}'.encode() not in (p/'environ').read_bytes().split(b'\0'):continue
   rows.append(dict(pid=int(p.name),start=(p/'stat').read_text().split()[21],argv=[x.decode() for x in argv if x]))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return rows
def clean():
 rows=matching([b'assembly_policy.train',b'assembly_policy.rl.remote'],OLD)
 for row in rows:
  p=Path('/proc')/str(row['pid']);assert (p/'stat').read_text().split()[21]==row['start'];os.kill(row['pid'],signal.SIGSTOP)
 if os.environ['SLURM_JOB_ID']=='3606264':
  for name in ['checkpoint.pt','log.jsonl','config.json']:
   p=OLD/name
   if p.exists():(OLD/(name+'.invalid-startup-preserved')).write_bytes(p.read_bytes())
 for row in rows:
  p=Path('/proc')/str(row['pid'])
  if p.exists() and (p/'stat').read_text().split()[21]==row['start']:os.kill(row['pid'],signal.SIGKILL)
 time.sleep(.2);remaining=matching([b'assembly_policy.train',b'assembly_policy.rl.remote'],OLD)
 receipt=dict(job=os.environ['SLURM_JOB_ID'],closed=rows,remaining=remaining)
 (ROOT/('startup-closure-'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(job=receipt['job'],closed=len(rows),remaining=len(remaining))))
 assert not remaining
if sys.argv[1]=='clean':clean();sys.exit()
# Freeze only the old controller; no batch or holding process is touched.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0')
  if p.stat().st_uid==os.getuid() and str(HERE/'cluster.py').encode() in argv and b'control' in argv:os.kill(int(p.name),signal.SIGKILL)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
cfg=json.loads((HERE/'cluster.json').read_text());jobs=[cfg['gpu_job'],*cfg['cpu_jobs']]
from concurrent.futures import ThreadPoolExecutor
def go(job):
 p=subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--ntasks=1','--cpus-per-task=1','--cpu-bind=none',PY,str(Path(__file__)),'clean'],capture_output=True,text=True,timeout=45,env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))})
 return dict(job=job,rc=p.returncode,stdout=p.stdout,stderr=p.stderr[-600:])
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(go,jobs))
(ROOT/('startup-closure-all-'+str(int(time.time()))+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
(ROOT/'startup-closure-all.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
(OLD/'INVALIDATED.json').write_text(json.dumps(dict(reason='Overlapping startup learners survived launcher SIGTERM; original logs/checkpoints preserved, no continuation from this run',closed_at=time.time()),indent=2)+'\n')
