"""User-approved idle-GPU failover; no concurrent learner, no allocation changes."""
from pathlib import Path
import os,json,fcntl,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;RUN=ROOT/'run-v2'
assert os.environ.get('SLURM_JOB_ID')
def call(args):return subprocess.run(args,capture_output=True,text=True,timeout=25)
def state(job):
 p=call(['squeue','-j',job,'-h','-o','%T']);assert p.returncode==0,p.stderr;return p.stdout.strip()
lock=(RUN/'gpu-failover.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=HERE/'cluster.json';cfg=json.loads(path.read_text());current=cfg['gpu_job'];assert current in ('3606264','3598717');candidate='3598717' if current=='3606264' else '3606264'
if state(current)=='RUNNING':print(json.dumps(dict(action='keep_monitor',gpu_job=current)));raise SystemExit()
owner=(RUN/'learner.lock').open('a')
try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print(json.dumps(dict(action='keep_monitor',reason='Current learner still owns lock')));raise SystemExit()
if state(candidate)!='RUNNING':print(json.dumps(dict(action='keep_monitor',reason='No running approved fallback')));raise SystemExit()
p=call(['srun','--jobid='+candidate,'--overlap','--exact','--ntasks=1','--cpus-per-task=1','--cpu-bind=none','nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
if p.returncode or p.stdout.strip():print(json.dumps(dict(action='keep_monitor',reason='Fallback not positively idle',returncode=p.returncode)));raise SystemExit()
assert state(current)!='RUNNING' and state(candidate)=='RUNNING'
cfg['gpu_job']=candidate;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(cfg,indent=2)+'\n');tmp.replace(path)
result=dict(action='gpu_failover',old_gpu_job=current,new_gpu_job=candidate,time=time.time(),learner_lock_exclusive=True,idle_probe=p.stdout,authorization='Human approved retainedGPU3598717 reuse on2026-11-07; same one Chimera-only run')
(RUN/('gpu-failover-'+str(time.time_ns())+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
