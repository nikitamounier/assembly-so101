import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a');OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
assert os.environ.get('SLURM_JOB_ID')=='3077705'
manifest=ROOT/'artifacts_local/chimera/source-hashes.json';guard=json.loads(manifest.read_text());name='operational/chimera-20261106/profile_worker.py'
assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==guard[name]
data=(ROOT/'artifacts_local/chimera/profile-worker-safe-clock.py').read_bytes();compile(data,name,'exec');(ROOT/name).write_bytes(data);guard[name]=hashlib.sha256(data).hexdigest();manifest.write_text(json.dumps(guard,indent=2)+'\n')
row=json.loads((OUT/'safe-worker-step.json').read_text());pid=row['srun_pid'];path=Path('/proc')/str(pid)
args=(path/'cmdline').read_bytes().decode().split('\0');assert '--jobid=3598717' in args and str(ROOT/'operational/chimera-20261106/profile_worker_entry.py') in args
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+60
while path.exists() and (path/'stat').read_text().split()[2]!='Z':
 assert time.monotonic()<deadline,'Stop not completed; do not duplicate'
 time.sleep(.5)
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl')
log=open(OUT/'gpu-cpu-workers.log','a');p=subprocess.Popen(['srun','--jobid=3598717','--overlap','--exact','--gpus=0','--ntasks=2','--cpus-per-task=2',str(ROOT/'.venv/bin/python'),str(ROOT/'operational/chimera-20261106/profile_worker_entry.py'),'--run-dir',str(ROOT/'runs/chimera-v13-8192-20261106'),'--newer-than','0'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
(OUT/'safe-worker-step.json').write_text(json.dumps(dict(srun_pid=p.pid,job='3598717',ntasks=2,cpus_per_task=2,epoch=0))+'\n')
print('replacement diagnostic worker step',p.pid,flush=True)
p.wait()
