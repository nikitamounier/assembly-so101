"""Allocated, restartable-by-manifest original-NIST profiling coordinator."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ.get('SLURM_JOB_ID')=='3077705'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/assembly-nist-eb5d15e')
PYTHON='/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv/bin/python'
OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
MANIFEST=ROOT/'artifacts_local/chimera/node-pool.json'
lock=open(OUT/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_')) and k!='CUDA_VISIBLE_DEVICES'}
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
learner=None


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def native(job):
    p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
    d=dict(x.split('=',1) for x in p.stdout.split() if '=' in x)
    return {k:d.get(k) for k in ['JobState','NodeList','Restarts','StartTime','EndTime']}


def remaining(row):
    try:return time.mktime(time.strptime(row['EndTime'],'%Y-%m-%dT%H:%M:%S'))-time.time()
    except (ValueError,KeyError):return 0


while True:
    pool=json.loads(MANIFEST.read_text())
    jobs={r['job_id']:native(r['job_id']) for r in pool['jobs']}
    banks=list((ROOT/'data/banks').glob('*_train.npz'))
    terminal=[j for j,r in jobs.items() if r['JobState'] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','COMPLETED']]
    phase='building_banks'
    if terminal:
        write(OUT/'controller.json',dict(phase='needs_attention',terminal_jobs=terminal,jobs=jobs,banks=len(banks),time=time.time()))
        raise RuntimeError(f'Baseline CPU terminal: {terminal}')
    all_running=all(r['JobState']=='RUNNING' for r in jobs.values())
    if len(banks)==20 and all_running:
        cfg=OUT/'worker-config.json'
        if not cfg.exists():
            selected=[];hosts=set()
            for job,row in sorted(jobs.items(),key=lambda kv:(not kv[1]['NodeList'].startswith('CPU'),kv[0])):
                if row['NodeList'] not in hosts:selected.append(job);hosts.add(row['NodeList'])
                if len(selected)==3:break
            if len(selected)<3:selected=list(jobs)[:3]
            write(cfg,dict(jobs=selected,expected_workers=3))
        (ROOT/'artifacts_local/chimera/ALL_BANKS_READY').touch()
        if pool['gpu_job'] is None:
            p=subprocess.run(['sbatch','--parsable','--job-name=nist-profile-gpu',f'--output={ROOT}/runs/profile/gpu-hold.log',str(ROOT/'operational/baseline_gpu.sbatch')],env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
            assert p.returncode==0,p.stderr
            job=p.stdout.strip().split(';')[0];assert job.isdigit(),p.stdout
            pool['gpu_job']=job;write(MANIFEST,pool)
        phase='waiting_gpu'
    if pool['gpu_job']:
        gpu=native(pool['gpu_job']);jobs[pool['gpu_job']]=gpu
        if learner is None and len(banks)==20 and all_running and gpu['JobState']=='RUNNING' and remaining(gpu)>900:
            run=ROOT/'runs/profile'
            command=[PYTHON,str(ROOT/'operational/profile_train.py'),'assembly_policy.train','--run-dir',str(run),'--tasks','all20','--bank-dir',str(ROOT/'data/banks'),'--device','cuda','--envs','8192','--steps','32','--iters','10000','--remote-workers','200','--accept-timeout','600','--min-conn-frac','1','--nthread','8','--save-every','1','--keep-every','1000000','--no-eval','--seed','20261105']
            write(run/'command.json',command);write(OUT/'original-command.json',[PYTHON,'-m','assembly_policy.train',*command[3:]])
            log=open(run/'trainer.log','a')
            learner=subprocess.Popen(['srun',f'--jobid={pool["gpu_job"]}','--overlap','--exact','--ntasks=1','--cpus-per-task=8',*command],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            phase='learner_starting'
        if learner is not None:
            phase='running'
            ready=[json.loads(p.read_text()) for p in OUT.glob('ready-*.json')]
            cfg=json.loads((OUT/'worker-config.json').read_text())
            if all(any(r['job']==job for r in ready) for job in cfg['jobs']):
                (OUT/'workers-ready').touch()
            status=OUT/'status.json'
            capture=json.loads(status.read_text()) if status.exists() else {}
            if capture.get('qualified') is True:
                assert learner.wait(timeout=120)==0
                for job in jobs:
                    p=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=30)
                    assert p.returncode==0,(job,p.stderr)
                write(OUT/'controller.json',dict(phase='complete',capture=capture,released_jobs=list(jobs),time=time.time()))
                break
            if learner.poll() is not None:
                write(OUT/'controller.json',dict(phase='learner_failed',exit_code=learner.returncode,jobs=jobs,capture=capture,time=time.time()))
                raise RuntimeError('Baseline learner failed; inspect checkpoint before retry')
            if gpu['JobState']=='RUNNING' and remaining(gpu)<600:
                raise RuntimeError('Baseline GPU renewal near; preserve checkpoint and inspect')
    write(OUT/'controller.json',dict(phase=phase,jobs=jobs,banks=len(banks),tiles=len(list((ROOT/'data/banks/tiles').glob('*.npz'))),time=time.time(),learner_srun_pid=learner.pid if learner else None))
    time.sleep(10)
