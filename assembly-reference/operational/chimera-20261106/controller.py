"""Coordinate retained allocations; all execution belongs inside a Slurm CPU step."""
import fcntl
import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
RUN=ROOT/'runs/chimera-v13-8192-20261106'
GPU='3598717'
CPU='3598730'
PYTHON=str(ROOT/'.venv/bin/python')
STATE=ROOT/'artifacts_local/chimera/controller.json'
WORKERS=250

def write(**fields):
    fields.update(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_job=GPU,cpu_job=CPU,worker_count=WORKERS,run=str(RUN))
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(fields,indent=2)+'\n');temporary.replace(STATE)

def native(job):
    result=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
    if result.returncode: return {'JobState':'UNKNOWN','error':result.stderr.strip()}
    return dict(field.split('=',1) for field in result.stdout.split() if '=' in field)

def remaining(row):
    try:return time.mktime(time.strptime(row['EndTime'],'%Y-%m-%dT%H:%M:%S'))-time.time()
    except (ValueError,KeyError):return 0

def launch(job,cpus,command,log,ntasks=1):
    stream=open(log,'a')
    stream.write('[assembly-step] '+json.dumps(dict(time=time.time(),job=job,ntasks=ntasks,cpus_per_task=cpus))+'\n');stream.flush()
    return subprocess.Popen(['srun',f'--jobid={job}','--overlap','--exact',f'--ntasks={ntasks}',f'--cpus-per-task={cpus}',*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**{k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'},'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MUJOCO_GL':'egl'})

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Run this coordinator in an allocated srun step'
    STATE.parent.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    lock=open(STATE.with_suffix('.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    learner=workers=None;stopping=False
    while True:
        jobs={job:native(job) for job in (GPU,CPU)}
        if not (STATE.parent/'READY').exists():
            write(phase='waiting_preflight',jobs=jobs)
        elif any(row['JobState'] in ('FAILED','CANCELLED','TIMEOUT','COMPLETED','OUT_OF_MEMORY') for row in jobs.values()):
            write(phase='allocation_terminal',jobs=jobs);return 2
        elif learner is None:
            if all(row['JobState']=='RUNNING' and remaining(row)>900 for row in jobs.values()):
                if workers is None or workers.poll() is not None:
                    workers=launch(CPU,2,[PYTHON,'-m','assembly_policy.rl.remote','--run-dir',str(RUN),'--nthread','2','--wait','7200','--poll','5'],RUN/'workers.log',WORKERS)
                    atexit.register(workers.terminate)
                command=[PYTHON,'-m','assembly_policy.train','--suite','so101','--tasks','all','--bank-dir',str(ROOT/'data/so101/banks_shared_v13'),'--run-dir',str(RUN),'--device','cuda','--envs','8192','--steps','32','--iters','10000','--remote-workers',str(WORKERS),'--accept-timeout','900','--min-conn-frac','1','--nthread','8','--max-hours','1000000000','--save-every','1','--keep-every','1000000','--no-eval','--so101-max-contact-depth','.0004','--so101-stability','pose','--so101-safety-penalty','1','--eps-q','.01','--seed','20261105']
                (RUN/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                learner=launch(GPU,8,[PYTHON,str(ROOT/'operational/chimera-20261106/learner.py'),*command],RUN/'trainer.log')
                atexit.register(learner.terminate)
                stopping=False
                write(phase='learner_starting',jobs=jobs,learner_srun_pid=learner.pid,worker_srun_pid=workers.pid)
            else:write(phase='waiting_allocations',jobs=jobs)
        elif learner.poll() is not None:
            code=learner.returncode
            if stopping:
                learner=None;write(phase='lease_checkpointed',jobs=jobs,learner_exit=code)
            elif code==3:
                if workers is not None and workers.poll() is None:
                    workers.terminate();workers.wait(timeout=60)
                workers=None;learner=None
                write(phase='waiting_allocations',jobs=jobs,learner_exit=code)
            elif code==0:
                write(phase='complete',jobs=jobs,learner_exit=code);return 0
            else:
                write(phase='learner_failed',jobs=jobs,learner_exit=code);return code
        else:
            if not stopping and any(row['JobState']=='RUNNING' and remaining(row)<900 for row in jobs.values()):
                stop=launch(GPU,1,[PYTHON,str(ROOT/'operational/chimera-20261106/learner.py'),'--stop'],RUN/'stop.log')
                code=stop.wait(timeout=60)
                if code:write(phase='checkpoint_request_failed',jobs=jobs);return code
                stopping=True
            write(phase='checkpoint_requested' if stopping else 'running',jobs=jobs,learner_srun_pid=learner.pid,worker_srun_pid=workers.pid,worker_step_exit=workers.poll())
        time.sleep(20)

if __name__=='__main__':sys.exit(main())
