"""Keep the learner independent of per-node rollout-job renewals."""
import atexit
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from controller import ROOT,RUN,GPU,PYTHON,STATE,native,remaining,launch

POOL=ROOT/'artifacts_local/chimera/node-pool.json'

def state(pool,**fields):
    fields.update(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_job=GPU,cpu_jobs=[r['job_id'] for r in pool['jobs']],worker_count=pool['target_workers'],run=str(RUN),pool_generation=pool['generation'])
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(fields,indent=2)+'\n');temp.replace(STATE)

def latest_iteration():
    path=RUN/'log.jsonl'
    if not path.exists():return 0
    lines=path.read_text().splitlines()
    return json.loads(lines[-1])['it'] if lines else 0

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Coordinator requires an allocated CPU step'
    lock=open(STATE.with_suffix('.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pool=json.loads(POOL.read_text());assert all(r['cpus']==10 and r['workers']==5 for r in pool['jobs'])
    min_workers=min(5,pool['target_workers'])
    learner=None;stopping=False;learner_generation=None
    while True:
        jobs={job:native(job) for job in [GPU,*[r['job_id'] for r in pool['jobs']]]}
        running=sum(r['workers'] for r in pool['jobs'] if jobs[r['job_id']]['JobState']=='RUNNING')
        terminal=[job for job,row in jobs.items() if row['JobState'] in ('FAILED','CANCELLED','TIMEOUT','COMPLETED','OUT_OF_MEMORY')]
        if GPU in terminal:
            state(pool,phase='allocation_terminal',jobs=jobs,terminal_jobs=terminal);return 2
        if learner is None:
            if jobs[GPU]['JobState']=='RUNNING' and remaining(jobs[GPU])>900 and running>=min_workers:
                command=json.loads((RUN/'command.json').read_text())
                for flag,value in (('--remote-workers',str(pool['target_workers'])),('--accept-timeout','30'),('--min-conn-frac',str(min_workers/pool['target_workers'])),('--save-every','1')):command[command.index(flag)+1]=value
                (RUN/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                learner=launch(GPU,8,[PYTHON,str(ROOT/'operational/chimera-20261106/learner.py'),*command],RUN/'trainer.log');atexit.register(learner.terminate);stopping=False
                learner_generation=int(jobs[GPU].get('Restarts','0'))
                state(pool,phase='learner_starting',jobs=jobs,running_workers=running,learner_srun_pid=learner.pid,terminal_jobs=terminal)
            else:state(pool,phase='waiting_allocations',jobs=jobs,running_workers=running,terminal_jobs=terminal)
        elif learner.poll() is not None:
            code=learner.returncode
            gpu_requeued=(jobs[GPU]['JobState'] in ('PENDING','CONFIGURING','REQUEUED') or
                          (jobs[GPU].get('Restarts') is not None and
                           int(jobs[GPU]['Restarts'])!=learner_generation))
            if stopping or code==3 or gpu_requeued:
                learner=None;state(pool,phase='waiting_allocations',jobs=jobs,learner_exit=code,gpu_requeued=gpu_requeued,terminal_jobs=terminal)
            elif code==0 and latest_iteration()>=10000:
                state(pool,phase='complete',jobs=jobs,learner_exit=code);return 0
            else:
                state(pool,phase='learner_failed',jobs=jobs,learner_exit=code,terminal_jobs=terminal);return code or 2
        else:
            if not stopping and jobs[GPU]['JobState']=='RUNNING' and remaining(jobs[GPU])<900:
                stop=launch(GPU,1,[PYTHON,str(ROOT/'operational/chimera-20261106/learner.py'),'--stop'],RUN/'stop.log')
                code=stop.wait(timeout=60)
                if code:state(pool,phase='checkpoint_request_failed',jobs=jobs);return code
                stopping=True
            state(pool,phase='checkpoint_requested' if stopping else 'running',jobs=jobs,running_workers=running,learner_srun_pid=learner.pid,terminal_jobs=terminal)
        time.sleep(2)

if __name__=='__main__':sys.exit(main())
