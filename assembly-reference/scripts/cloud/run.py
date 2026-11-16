"""Bounded native-simulation/CUDA-learner launcher with a fresh physics preflight."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from assembly_policy.so101.bank_paths import default_training_bank_directory


def run_trainer(command,environment,log):
    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    def stop(*_):
        if process.poll() is None:process.send_signal(signal.SIGTERM)
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        try:return process.wait()
        except KeyboardInterrupt:
            stop();return process.wait()
    finally:signal.signal(signal.SIGTERM,previous)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'configs/cloud_stress.json')
    p.add_argument('--bank-dir',type=Path,default=default_training_bank_directory())
    p.add_argument('--envs',type=int);p.add_argument('--threads',type=int);p.add_argument('--iterations',type=int);p.add_argument('--max-hours',type=float)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cfg=json.loads(a.profile.read_text())
    for key in ('envs','threads','iterations','max_hours'):
        if getattr(a,key) is not None:cfg[key]=getattr(a,key)
    if cfg.get('threads') is None:
        cfg['threads']=int(os.environ.get('SLURM_CPUS_PER_TASK',8))
    if any(not math.isfinite(cfg[k]) or cfg[k]<=0 for k in ('envs','threads','iterations','max_hours','steps')):raise ValueError('Positive resource/iteration budgets required')
    if cfg['physics']!='native' or cfg['device']!='cuda':raise ValueError('This handoff qualifies native CPU physics with one CUDA learner')
    if not (a.bank_dir/'manifest.json').exists():
        raise ValueError(f'Qualified bank missing at {a.bank_dir}; see docs/SUBTASK_CURRICULUM.md')
    from assembly_policy.so101.resets import check_bank,directory_signature
    check_bank(a.bank_dir)
    # Read only the immutable bank arrays needed to scale the declared SGS floor.
    import numpy as np
    count=sum(len(np.load(a.bank_dir/f'{task}_train.npz')['states']) for task in ('motor','horn','screw','retainer','yoke'))
    command=[sys.executable,'-m','assembly_policy.train','--suite','so101','--tasks',cfg['tasks'],'--bank-dir',str(a.bank_dir.resolve()),'--run-dir',str(a.run_dir.resolve()),'--device',cfg['device'],'--envs',str(cfg['envs']),'--steps',str(cfg['steps']),'--nthread',str(cfg['threads']),'--iters',str(cfg['iterations']),'--max-hours',str(cfg['max_hours']),'--save-every',str(cfg['save_every']),'--keep-every','1000000','--no-eval','--so101-max-contact-depth',str(cfg['contact_depth_m']),'--so101-stability',cfg['stability'],'--so101-safety-penalty',str(cfg['safety_penalty']),'--eps-q',str(cfg['sampler_floor_mass']/count),'--seed',str(cfg['seed'])]
    if cfg.get('review_seconds'):
        if cfg.get('allow_task_mechanics_failure',False):raise ValueError('Supervised local run requires strict task mechanics')
        from datetime import timedelta
        end=(datetime.now(timezone.utc)+timedelta(hours=cfg['max_hours'])).isoformat()
        command=[sys.executable,str(ROOT/'scripts/so101_overnight.py'),'--run-dir',str(a.run_dir.resolve()),
                 '--end-utc',end,'--bank-dir',str(a.bank_dir.resolve()),'--bank-signature',directory_signature(a.bank_dir),
                 '--preflight',str(a.run_dir.resolve()/'preflight.json'),'--envs',str(cfg['envs']),
                 '--threads',str(cfg['threads']),'--steps',str(cfg['steps']),'--iterations',str(cfg['iterations']),
                 '--save-every',str(cfg['save_every']),'--seed',str(cfg['seed']),
                 '--max-contact-depth',str(cfg['contact_depth_m']),'--stability',cfg['stability'],
                 '--safety-penalty',str(cfg['safety_penalty']),'--sampler-floor-mass',str(cfg['sampler_floor_mass']),
                 '--review-seconds',str(cfg['review_seconds']),
                 '--eval-timeout',str(cfg['evaluation_timeout_seconds']),
                 '--eval-trials',str(cfg['evaluation_trials_per_family'])]
    print(shlex.join(command),flush=True)
    if a.dry_run:return
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=False)
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MUJOCO_GL':'egl'}
    # Reserve a new output directory. Existing runs are resumed explicitly with
    # the recorded trainer command, never accidentally by rerunning this launcher.
    preflight=[sys.executable,'-m','scripts.so101_preflight','--bank-dir',str(a.bank_dir.resolve()),'--out',str(run/'preflight.json'),'--max-contact-depth',str(cfg['contact_depth_m']),'--stability',cfg['stability'],'--safety-penalty',str(cfg['safety_penalty'])]
    if cfg.get('allow_task_mechanics_failure',False):preflight.append('--allow-task-mechanics-failure')
    try:
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (run/'launch.json').write_text(json.dumps(dict(start_utc=datetime.now(timezone.utc).isoformat(),config=cfg,command=command,source=source,gpu=hardware),indent=2)+'\n')
        subprocess.run(preflight,cwd=ROOT,env=environment,check=True)
        with (run/'trainer.log').open('x') as log:
            code=run_trainer(command,environment,log)
        (run/'exit.json').write_text(json.dumps(dict(returncode=code,end_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        if code:raise SystemExit(code)
    except Exception as error:
        (run/'launch-error.txt').write_text(type(error).__name__+': '+str(error)+'\n');raise


if __name__=='__main__':main()
