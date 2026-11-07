"""Bounded local CUDA learner supervisor with checkpoints and two-hour reviews.

No hardware or cloud operations. Create RUN_DIR/STOP for a graceful stop.
Numerical failures stop for review; unexpected process exits get at most two
checkpoint restarts. A falling loss is never treated as task success.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assembly_policy.so101.limits import PENETRATION_LIMIT_M


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def records(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue  # final line may still be flushing
    return result


def snapshot(run):
    rows=records(run/'log.jsonl')
    memory=psutil.virtual_memory();disk=shutil.disk_usage(run)
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                                     '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):gpu=None
    last=rows[-1] if rows else None
    return dict(time=datetime.now(timezone.utc).isoformat(),latest=last,iterations=len(rows),
                available_memory_gb=memory.available/2**30,disk_free_gb=disk.free/2**30,gpu_csv=gpu,
                log_age_s=time.time()-(run/'log.jsonl').stat().st_mtime if rows else None)


def parse_utc(value):
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0:
        raise ValueError('An explicit UTC offset is required')
    return parsed.timestamp()


def require_headroom(run):
    if shutil.disk_usage(run).free<2*2**30 or psutil.virtual_memory().available<2*2**30:
        raise ValueError('At least 2 GiB free disk and available host memory required')


def qualify_bank_layout(directory,train_per_family,eval_per_family):
    """Require all 25 task/family cells, independently in both splits."""
    for task in ('motor','horn','screw','retainer','yoke'):
        for split,count in (('train',train_per_family),('eval',eval_per_family)):
            path=directory/('family_eval' if split=='eval' else '')/f'{task}_{split}.npz'
            with np.load(path) as bank:
                families=bank['family']
                if len(bank['states'])!=5*count or any(np.count_nonzero(families==i)!=count for i in range(5)):
                    raise ValueError(f'Incomplete or unbalanced task/family bank: {path}')


def run_bounded(command,environment,log,deadline,stop_file=None):
    """Own the child process group, including initialization and evaluation."""
    if time.time()>=deadline:raise subprocess.TimeoutExpired(command,0)
    if stop_file is not None and stop_file.exists():raise InterruptedError('User STOP file')
    child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while child.poll() is None:
            if stop_file is not None and stop_file.exists():raise InterruptedError('User STOP file')
            if time.time()>=deadline:raise subprocess.TimeoutExpired(command,0)
            try:child.wait(timeout=min(.25,max(.01,deadline-time.time())))
            except subprocess.TimeoutExpired:pass
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def initialize_baseline(argv,trainer_factory=None):
    """Calibrate observations with one rollout, retaining an unupdated actor.

    This rollout updates normalization, SGS outcomes and transition accounting;
    no optimizer update occurs. The subsequent run resumes iteration zero and
    performs its separately declared PPO update budget.
    """
    import torch
    from assembly_policy.train import Trainer,parse
    trainer=(trainer_factory or Trainer)(parse(argv))
    baseline=trainer.run/'baseline_policy.pt'
    try:
        if trainer.it!=0 or baseline.exists():raise ValueError('Baseline calibration requires a fresh iteration-zero run')
        before={name:value.detach().cpu().clone() for name,value in trainer.model.named_parameters()}
        transitions_before=trainer.transitions
        normalizer_before=float(trainer.model.norm.count)
        batch,stats=trainer.collect()
        if stats.get('unstable',0):raise FloatingPointError('Unstable baseline calibration rollout')
        if any(not torch.isfinite(value).all() for value in batch.values()):
            raise FloatingPointError('Nonfinite baseline calibration rollout')
        if any(not torch.equal(before[name],value.detach().cpu()) for name,value in trainer.model.named_parameters()):
            raise ValueError('Baseline calibration modified model parameters')
        if trainer.it!=0 or trainer.opt.state:raise ValueError('Baseline calibration performed an optimizer update')
        if trainer.transitions<=transitions_before or float(trainer.model.norm.count)<=normalizer_before:
            raise ValueError('Baseline calibration produced no transitions or normalization observations')
        if any(not torch.isfinite(value).all() for value in trainer.model.state_dict().values()):
            raise FloatingPointError('Nonfinite calibrated baseline policy')
        trainer.save()
        # This policy is immutable; checkpoint.pt continues to roll during PPO.
        policy=dict(model=trainer.model.state_dict(),it=0,transitions=trainer.transitions,args=trainer.args)
        temporary=baseline.with_suffix('.pt.tmp')
        torch.save(policy,temporary);temporary.replace(baseline)
        report=dict(policy=str(baseline.resolve()),policy_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                    calibration_transitions=trainer.transitions-transitions_before,
                    total_transitions=trainer.transitions,normalizer_count_before=normalizer_before,
                    normalizer_count_after=float(trainer.model.norm.count),optimizer_updates=0,
                    model_parameters_unchanged=True,unstable=stats.get('unstable',0),
                    calibration_safety_episodes=stats.get('safety',0),
                    scope='One on-policy rollout fits normalization and updates SGS outcomes; discarded for PPO optimization')
        write_json(trainer.run/'baseline_calibration.json',report)
        return report
    finally:trainer.env.close()


def evaluation_workers(args):
    return min(getattr(args,'envs',1),getattr(args,'threads',1),8)


def identical_loose_evaluation(bank_dir,trials):
    """Prove the loose episodes have identical states and episode seed indices."""
    for task in ('motor','horn','screw','retainer','yoke'):
        try:
            with np.load(bank_dir/f'{task}_eval.npz') as loose, np.load(bank_dir/'family_eval'/f'{task}_eval.npz') as all_families:
                first=np.flatnonzero(loose['family']==0)[:trials]
                second=np.flatnonzero(all_families['family']==0)[:trials]
                # Evaluation seeds use task index + bank entry index, not the
                # stored generation seed. Both must match for this shortcut.
                if (not len(first) or not np.array_equal(first,second) or
                    not np.array_equal(loose['states'][first],all_families['states'][second]) or
                    not np.array_equal(loose['seed'][first],all_families['seed'][second])):
                    return False
        except (OSError,ValueError,KeyError):
            return False
    return True


def project_loose_evaluation(folder):
    source=folder/'all.json'
    result=json.loads(source.read_text())
    loose={**result,'families':['loose'],'tasks':{},
           'derived_from':str(source.resolve()),
           'derivation':'derived from all-family evaluation; identical states and per-entry seeds'}
    for task,values in result['tasks'].items():
        episodes=[episode for episode in values['episodes'] if episode['family']=='loose']
        loose['tasks'][task]={**values,'episodes':episodes,'trials':len(episodes),
                              'successes':sum(episode['success'] for episode in episodes)}
    write_json(folder/'loose.json',loose)


def evaluate_checkpoint(checkpoint,folder,args,environment):
    """Bound independent subtask-loose and all-family outcome checks."""
    folder.mkdir(parents=True,exist_ok=True)
    deadline=min(time.time()+args.eval_timeout,getattr(args,'deadline',float('inf')))
    reuse_loose=identical_loose_evaluation(args.bank_dir,args.eval_trials)
    for families in (('all',) if reuse_loose else ('loose','all')):
        out=folder/f'{families}.json'
        command=[sys.executable,'-m','assembly_policy.so101.evaluate','--policy',str(checkpoint),
                 '--bank-dir',str(args.bank_dir.resolve()),'--out',str(out),'--nthread','1',
                 '--families',families,'--trials',str(args.eval_trials),
                 '--workers',str(evaluation_workers(args))]
        try:
            with (folder/f'{families}.log').open('w') as log:
                run_bounded(command,environment,log,deadline,getattr(args,'stop_file',None))
            if reuse_loose:project_loose_evaluation(folder)
        except Exception as error:
            write_json(folder/'error.json',{'error':type(error).__name__+': '+str(error)})
            raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--end-utc',required=True,help='ISO8601 end time including +00:00')
    ap.add_argument('--envs',type=int,default=8)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--bank-dir',type=Path,required=True)
    ap.add_argument('--bank-signature',required=True,help='Expected directory_signature from qualification')
    ap.add_argument('--iterations',type=int,default=20)
    ap.add_argument('--save-every',type=int,default=1)
    ap.add_argument('--seed',type=int,default=20261107)
    ap.add_argument('--eval-timeout',type=int,default=1800)
    ap.add_argument('--eval-trials',type=int,default=1)
    ap.add_argument('--train-per-family',type=int,default=256)
    ap.add_argument('--eval-per-family',type=int,default=64)
    ap.add_argument('--init-policy',type=Path)
    ap.add_argument('--preflight',type=Path,default=ROOT/'artifacts_local/preflight.json')
    ap.add_argument('--max-contact-depth',type=float,default=PENETRATION_LIMIT_M)
    ap.add_argument('--stability',choices=['velocity','pose'],default='pose')
    ap.add_argument('--safety-penalty',type=float,default=1.)
    ap.add_argument('--physics',choices=['native','whole_helper_sdf'],default='native')
    ap.add_argument('--physics-model',type=Path)
    ap.add_argument('--review-seconds',type=int,default=7200)
    ap.add_argument('--first-review-utc',help='Optional first review ISO8601 time; subsequent reviews use --review-seconds')
    ap.add_argument('--sampler-floor-mass',type=float,default=1.6,
                    help='Aggregate SGS floor mass, divided by the exact TRAIN bank size')
    a=ap.parse_args()
    if any(getattr(a,k)<=0 for k in ('envs','threads','steps','iterations','save_every','review_seconds','eval_timeout','eval_trials','train_per_family','eval_per_family')):
        raise ValueError('Positive resource, update and evaluation budgets required')
    if a.threads>(os.cpu_count() or 1):raise ValueError('Simulation threads exceed logical CPUs')
    if a.physics!='native' or a.max_contact_depth!=PENETRATION_LIMIT_M or a.safety_penalty!=1. or a.stability!='pose':
        raise ValueError('Local run requires native physics, 0.5 mm guard, pose stability and safety penalty 1.0')
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'checkpoint.pt').exists():raise ValueError('Use a fresh run directory')
    require_headroom(run)
    lock=(run/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=parse_utc(a.end_utc)
    if end<=time.time():raise ValueError('End time is in the past')
    training_end=end-min(a.eval_timeout,(end-time.time())/4)
    a.deadline=training_end;a.stop_file=run/'STOP'
    from assembly_policy.so101.resets import signature,check_bank,directory_signature
    check_bank(a.bank_dir)
    if directory_signature(a.bank_dir)!=a.bank_signature:raise ValueError('Qualified bank signature differs')
    qualify_bank_layout(a.bank_dir,a.train_per_family,a.eval_per_family)
    bank_count=sum(len(np.load(a.bank_dir/f'{name}_train.npz')['states'])
                   for name in ('motor','horn','screw','retainer','yoke'))
    eps_q=a.sampler_floor_mass/bank_count
    preflight=json.loads(a.preflight.read_text())
    physics_signature=signature()
    if (preflight['model_signature']!=signature() or not preflight['checks'] or
        preflight.get('allow_task_mechanics_failure',False) or
        not preflight['checks'].get('thread_tracks_pitch',False) or not all(preflight['checks'].values()) or
        preflight.get('bank_directory_signature')!=directory_signature(a.bank_dir) or
        preflight.get('max_contact_depth')!=a.max_contact_depth or
        preflight.get('stability_metric','velocity')!=a.stability or
        preflight.get('safety_penalty',0.)!=a.safety_penalty or
        preflight.get('physics_backend','native')!=a.physics or
        preflight.get('physics_signature',preflight['model_signature'])!=physics_signature):
        raise ValueError('A passing preflight for the current model and exact bank is required')
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MUJOCO_GL':'egl'}
    process=None;restarts=0
    reviews=max((int(p.name) for p in (run/'reviews').glob('*') if p.is_dir() and p.name.isdigit()),default=0)
    next_review=parse_utc(a.first_review_utc) if a.first_review_utc else time.time()+a.review_seconds
    status={'supervisor_pid':os.getpid(),'run_dir':str(run),'end_utc':a.end_utc,
            'model_signature':signature(),'backend':a.physics+'_physics_cuda_ppo','state':'starting'}
    status['physics_signature']=physics_signature
    status['safety_penalty']=a.safety_penalty
    status['bank_dir']=str(a.bank_dir.resolve())
    status['bank_directory_signature']=directory_signature(a.bank_dir)
    status.update(bank_entries=bank_count,sampler_eps_q=eps_q,sampler_floor_mass=a.sampler_floor_mass)
    write_json(run/'supervisor.json',status)
    evaluations=[]
    status['update_budget']=a.iterations
    status['retention']='immutable normalized baseline policy; rolling checkpoint + atomic temporary + sampler; one final model; at most one temporary review snapshot'
    status['evaluation_timeout_seconds']=a.eval_timeout
    status['training_end_utc']=datetime.fromtimestamp(training_end,timezone.utc).isoformat()
    status['hard_end_includes_evaluation']=True
    def stop(reason):
        status['state']='stopping';status['reason']=reason;write_json(run/'supervisor.json',status)
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=max(.01,min(180,end-time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();status['forced_kill']=True
    def request_stop(*_):
        (run/'STOP').touch()
    previous_term=signal.signal(signal.SIGTERM,request_stop)
    previous_int=signal.signal(signal.SIGINT,request_stop)
    try:
        while time.time()<training_end and not (run/'STOP').exists():
            if process is None or process.poll() is not None:
                if process is not None:
                    status['previous_returncode']=process.returncode
                    prior_log=run/f'trainer_attempt_{restarts}.log'
                    if prior_log.exists() and 'FloatingPointError' in prior_log.read_text()[-32768:]:
                        status.update(state='needs_review',reason='Numerical failure requires diagnosis');break
                    if process.returncode==0:
                        status.update(state='completed',reason='Bounded trainer completed');break
                    restarts+=1
                    if restarts>2:
                        status.update(state='needs_review',reason='Repeated unexpected trainer exits');break
                command=[sys.executable,'-m','assembly_policy.train','--suite','so101','--run-dir',str(run),
                         '--bank-dir',str(a.bank_dir.resolve()),
                         '--envs',str(a.envs),'--nthread',str(a.threads),'--steps',str(a.steps),
                         '--tasks','all','--device','cuda','--iters',str(a.iterations),'--save-every',str(a.save_every),'--keep-every','1000000000',
                         '--bank-per-family','0','--seed',str(a.seed),
                         '--eps-q',str(eps_q),
                         '--no-eval','--max-hours',str(max(.01,(training_end-time.time())/3600))]
                if a.init_policy and restarts==0:command+=['--init-policy',str(a.init_policy.resolve())]
                command+=['--so101-stability',a.stability]
                command+=['--so101-safety-penalty',str(a.safety_penalty)]
                if a.max_contact_depth is not None:command+=['--so101-max-contact-depth',str(a.max_contact_depth)]
                command+=['--so101-physics',a.physics]
                if a.physics_model:command+=['--so101-physics-model',str(a.physics_model.resolve())]
                require_headroom(run)
                if restarts==0:
                    with (run/'initialization.log').open('w') as initial_log:
                        run_bounded([sys.executable,str(Path(__file__).resolve()),'--initialize-baseline',*command[3:]],environment,initial_log,
                                    min(training_end,time.time()+a.eval_timeout),run/'STOP')
                    status['baseline_calibration']=json.loads((run/'baseline_calibration.json').read_text())
                    status['baseline_policy']=str(run/'baseline_policy.pt')
                    write_json(run/'supervisor.json',status)
                    evaluate_checkpoint(run/'baseline_policy.pt',run/'initial',a,environment)
                    if time.time()>=training_end:raise ValueError('Initial evaluation exhausted run time budget')
                log=(run/f'trainer_attempt_{restarts}.log').open('a')
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                log.close();status.update(state='running',trainer_pid=process.pid,restarts=restarts,command=command)
                attempt_started=time.time()
                write_json(run/'supervisor.json',status)
            health=snapshot(run);health['trainer_pid']=process.pid
            write_json(run/'health.json',health)
            with (run/'health.jsonl').open('a') as f:f.write(json.dumps(health)+'\n')
            row=health['latest']
            failure=None
            if any(p.poll() not in (None,0) for p in evaluations):failure='Independent periodic evaluation failed'
            if row:
                for key in ('fps','pi','vf','ent','kl','std','lr'):
                    if not math.isfinite(row[key]):failure=f'Nonfinite {key}'
                if row.get('unstable',0)>0:failure='Numerical reset/nonfinite simulation state'
                if min(health['log_age_s'],time.time()-attempt_started)>1200:
                    failure='No completed learner iteration for twenty minutes'
            elif time.time()-attempt_started>1200:failure='No first learner iteration for twenty minutes'
            if health['available_memory_gb']<2:failure='Insufficient host memory headroom'
            if health['disk_free_gb']<2:failure='Insufficient checkpoint disk headroom'
            if failure:
                stop(failure);status.update(state='needs_review',reason=failure);break
            if time.time()>=next_review:
                reviews+=1;review=run/'reviews'/f'{reviews:02d}';review.mkdir(parents=True,exist_ok=True)
                write_json(review/'health.json',health)
                evaluations=[p for p in evaluations if p.poll() is None]
                if not evaluations and (run/'checkpoint.pt').exists():
                    require_headroom(run)
                    checkpoint=review/'checkpoint.pt';shutil.copy2(run/'checkpoint.pt',checkpoint)
                    log=(review/'evaluation.log').open('w')
                    worker=[sys.executable,str(Path(__file__).resolve()),'--evaluate-snapshot',str(checkpoint),
                            '--evaluation-bank',str(a.bank_dir.resolve()),'--evaluation-timeout',str(a.eval_timeout),
                            '--evaluation-trials',str(a.eval_trials),'--evaluation-workers',str(evaluation_workers(a)),
                            '--evaluation-deadline',str(training_end),
                            '--evaluation-stop-file',str(run/'STOP')]
                    p=subprocess.Popen(worker,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    log.close();evaluations.append(p)
                    status['evaluation_pids']=[p.pid for p in evaluations]
                else:write_json(review/'evaluation-skipped.json',{'reason':'Previous evaluation active or no checkpoint'})
                next_review+=a.review_seconds
                status['reviews_started']=reviews;write_json(run/'supervisor.json',status)
            time.sleep(30)
        if status['state'] not in ('needs_review','completed'):
            stop('User STOP file' if (run/'STOP').exists() else 'Overnight time boundary')
            status['state']='stopped'
    except BaseException as error:
        stop(type(error).__name__+': '+str(error))
        if isinstance(error,InterruptedError) and (run/'STOP').exists():status['state']='stopped'
        else:status['state']='needs_review';raise
    finally:
        for child in evaluations:
            if child.poll() is None:
                if (run/'STOP').exists() or status['state']=='needs_review':os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=a.eval_timeout+30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait()
        if status['state'] in ('stopped','completed') and (run/'checkpoint.pt').exists() and not (run/'STOP').exists():
            a.deadline=end
            try:evaluate_checkpoint(run/'checkpoint.pt',run/'final',a,environment)
            except Exception as error:status.update(state='needs_review',final_evaluation_error=str(error))
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(run/'supervisor.json',status)
        write_json(run/'final_health.json',snapshot(run))
        signal.signal(signal.SIGTERM,previous_term)
        signal.signal(signal.SIGINT,previous_int)
        lock.close()


if __name__=='__main__':
    if '--evaluate-snapshot' in sys.argv:
        worker=argparse.ArgumentParser()
        worker.add_argument('--evaluate-snapshot',type=Path,required=True)
        worker.add_argument('--evaluation-bank',type=Path,required=True)
        worker.add_argument('--evaluation-timeout',type=int,required=True)
        worker.add_argument('--evaluation-trials',type=int,required=True)
        worker.add_argument('--evaluation-workers',type=int,default=1)
        worker.add_argument('--evaluation-deadline',type=float,required=True)
        worker.add_argument('--evaluation-stop-file',type=Path,required=True)
        w=worker.parse_args()
        settings=argparse.Namespace(bank_dir=w.evaluation_bank,eval_timeout=w.evaluation_timeout,eval_trials=w.evaluation_trials,deadline=w.evaluation_deadline,stop_file=w.evaluation_stop_file,envs=w.evaluation_workers,threads=w.evaluation_workers)
        def worker_stop(*_):raise InterruptedError('Evaluation worker stopped')
        signal.signal(signal.SIGTERM,worker_stop)
        signal.signal(signal.SIGINT,worker_stop)
        try:evaluate_checkpoint(w.evaluate_snapshot,w.evaluate_snapshot.parent,settings,os.environ)
        finally:w.evaluate_snapshot.unlink(missing_ok=True)
    elif '--initialize-baseline' in sys.argv:
        initialize_baseline(sys.argv[2:])
    else:main()
