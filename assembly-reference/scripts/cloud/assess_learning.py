"""Compare unupdated/final mean policies on identical independent native EVAL starts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from assembly_policy.so101.workcell import default_bank_directory
TASKS=('motor','horn','screw','retainer','yoke')


def summarize(results):
    summary={}
    for phase in ('initial','final'):
        episodes=[e for r in results if r['phase']==phase for e in r['episodes']]
        n=len(episodes)
        if not n:raise ValueError('Missing assessment episodes')
        summary[phase]=dict(episodes=n,successes=sum(e['success'] for e in episodes),safety_stops=sum(e['safety'] for e in episodes),timeouts=sum(e['timeout'] for e in episodes),mean_best_error_m=sum(e['best_error_m'] for e in episodes)/n)
    before,after=summary['initial'],summary['final']
    if before['episodes']!=after['episodes']:raise ValueError('Mismatched assessment sizes')
    summary['task_improvement']=('demonstrated_on_this_cohort' if after['successes']>before['successes'] and after['safety_stops']<=before['safety_stops'] else 'not_demonstrated_on_this_cohort')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--bank-dir',type=Path,default=default_bank_directory())
    p.add_argument('--trials',type=int,default=16)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--tasks',default='+'.join(TASKS))
    a=p.parse_args();tasks=a.tasks.split('+')
    if a.trials<1 or a.workers<1 or not set(tasks)<=set(TASKS):raise ValueError('Positive trials/workers and known tasks required')
    run=a.run_dir.resolve();out=run/'learning-assessment';out.mkdir(exist_ok=False)
    policies={'initial':run/'initial_policy.pt','final':run/'checkpoint.pt'}
    for path in policies.values():
        if not path.is_file():raise FileNotFoundError(f'Missing policy snapshot: {path}')
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def evaluate(item):
        phase,task=item;result=out/f'{phase}-{task}.json'
        command=[sys.executable,'-m','assembly_policy.so101.evaluate','--policy',str(policies[phase]),'--bank-dir',str(a.bank_dir.resolve()),'--tasks',task,'--trials',str(a.trials),'--nthread','1','--action-mode','mean','--seed','501','--out',str(result)]
        with (out/f'{phase}-{task}.log').open('x') as log:subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=json.loads(result.read_text());return dict(phase=phase,task=task,**data['tasks'][task])
    with ThreadPoolExecutor(max_workers=min(a.workers,2*len(tasks))) as pool:
        results=list(pool.map(evaluate,[(phase,task) for phase in policies for task in tasks]))
    report=dict(scope='Same independent EVAL states, mean actions; finite sample task assessment, not generalization proof',tasks=results,summary=summarize(results))
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()
