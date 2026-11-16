"""Partition only the frozen generator's slot iteration; preserve targets/seeds/gates."""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path

def atomic(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def selected(n,part,parts):
    return range(part,n,parts)

def seed_receipts(master,out,part,parts):
    for f in (master/'attempts').glob('*.json'):
        x=json.loads(f.read_text())
        if x['slot']%parts!=part:continue
        for key in ('archive','physical_input_archive','qualified_archive'):
            if x.get(key):
                src=master/x[key];dst=out/x[key];dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():shutil.copy2(src,dst)
        dst=out/'attempts'/f.name;dst.parent.mkdir(exist_ok=True)
        if not dst.exists():shutil.copy2(f,dst)

def run_partition(root, stage,part,parts,attempts,publish_only=False):
    project=root/'robotics-assembly-3.0';plan=project/'outputs/reset-bank-1800-20261107-v4'
    out=plan/'shards'/stage if publish_only else plan/'partitions'/stage/str(part)
    out.mkdir(parents=True,exist_ok=True)
    source=project/'scripts/build_wowrobo_randomized_bank.py'
    tree=ast.parse(source.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='range']
    assert len(calls)==2, 'Frozen generator loop layout changed'
    assert any(isinstance(n.args[0],ast.Subscript) and ast.unparse(n.args[0])=='target[split]' for n in calls)
    assert attempts in (20,40,80,160)
    spec=importlib.util.spec_from_file_location('frozen_partition_generator',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Both calls are isolated here: slot counts60/12 vs attempt limits20/40/80/160.
    # Keep original global slot indices, full quota, RNG seed mapping and identity.
    module.range=lambda n: (() if publish_only else selected(n,part,parts)) if n in (60,12) else range(n)
    sys.argv=[str(source),'--config',str(plan/'config.json'),'--output',str(out),'--stages',stage,'--validation','physical','--max-attempts',str(attempts),'--physical-profile',str(plan/'physical'),'--settle-seconds','.08','--hold-seconds','.2','--full']
    return module.main()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--stage');p.add_argument('--part',type=int);p.add_argument('--parts',type=int,default=4);p.add_argument('--attempts',type=int,default=20);p.add_argument('--child',action='store_true');p.add_argument('--publish-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    project=root/'robotics-assembly-3.0';sys.path.insert(0,str(project));os.chdir(project)
    if a.child or a.publish_only:return run_partition(root,a.stage,a.part,a.parts,a.attempts,a.publish_only)
    assignment=json.loads((root/'launch/partition-assignment.json').read_text());assert assignment['open'];item=assignment['items'][a.index];stage=item['stage'];part=item['part'];parts=item['parts'];plan=project/'outputs/reset-bank-1800-20261107-v4';out=plan/'partitions'/stage/str(part);out.mkdir(parents=True,exist_ok=True)
    from farm_control import control_lock
    with (out/'.worker.lock').open('a') as lock:
        with control_lock(root):
            assert json.loads((root/'launch/partition-assignment.json').read_text())['open']
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.set_inheritable(lock.fileno(),True)
        seed_receipts(plan/'shards'/stage,out,part,parts)
        base=dict(item,job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,pid=os.getpid(),generator_sha256=hashlib.sha256((project/'scripts/build_wowrobo_randomized_bank.py').read_bytes()).hexdigest())
        wanted={(split,i) for split,n in [('train',60),('eval',12)] for i in selected(n,part,parts)}
        for attempts in (20,40,80,160):
            atomic(out/'partition-worker.json',dict(base,status='running',unix=time.time(),attempt_limit=attempts))
            cmd=[sys.executable,__file__,'--root',str(root),'--child','--stage',stage,'--part',str(part),'--parts',str(parts),'--attempts',str(attempts)]
            with (out/'generation.log').open('a') as log:code=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),)).returncode
            run_path=out/'run.json'
            master_run=plan/'shards'/stage/'run.json'
            if run_path.exists() and master_run.exists():
                assert json.loads(run_path.read_text())['identity'] == json.loads(master_run.read_text())['identity'], 'Source/model identity changed'
            rows=[json.loads(x.read_text()) for x in (out/'attempts').glob('*.json')];have={(x['split'],x['slot']) for x in rows if x.get('training_eligible')}
            accepted = [x for x in rows if x.get("training_eligible")]
            assert have <= wanted, "Partition wrote an unowned slot"
            assert len(accepted) == len(have), "Duplicate admitted slot"
            if wanted == have:
                atomic(out/'partition-worker.json',dict(base,status='complete',accepted=len(wanted),unix=time.time()));return 0
            if code not in (0,2):
                atomic(out/'partition-worker.json',dict(base,status='error',exit=code,unix=time.time()));return code
        atomic(out/'partition-worker.json',dict(base,status='incomplete',missing=sorted(wanted-have),unix=time.time()));return 2

if __name__=='__main__':raise SystemExit(main())
