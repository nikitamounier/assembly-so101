"""Resume original upstream reset generation by caching each original seed tile."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import sys
from multiprocessing import Pool
import numpy as np

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/assembly-nist-eb5d15e')
sys.path.insert(0,str(ROOT))
from scripts import build_banks as upstream
from assembly_policy.sim.env import EnvConfig
from assembly_policy.sim.tasks import resolve_tasks

original_work=upstream.work
BANK=ROOT/'data/banks'
TILES=BANK/'tiles'


def work(args):
    key=hashlib.sha256(json.dumps(args).encode()).hexdigest()
    path=TILES/(key+'.npz')
    if path.exists():
        z=np.load(path)
        return args[2],args[3],z['states'],z['levels'],int(z['tries']),float(z['elapsed'])
    fam,worker,states,levels,tries,elapsed=original_work(args)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,states=states,levels=levels,tries=tries,elapsed=elapsed)
    tmp.replace(path)
    return fam,worker,states,levels,tries,elapsed


def build_task(index, blocking=True):
    task=resolve_tasks('all20')[index]
    BANK.mkdir(parents=True,exist_ok=True);TILES.mkdir(exist_ok=True)
    lock=open(BANK/f'{task}.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        lock.close();return False
    if not (BANK/f'{task}_train.npz').exists():
        upstream.work=work
        with Pool(10) as pool:
            upstream.build(task,'train',640,pool,10,BANK,EnvConfig.fine_families)
    lock.close()
    print(json.dumps(dict(task=task,ready=True)),flush=True)
    return True


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    if sys.argv[1]=='sweep':
        # First five owners were already running before locks were introduced;
        # leave those alone. Reuse completed owners' idle allocated CPUs for
        # tasks 5..19, respecting locks if their original jobs start later.
        for index in range(5,20):build_task(index,blocking=False)
    else:build_task(int(sys.argv[1]))
