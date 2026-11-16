"""Two CPU-only diagnostic workers in spare retained-GPU allocation CPUs."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--run-dir');p.add_argument('--newer-than',type=float,required=True);a=p.parse_args()
import profile_worker
path=Path(a.run_dir)/'server.json'
while True:
    try:
        if json.loads(path.read_text())['time']>a.newer_than:break
    except (FileNotFoundError,ValueError):pass
    time.sleep(.01)
from assembly_policy.rl.remote import worker_main
worker_main(a.run_dir,2,7200,.1)
