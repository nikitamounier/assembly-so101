"""Pack independently leased seed groups within one memory-sized Slurm job."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--indices',required=True);a=p.parse_args();r=a.root.resolve();indices=[int(x) for x in a.indices.split(',')];assert len(indices)==len(set(indices))
def run(i):
 return i,subprocess.run([sys.executable,str(r/'launch/partition_worker.py'),'--root',str(r),'--index',str(i)]).returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
 for i,code in pool.map(run,indices):print(json.dumps({'index':i,'exit':code,'unix':time.time()}),flush=True)
