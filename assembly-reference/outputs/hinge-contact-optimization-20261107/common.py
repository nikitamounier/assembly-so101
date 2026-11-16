"""Opt-in profiling of immutable accepted-v4 sources; no live process attachment."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(os.environ['PROFILE_ROOT'])
OUT = Path(os.environ['PROFILE_OUT'])
CANON = ROOT/'robotics-assembly-3.0'
SIGNATURE = '8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2)+'\n')

def backend(variant='baseline'):
    variant=os.environ['PROFILE_VARIANT']
    experiment=OUT.parent
    name='_wowrobo_training_native_backend'
    path=experiment/('backend-original.py' if variant=='original' else 'backend-exact.py')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    mod.SOURCE=experiment/variant
    original=mod.load_backend
    mod.load_backend=lambda manifest: original(manifest,OUT/'native')
    return mod

class Sampler:
    def __init__(self,path):
        self.path=Path(path);self.ident=threading.get_ident();self.stop=threading.Event();self.rows=[]
    def start(self):
        self.begin=time.perf_counter();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        previous=self.begin
        while not self.stop.wait(.02):
            at=time.perf_counter();f=sys._current_frames().get(self.ident);stack=[]
            while f:
                stack.append(f'{f.f_code.co_name} ({f.f_code.co_filename}:{f.f_code.co_firstlineno})');f=f.f_back
            self.rows.append((stack[::-1],at-previous));previous=at
    def finish(self):
        self.end=time.perf_counter();self.stop.set();self.thread.join()
        if self.rows:self.rows[-1]=(self.rows[-1][0],self.rows[-1][1]+self.end-self.begin-sum(x[1] for x in self.rows))
        frames=[];index={};samples=[];weights=[]
        for stack,w in self.rows:
            ids=[]
            for frame in stack:
                if frame not in index:index[frame]=len(frames);frames.append({'name':frame})
                ids.append(index[frame])
            samples.append(ids);weights.append(w)
        write(self.path,{'$schema':'https://www.speedscope.app/file-format-schema.json','shared':{'frames':frames},'profiles':[{'type':'sampled','name':self.path.stem+' Python wall stacks, 20 ms target; GIL scheduling bias possible','unit':'seconds','startValue':0,'endValue':self.end-self.begin,'samples':samples,'weights':weights}]})

def hardware():
    import socket
    affinity=sorted(os.sched_getaffinity(0));cores=[]
    for c in affinity:
        p=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        cores.append([int((p/'physical_package_id').read_text()),int((p/'core_id').read_text())])
    return dict(host=socket.gethostname(),pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),affinity=affinity,physical_cores=sorted(map(list,set(map(tuple,cores)))),logical_cpus=len(affinity),physical_core_count=len(set(map(tuple,cores))))
