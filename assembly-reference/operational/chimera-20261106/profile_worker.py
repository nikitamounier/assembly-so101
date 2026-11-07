"""Safe worker wall stacks, process CPU time and native MuJoCo phase timers.

No external attachment or asynchronous native unwinding. Two diagnostic workers
run on spare CPUs in the retained GPU allocation. Native phase timers use a C
monotonic-clock callback, avoiding Python callbacks/GIL inside the simulator.
"""
import ctypes
import functools
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import numpy as np
import mujoco
from multiprocessing.connection import Connection

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
CLOCK=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/artifacts_local/chimera/profiler-clock.so')
TOKEN=f'{os.environ["SLURM_JOB_ID"]}-{os.environ.get("SLURM_PROCID","0")}-{os.getpid()}'
base=dict(job=os.environ['SLURM_JOB_ID'],rank=os.environ.get('SLURM_PROCID'),pid=os.getpid(),host=socket.gethostname(),token=TOKEN)
(OUT/f'ready-{TOKEN}.json').write_text(json.dumps(base)+'\n')
capture=None
old_recv,old_send=Connection.recv,Connection.send
lib=ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))))
clocklib=ctypes.CDLL(str(CLOCK))
clockslot=ctypes.c_void_p.in_dll(lib,'mjcb_time')
clockptr=ctypes.cast(clocklib.assembly_profile_time,ctypes.c_void_p).value
TIMER_NAMES=[mujoco.mjtTimer(i).name for i in range(int(mujoco.mjtTimer.mjNTIMER))]


def timer_data(env):
    if hasattr(env,'work'):
        return [('rollout_last_trajectory',i,d) for i,d in enumerate(env.work)]+[('forward_env',i,d) for i,d in enumerate(env.data)]
    return [('task_data',f'{m.name}:{i}',d) for m in env.models for i,d in enumerate(m.datas)]


def timers(env):
    return [dict(kind=kind,index=i,duration=[float(t.duration) for t in d.timer],number=[int(t.number) for t in d.timer]) for kind,i,d in timer_data(env)]


class WorkerCapture:
    def __init__(self,path,env):
        self.path=Path(path);self.env=env;self.steps=0
        self.start=time.time();self.cpu=time.process_time();self.main=threading.get_ident()
        self.samples=[];self.phases=[];self.step_times=[];self.timer_steps=[]
        self.done=threading.Event();self.previous_clock=clockslot.value
        if self.previous_clock is None:clockslot.value=clockptr
        self.initial_timers=timers(env)
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()

    def sample(self):
        me=threading.get_ident()
        while not self.done.wait(.02):
            at=time.time()-self.start
            for ident,frame in sys._current_frames().items():
                if ident==me:continue
                stack=[]
                while frame:
                    c=frame.f_code
                    stack.append(f'{c.co_name} ({c.co_filename}:{c.co_firstlineno})')
                    frame=frame.f_back
                self.samples.append((at,ident,stack[::-1]))

    def stop(self):
        clockslot.value=self.previous_clock
        self.done.set();self.thread.join(5)
        meta={**base,'steps':self.steps,'start_utc':self.start,'end_utc':time.time(),'cpu_s':time.process_time()-self.cpu,'main_thread':self.main,'sampling':'20ms Python wall stacks, all Python threads; native C++ physics represented by call-site frames and MuJoCo phase timers, not native stack samples','phases':self.phases,'step_times':self.step_times,'timer_names':TIMER_NAMES,'initial_timers':self.initial_timers,'timer_steps':self.timer_steps}
        (self.path/f'worker-{TOKEN}.wall.json').write_text(json.dumps(self.samples)+'\n')
        target=self.path/f'worker-{TOKEN}.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(meta)+'\n');temp.replace(target)


def recv(self):
    global capture
    while True:
        msg=old_recv(self)
        if isinstance(msg,dict) and msg.get('cmd')=='profile_stop':
            if capture is not None:capture.stop();capture=None
            continue
        if isinstance(msg,dict) and msg.get('cmd')=='step':
            path=msg.get('profile_path')
            if path and capture is None:
                frame=sys._getframe(1)
                while frame is not None and not (frame.f_code.co_name=='_serve' and 'env' in frame.f_locals):frame=frame.f_back
                assert frame is not None
                capture=WorkerCapture(path,frame.f_locals['env'])
                del frame
            if capture is not None:
                capture.steps+=1;capture.step_start=time.perf_counter();capture.step_cpu=time.process_time()
        return msg


def send(self,msg):
    if isinstance(msg,dict) and 'proto' in msg and 'host' in msg:msg={**msg,'profile_worker':base}
    if capture is not None and isinstance(msg,dict) and 'ct' in msg:
        capture.step_times.append(dict(step=capture.steps,wall_s=time.perf_counter()-capture.step_start,cpu_s=time.process_time()-capture.step_cpu,reported_ct_s=float(msg['ct'])))
        capture.timer_steps.append(timers(capture.env))
    return old_send(self,msg)


def wrap(cls,name,label):
    original=getattr(cls,name)
    @functools.wraps(original)
    def call(self,*args,**kwargs):
        cap=capture
        if cap is None:return original(self,*args,**kwargs)
        wall=time.perf_counter();cpu=time.process_time()
        try:return original(self,*args,**kwargs)
        finally:cap.phases.append(dict(name=label,step=cap.steps,wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,thread=threading.get_ident()))
    setattr(cls,name,call)

from mujoco.rollout import Rollout
wrap(Rollout,'rollout','native_rollout')
if (ROOT/'assembly_policy/so101/env.py').exists():
    from assembly_policy.so101.env import BatchedAssemblyEnv as SOEnv
    for name in ['step','_refresh','assembly_distance','_check_material_success','set_states']:
        wrap(SOEnv,name,'so101.'+name)
else:
    from assembly_policy.sim.env import BatchedAssemblyEnv as NISTEnv
    for name in ['step','set_states','_work','assembly_distance']:
        if hasattr(NISTEnv,name):wrap(NISTEnv,name,'nist.'+name)
Connection.recv=recv;Connection.send=send
