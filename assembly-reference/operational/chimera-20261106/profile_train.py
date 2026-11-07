"""Opt-in observational wrappers; PPO/physics methods are invoked unchanged.

Launch: python profile_train.py assembly_policy.train <original trainer flags>.
One warmup iteration, then up to four candidates. Export only after checkpoint.
The wall sampler observes Python stacks, including waits; it is NOT native CPU
sampling and cannot resolve MuJoCo internals in other processes.
"""
import cProfile
import functools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from assembly_policy import train

OUT = ROOT / 'artifacts_local/chimera/ppo-profile-20261106'
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') == '3598717'
assert sys.argv.pop(1) == 'assembly_policy.train'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def jobs():
    pool = json.loads((ROOT / 'artifacts_local/chimera/node-pool.json').read_text())
    result = {}
    for job in pool.get('profile_native_jobs', ['3598717', *[str(r['job_id']) for r in pool['jobs']]]):
        p = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True, text=True, timeout=20)
        d = dict(s.split('=', 1) for s in p.stdout.split() if '=' in s)
        result[job] = {k: d.get(k) for k in ['JobState', 'NodeList', 'Restarts', 'StartTime', 'EndTime']}
    return result


class Capture:
    def __init__(self, tr, attempt):
        self.tr = tr
        self.path = OUT / f'attempt-{attempt}-iteration-{tr.it+1}'
        self.path.mkdir(exist_ok=True)
        self.before = jobs()
        self.start_it, self.start_transitions = tr.it, tr.transitions
        self.connections = tuple(tr.env.conn_of)
        self.worker_config = json.loads((OUT/'worker-config.json').read_text())
        self.main = threading.get_ident()
        self.samples, self.steps, self.spans, self.receives = [], [], [], []
        self.done = threading.Event()
        self.start = time.perf_counter()
        self.start_utc = time.time()
        self.cp = cProfile.Profile()
        self.tp = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True)
        self.tp.start()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        self.cp.enable()
        tr.env.reset_profile()
        write(OUT / 'status.json', dict(phase='capturing', iteration=tr.it+1, path=str(self.path), start=self.start_utc))

    def sample(self):
        while not self.done.wait(.02):
            frame = sys._current_frames().get(self.main)
            stack = []
            while frame:
                code = frame.f_code
                stack.append(f'{code.co_name} ({code.co_filename}:{code.co_firstlineno})')
                frame = frame.f_back
            self.samples.append((time.perf_counter()-self.start, stack[::-1]))

    def finish(self):
        self.cp.disable()
        self.end = time.perf_counter()
        self.done.set()
        self.thread.join(5)
        self.tp.stop()
        self.cp.dump_stats(str(self.path / 'python.pstats'))
        self.tp.export_chrome_trace(str(self.path / 'torch-trace.json'))
        for conn in self.connections:
            try:
                conn.send(dict(cmd='profile_stop', profile_path=str(self.path)))
            except (OSError,EOFError):
                pass
        deadline = time.monotonic()+30
        worker_results = []
        while time.monotonic()<deadline:
            worker_results = [json.loads(p.read_text()) for p in self.path.glob('worker-*.json') if not p.name.endswith('.wall.json')]
            if len(worker_results)>=self.worker_config['expected_workers']:
                break
            time.sleep(.2)
        after = jobs()
        tr = self.tr
        full = tr.transitions-self.start_transitions == tr.a.envs*tr.a.steps
        stable = len(self.steps) == tr.a.steps and all(s['stable'] and s['active']==tr.a.envs for s in self.steps)
        native = self.before == after and all(r['JobState']=='RUNNING' for r in after.values())
        workers_ok = len(worker_results)==self.worker_config['expected_workers'] and all(w['steps']==tr.a.steps for w in worker_results)
        qualified = full and stable and native and workers_ok and tr.it == self.start_it+1
        frames, indices, samples, weights = [], {}, [], []
        previous = 0.0
        for at, stack in self.samples:
            ids = []
            for frame in stack:
                if frame not in indices:
                    indices[frame] = len(frames)
                    frames.append(dict(name=frame))
                ids.append(indices[frame])
            samples.append(ids)
            weights.append(at-previous)
            previous = at
        if samples:
            weights[-1] += self.end-self.start-previous
        write(self.path/'wall.speedscope.json', {'$schema':'https://www.speedscope.app/file-format-schema.json', 'shared':dict(frames=frames), 'profiles':[dict(type='sampled', name=f'PPO iteration {tr.it}: learner Python wall stacks', unit='seconds', startValue=0, endValue=self.end-self.start, samples=samples, weights=weights)], 'activeProfileIndex':0})
        events = self.tp.key_averages()
        ops = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total, total_cpu_us=e.cpu_time_total, self_device_us=e.self_device_time_total, total_device_us=e.device_time_total) for e in events]
        write(self.path/'operators.json', ops)
        write(self.path/'result.json', dict(qualified=qualified, full_transitions=full, stable_membership=stable, native_jobs_unchanged=native, worker_profiles_complete=workers_ok, worker_results=worker_results, iteration=tr.it, transitions=tr.transitions-self.start_transitions, start_transitions=self.start_transitions, wall_s=self.end-self.start, start_utc=self.start_utc, end_utc=time.time(), pid=os.getpid(), host=socket.gethostname(), jobs_before=self.before, jobs_after=after, steps=self.steps, spans=self.spans, receives=self.receives, remote_profile=tr.env.profile(), sampling='20ms Python wall stacks on main thread; no native frames; GIL scheduling may bias samples', sampler_count=len(samples), torch_version=torch.__version__))
        write(OUT/'status.json', dict(phase='complete' if qualified else 'rejected', qualified=qualified, iteration=tr.it, path=str(self.path)))
        print(f'[profile] iteration={tr.it} qualified={qualified} output={self.path}', flush=True)
        return qualified


active = None
attempts = 0
warmup = True
finished = False


def wrap(method, name):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        global active, attempts, warmup, finished
        if name == 'collect' and not finished:
            if warmup:
                warmup = False
            elif active is None and (OUT/'workers-ready').exists() and all(c is not None for c in self.env.conn_of) and bool(self.env.alive.all()):
                attempts += 1
                active = Capture(self, attempts)
        cap = active
        if cap is None:
            return method(self, *args, **kwargs)
        start = time.perf_counter()
        with torch.profiler.record_function('PPO::'+name):
            result = method(self, *args, **kwargs)
        cap.spans.append(dict(name=name, start_s=start-cap.start, duration_s=time.perf_counter()-start))
        if name == 'save' and self.it == cap.start_it+1:
            finished = cap.finish() or attempts >= 4
            active = None
            if finished:
                original = OUT / 'original-command.json'
                write(self.run / 'command.json', json.loads(original.read_text()))
        return result
    return call


from assembly_policy.rl.remote import RemoteVecEnv
original_step = RemoteVecEnv.step


def step(self, action):
    cap = active
    if cap is None:
        return original_step(self, action)
    start = time.perf_counter()
    old_sum, old_n = self.prof['ct_sum'].copy(), self.prof['ct_n'].copy()
    with torch.profiler.record_function(f'PPO::remote_step_{len(cap.steps)}'):
        result = original_step(self, action)
    cap.steps.append(dict(index=len(cap.steps), start_s=start-cap.start, wall_s=time.perf_counter()-start, active=int(self.alive.sum()), stable=all(a is b for a,b in zip(self.conn_of, cap.connections)), worker_ct_s=(self.prof['ct_sum']-old_sum).tolist(), worker_replies=(self.prof['ct_n']-old_n).tolist(), hosts=list(self.hosts)))
    return result


RemoteVecEnv.step = step
for name in ['collect', 'update', 'save']:
    setattr(train.Trainer, name, wrap(getattr(train.Trainer, name), name))

# Record exact reply arrival and payload worker-compute timing without changing
# wire content. Connection.recv includes blocking read and unpickling together.
from multiprocessing.connection import Connection
original_recv = Connection.recv
original_send = Connection.send


def send(self, obj):
    if active is not None and isinstance(obj, dict) and obj.get('cmd')=='step':
        obj = {**obj, 'profile_path':str(active.path)}
    # Existing workers do not understand the extra stop control message. Send
    # only to instrumented connections, identified from their hello metadata.
    if isinstance(obj, dict) and obj.get('cmd')=='profile_stop':
        if id(self) not in instrumented_connections:
            return None
    return original_send(self, obj)


instrumented_connections = set()


def recv(self):
    cap = active
    start = time.perf_counter()
    result = original_recv(self)
    if isinstance(result, dict) and result.get('profile_worker'):
        instrumented_connections.add(id(self))
    if cap is not None and isinstance(result, dict) and 'ct' in result:
        index = next((i for i,c in enumerate(cap.connections) if c is self), -1)
        cap.receives.append(dict(worker=index, step=len(cap.steps), start_s=start-cap.start, wall_s=time.perf_counter()-start, ct_s=float(result['ct'])))
    return result


Connection.recv = recv
Connection.send = send
write(OUT/'status.json', dict(phase='warmup', pid=os.getpid(), time=time.time()))
train.main()
