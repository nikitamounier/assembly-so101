"""Bounded, observational actual-training replays. No physics or image rendering."""
from __future__ import annotations

import io
import hashlib
import json
import queue
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    temporary.replace(path)


class ReplayWriter:
    """One disk writer with a hard queue bound; full queues drop telemetry only."""

    def __init__(self, run, manifest, queue_size=4):
        self.root = Path(run) / 'training-replays'
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.pending = queue.Queue(maxsize=queue_size)
        self.index = {'schema': 'wowrobo-training-replays-v1', 'run': str(run), 'clips': {}}
        index_path = self.root / 'index.json'
        if index_path.exists():
            prior = json.loads(index_path.read_text())
            if prior.get('identity') == manifest['identity']:
                self.index = prior
        self.index['identity'] = manifest['identity']
        atomic_json(self.root / 'manifest.json', manifest)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, name='assembly-replay-writer', daemon=True)
        self.thread.start()

    def submit(self, clip):
        if self.error:
            return False
        try:
            self.pending.put_nowait(clip)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _write(self):
        while True:
            clip = self.pending.get()
            try:
                if clip is None:
                    return
                frames, times = clip['frames'], clip['times']
                metadata = {k: v for k, v in clip.items() if k not in ('frames', 'times')}
                key = metadata['task'] + '/' + metadata['stage']
                outcome = 'success' if metadata['success'] else 'failure'
                # Exactly two bounded artifact slots per task/start, replaced atomically.
                filename = metadata['task'] + '--' + metadata['stage'] + '--' + outcome + '.npz'
                path = self.root / filename
                temporary = path.with_suffix('.npz.tmp')
                with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for name, value in [('frames', frames), ('times', times)]:
                        stream = io.BytesIO()
                        np.lib.format.write_array(stream, np.asarray(value, np.float32), allow_pickle=False)
                        archive.writestr(name + '.npy', stream.getvalue())
                temporary.replace(path)
                metadata['file'] = filename
                metadata['bytes'] = path.stat().st_size
                metadata['data_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.index['clips'].setdefault(key, {})[outcome] = metadata
                self.index['updated_at'] = time.time()
                atomic_json(self.root / 'index.json', self.index)
            except Exception as exc:
                # Recording failures never terminate or block PPO.
                self.error = type(exc).__name__ + ': ' + str(exc)
            finally:
                self.pending.task_done()

    def close(self, timeout=2):
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            return
        self.thread.join(timeout)


class EpisodeRecorder:
    """At most one candidate per (task, starting state), not per training world.

    The initial desynchronized partial episodes and any worker-loss segments are
    discarded. Terminal frames come from pre-reset final_obs. Only already
    transferred compact body poses are copied; there is no new worker RPC.
    """

    def __init__(self, run, entry_metadata, body_names, n_envs, identity,
                 horizon=375, dt=0.04, frame_stride=2, refresh_seconds=300, writer=None):
        self.run = str(run)
        self.entries = list(entry_metadata)
        self.body_names = tuple(body_names)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.stride = int(frame_stride)
        if self.stride < 1:
            raise ValueError('Positive replay stride required')
        self.refresh_seconds = refresh_seconds
        self.identity = dict(identity)
        self.identity.setdefault('run', self.run)
        self.keys = []
        self.group_of_entry = []
        for row in self.entries:
            task, stage = row['task'], row['stage']
            if not re.fullmatch(r'[a-zA-Z0-9_]+', task + stage):
                raise ValueError('Unsafe replay task or stage name')
            key = task + '/' + stage
            if key not in self.keys:
                self.keys.append(key)
            self.group_of_entry.append(self.keys.index(key))
        self.group_of_entry = np.asarray(self.group_of_entry)
        groups = len(self.keys)
        capacity = self.horizon // self.stride + 3
        self.frames = np.empty((groups, capacity, len(body_names), 7), np.float32)
        self.times = np.empty((groups, capacity), np.float32)
        self.environment = np.full(groups, -1, np.int64)
        self.entry = np.full(groups, -1, np.int64)
        self.length = np.zeros(groups, np.int64)
        self.frame_count = np.zeros(groups, np.int64)
        self.returns = np.zeros(groups, float)
        self.fresh = np.zeros(n_envs, bool)
        self.success_saved = np.zeros(groups, bool)
        self.last_saved = np.full((groups, 2), -np.inf)
        manifest = dict(schema='body-world-poses-v1', body_names=list(body_names),
                        identity=self.identity, entries=self.entries, dt=self.dt, frame_stride=self.stride)
        self.writer = writer if writer is not None else ReplayWriter(run, manifest)

    def _append(self, group, obs, row, step):
        count = self.frame_count[group]
        if count >= self.frames.shape[1]:
            return False
        self.frames[group, count, :, :3] = obs['part_p'][row]
        self.frames[group, count, :, 3:] = obs['part_q'][row]
        self.times[group, count] = step * self.dt
        self.frame_count[group] += 1
        return True

    def before_step(self, obs, entries, alive, now=None):
        if getattr(self.writer, 'error', None):
            return
        now = time.time() if now is None else now
        entries = np.asarray(entries)
        alive = np.asarray(alive, bool)
        for group, environment in enumerate(self.environment):
            if environment >= 0 and (not alive[environment] or entries[environment] != self.entry[group]):
                self.environment[group] = -1
        fresh = np.flatnonzero(self.fresh & alive)
        if len(fresh):
            _, first = np.unique(self.group_of_entry[entries[fresh]], return_index=True)
            fresh = fresh[first]
        for environment in fresh:
            entry = int(entries[environment])
            group = self.group_of_entry[entry]
            if self.environment[group] >= 0:
                continue
            if self.success_saved[group] and now - self.last_saved[group, 1] < self.refresh_seconds:
                continue
            self.environment[group] = environment
            self.entry[group] = entry
            self.length[group] = 0
            self.frame_count[group] = 0
            self.returns[group] = 0
        self.fresh[:] = False
        for group, environment in enumerate(self.environment):
            if environment >= 0 and self.length[group] % self.stride == 0:
                if not self._append(group, obs, environment, self.length[group]):
                    self.environment[group] = -1

    def after_step(self, reward, done, info, final_obs, iteration, now=None):
        now = time.time() if now is None else now
        done = np.asarray(done, bool)
        valid = np.asarray(info['valid'], bool)
        done_ids = np.flatnonzero(done)
        terminal_row = {int(i): row for row, i in enumerate(done_ids)}
        for group, environment in enumerate(self.environment):
            if environment < 0:
                continue
            if not valid[environment]:
                self.environment[group] = -1
                continue
            self.length[group] += 1
            self.returns[group] += reward[environment]
            if self.length[group] > self.horizon:
                self.environment[group] = -1
                continue
            if not done[environment]:
                continue
            success = bool(info['success'][environment])
            full = final_obs is not None and self._append(group, final_obs, terminal_row[int(environment)], self.length[group])
            should_save = success or not self.success_saved[group]
            if full and should_save and now - self.last_saved[group, int(success)] >= self.refresh_seconds:
                count = self.frame_count[group]
                row = self.entries[self.entry[group]]
                clip = dict(id=f'{time.time_ns()}-{group}', recorded_at=now, iteration=int(iteration),
                            task=row['task'], stage=row['stage'], bank_entry=int(self.entry[group]),
                            success=success, complete_episode=True, source='actual_training_episode',
                            terminal_reason='success' if success else 'safety' if info.get('safety', [False] * len(done))[environment] else 'timeout' if info.get('timeout', [False] * len(done))[environment] else 'failure',
                            episode_return=float(self.returns[group]), duration_s=float(self.length[group]*self.dt),
                            dt=self.dt, identity=self.identity,
                            frames=self.frames[group, :count].copy(), times=self.times[group, :count].copy())
                if self.writer.submit(clip):
                    self.last_saved[group, int(success)] = now
                    self.success_saved[group] |= success
            self.environment[group] = -1
        self.fresh[done & valid] = True

    def close(self):
        self.writer.close()
