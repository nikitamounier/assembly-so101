"""Loopback dashboard with one allocated, read-only SSH observation every ten seconds."""
import asyncio
import json
import shlex
import time
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import web

from assembly_policy.cube_trial.core import FAMILIES, JOINTS, model, generate
from assembly_policy.viewer.scene import visual_scene, scene_poses

ROOT = Path(__file__).resolve().parent
REMOTE = '/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6'


class Dashboard:
    def __init__(self):
        self.live = dict(connection='connecting', history=[], episode_id=None, errors=[])
        self.episode = None
        self.replay = None
        self.episodes = {}
        self.replays = {}
        self.m = model()
        self.scene = visual_scene(self.m)
        self.d, _ = generate('loose', 0, self.m)
        self.initial = scene_poses(self.m, self.d, self.scene)
        self.joints = [self.m.joint(s+'_'+j).qposadr[0] for s in ('left', 'right') for j in JOINTS]
        self.cache = ROOT / 'live-cache.json'
        if self.cache.exists():
            try:
                self.live = json.loads(self.cache.read_text())
                self.live['connection'] = 'connecting'
            except json.JSONDecodeError:
                pass

    async def bridge(self):
        while True:
            command = [REMOTE+'/.venv/bin/python', REMOTE+'/operational/cube-trial-20261107/dashboard_export.py']
            if self.episode:
                command += ['--known-episode-id', self.episode['id']]
            command += ['--known-family-episodes', json.dumps({f: c['id'] for f, c in self.episodes.items()})]
            remote = shlex.join(['srun', '--jobid=3077705', '--overlap', '--exact', '--ntasks=1', '--cpus-per-task=1', *command])
            process = None
            try:
                process = await asyncio.create_subprocess_exec('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'chimera-login', remote,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
                if process.returncode:
                    raise RuntimeError(stderr.decode()[-350:] or 'Allocated observation failed')
                payload = json.loads(stdout)
                episode = payload.pop('episode', None)
                clips = payload.pop('episodes', {})
                if payload.get('run') != self.live.get('run'):
                    self.episode = None
                    self.episodes.clear()
                    self.replays.clear()
                self.episodes.update(clips)
                self.episodes = {f:c for f,c in self.episodes.items() if payload.get('episode_ids', {}).get(f) == c['id']}
                if not payload.get('episode_id'):
                    self.episode = None
                if episode:
                    self.episode = episode
                    self.replay = None
                payload['connection'] = 'live'
                payload['received_at'] = time.time()
                self.live = payload
                self.cache.write_text(json.dumps(payload))
            except asyncio.CancelledError:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            except Exception as error:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
                self.live = dict(self.live, connection='unavailable', connection_error=str(error))
            await asyncio.sleep(10)

    def failed_diagnostic(self, family):
        if family != 'pre_handoff' or self.live.get('episode_ids', {}).get(family):
            return None
        stats = self.live.get('summary', {}).get('families', {}).get(family, {})
        if not stats or stats.get('successes', 0):
            return None
        path = ROOT/'diagnostic-pre-handoff.json'
        if not path.exists():
            return None
        clip = json.loads(path.read_text())
        config = json.loads((ROOT.parent/'assembly-so101/operational/cube-trial-20261107/cluster.json').read_text())
        if (clip.get('run') == self.live.get('run') and clip.get('bank') == config.get('bank')
                and clip.get('family') == family and clip.get('success') is False
                and clip.get('complete_episode') is True):
            return clip
        return None

    def render_episode(self, family=None):
        clip = self.episodes.get(family) if family else self.episode
        if not clip:
            clip = self.failed_diagnostic(family)
        if not clip:
            return None
        cached = self.replays.get(family or '*')
        if cached and cached['id'] == clip['id']:
            return cached
        if clip:
            d = mujoco.MjData(self.m)
            frames = []
            cube = self.m.joint('cube_free').qposadr[0]
            target = self.m.body('target').mocapid[0]
            for frame in clip['frames']:
                f = np.asarray(frame)
                d.qpos[self.joints] = f[:12]
                d.qpos[cube:cube+7] = f[12:19]
                d.mocap_pos[target] = f[19:22]
                d.mocap_quat[target] = f[22:26]
                mujoco.mj_kinematics(self.m, d)
                frames.append(scene_poses(self.m, d, self.scene))
            replay = dict(**{k:v for k,v in clip.items() if k!='frames'}, frames=frames)
            self.replays[family or '*'] = replay
            return replay

    async def live_handler(self, request):
        payload = dict(self.live)
        diagnostic = self.failed_diagnostic('pre_handoff')
        if diagnostic:
            payload['episode_ids'] = dict(payload.get('episode_ids', {}), pre_handoff=diagnostic['id'])
        context_path = ROOT/'run-context.json'
        if context_path.exists():
            context = json.loads(context_path.read_text())
            if context.get('run') == payload.get('run'):
                payload['run_context'] = context
        return web.json_response(payload, headers={'Cache-Control':'no-store'})

    async def episode_handler(self, request):
        family = request.query.get("family")
        if family is not None and family not in FAMILIES:
            raise web.HTTPBadRequest(text="Unknown reset family")
        replay = await asyncio.to_thread(self.render_episode, family)
        return web.json_response(replay, headers={'Cache-Control':'no-store'})

    async def startup(self, app):
        self.task = asyncio.create_task(self.bridge())

    async def cleanup(self, app):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


def main():
    dash = Dashboard()
    app = web.Application()
    app.router.add_get('/api/live', dash.live_handler)
    app.router.add_get('/api/episode', dash.episode_handler)
    app.router.add_get('/api/scene', lambda r: web.json_response(dash.scene))
    app.router.add_get('/api/initial', lambda r: web.json_response(dash.initial))
    app.router.add_get('/', lambda r: web.FileResponse(ROOT/'dist/index.html'))
    app.router.add_static('/assets', ROOT/'dist/assets')
    app.on_startup.append(dash.startup)
    app.on_cleanup.append(dash.cleanup)
    web.run_app(app, host='127.0.0.1', port=8880, print=print)


if __name__ == '__main__':
    main()
