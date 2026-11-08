"""Loopback dashboard with one allocated, read-only SSH observation every ten seconds."""

import argparse
import asyncio
import json
import shlex
import time
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import web

from assembly_policy.cube_trial.core import FAMILIES, JOINTS, generate, model
from assembly_policy.viewer.scene import scene_poses, visual_scene

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


class Dashboard:
    def __init__(self, *, remote_root=None, monitor_job=None, ssh_host="chimera-login", state_dir=None):
        self.remote_root = remote_root
        self.monitor_job = monitor_job
        self.ssh_host = ssh_host
        self.state_dir = Path(state_dir) if state_dir else REPO / "artifacts_local/cube-dashboard"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.live = dict(connection="connecting", history=[], episode_id=None, errors=[])
        self.episode = None
        self.replay = None
        self.episodes = {}
        self.replays = {}
        self.m = model()
        self.scene = visual_scene(self.m)
        self.d, _ = generate("loose", 0, self.m)
        self.initial = scene_poses(self.m, self.d, self.scene)
        self.joints = [self.m.joint(s + "_" + j).qposadr[0] for s in ("left", "right") for j in JOINTS]
        self.cache = self.state_dir / "live-cache.json"
        if self.cache.exists():
            try:
                self.live = json.loads(self.cache.read_text())
                self.live["connection"] = "connecting"
            except json.JSONDecodeError:
                pass

    async def bridge(self):
        while True:
            command = [
                self.remote_root + "/.venv/bin/python",
                self.remote_root + "/operational/cube-trial-20261107/dashboard_export.py",
            ]
            if self.episode:
                command += ["--known-episode-id", self.episode["id"]]
            command += ["--known-family-episodes", json.dumps({f: c["id"] for f, c in self.episodes.items()})]
            remote = shlex.join(
                [
                    "srun",
                    "--jobid=" + self.monitor_job,
                    "--overlap",
                    "--exact",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    *command,
                ]
            )
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    self.ssh_host,
                    remote,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
                if process.returncode:
                    raise RuntimeError(stderr.decode()[-350:] or "Allocated observation failed")
                payload = json.loads(stdout)
                episode = payload.pop("episode", None)
                clips = payload.pop("episodes", {})
                if payload.get("run") != self.live.get("run"):
                    self.episode = None
                    self.episodes.clear()
                    self.replays.clear()
                self.episodes.update(clips)
                self.episodes = {
                    f: c for f, c in self.episodes.items() if payload.get("episode_ids", {}).get(f) == c["id"]
                }
                if not payload.get("episode_id"):
                    self.episode = None
                if episode:
                    self.episode = episode
                    self.replay = None
                payload["connection"] = "live"
                payload["received_at"] = time.time()
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
                self.live = dict(self.live, connection="unavailable", connection_error=str(error))
            await asyncio.sleep(10)

    def failed_diagnostic(self, family):
        if family != "pre_handoff" or self.live.get("episode_ids", {}).get(family):
            return None
        stats = self.live.get("summary", {}).get("families", {}).get(family, {})
        if not stats or stats.get("successes", 0):
            return None
        path = self.state_dir / "diagnostic-pre-handoff.json"
        if not path.exists():
            return None
        clip = json.loads(path.read_text())
        config_path = REPO / "operational/cube-trial-20261107/cluster.json"
        if not config_path.exists():
            return None
        config = json.loads(config_path.read_text())
        if (
            clip.get("run") == self.live.get("run")
            and clip.get("bank") == config.get("bank")
            and clip.get("family") == family
            and clip.get("success") is False
            and clip.get("complete_episode") is True
        ):
            return clip
        return None

    def render_episode(self, family=None):
        clip = self.episodes.get(family) if family else self.episode
        if not clip:
            clip = self.failed_diagnostic(family)
        if not clip:
            return None
        cached = self.replays.get(family or "*")
        if cached and cached["id"] == clip["id"]:
            return cached
        if clip:
            d = mujoco.MjData(self.m)
            frames = []
            cube = self.m.joint("cube_free").qposadr[0]
            target = self.m.body("target").mocapid[0]
            for frame in clip["frames"]:
                f = np.asarray(frame)
                d.qpos[self.joints] = f[:12]
                d.qpos[cube : cube + 7] = f[12:19]
                d.mocap_pos[target] = f[19:22]
                d.mocap_quat[target] = f[22:26]
                mujoco.mj_kinematics(self.m, d)
                frames.append(scene_poses(self.m, d, self.scene))
            replay = dict(**{k: v for k, v in clip.items() if k != "frames"}, frames=frames)
            self.replays[family or "*"] = replay
            return replay

    async def live_handler(self, request):
        payload = dict(self.live)
        diagnostic = self.failed_diagnostic("pre_handoff")
        if diagnostic:
            payload["episode_ids"] = dict(payload.get("episode_ids", {}), pre_handoff=diagnostic["id"])
        context_path = self.state_dir / "run-context.json"
        if context_path.exists():
            context = json.loads(context_path.read_text())
            if context.get("run") == payload.get("run"):
                payload["run_context"] = context
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def episode_handler(self, request):
        family = request.query.get("family")
        if family is not None and family not in FAMILIES:
            raise web.HTTPBadRequest(text="Unknown reset family")
        replay = await asyncio.to_thread(self.render_episode, family)
        return web.json_response(replay, headers={"Cache-Control": "no-store"})

    async def startup(self, app):
        self.task = asyncio.create_task(self.bridge()) if self.remote_root else None
        if self.task is None:
            self.live["connection"] = "offline"

    async def cleanup(self, app):
        if self.task is None:
            return
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Loopback cube reward dashboard; offline unless configured")
    parser.add_argument("--remote-root", help="Remote checkout containing the allocated dashboard exporter")
    parser.add_argument(
        "--monitor-job", help="An already authorized Slurm CPU allocation for read-only exports"
    )
    parser.add_argument("--ssh-host", default="chimera-login")
    parser.add_argument(
        "--state-dir", type=Path, help="Local cache, optional run context and diagnostic clips"
    )
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    if bool(args.remote_root) != bool(args.monitor_job):
        parser.error("--remote-root and --monitor-job must be supplied together")
    if not (ROOT / "dist/index.html").exists():
        parser.error(
            "Build the dashboard with web/node_modules/.bin/vite build --config dashboards/cube/vite.config.mjs"
        )
    dash = Dashboard(
        remote_root=args.remote_root,
        monitor_job=args.monitor_job,
        ssh_host=args.ssh_host,
        state_dir=args.state_dir,
    )
    app = web.Application()
    app.router.add_get("/api/live", dash.live_handler)
    app.router.add_get("/api/episode", dash.episode_handler)
    app.router.add_get("/api/scene", lambda r: web.json_response(dash.scene))
    app.router.add_get("/api/initial", lambda r: web.json_response(dash.initial))
    app.router.add_get("/", lambda r: web.FileResponse(ROOT / "dist/index.html"))
    app.router.add_static("/assets", ROOT / "dist/assets")
    app.on_startup.append(dash.startup)
    app.on_cleanup.append(dash.cleanup)
    web.run_app(app, host="127.0.0.1", port=args.port, print=print)


if __name__ == "__main__":
    main()
