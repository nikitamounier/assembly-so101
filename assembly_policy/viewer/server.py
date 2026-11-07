"""Loopback application server and same-origin official WebViewer gateway."""

import asyncio
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path

import mujoco
from aiohttp import ClientSession, WSMsgType, web
from yarl import URL

from assembly_policy.so101.wowrobo_reward import RewardRegions

from .banks import ResetCatalog
from .point_cloud import PolicyPointCloud
from .scene import display_xml, scene_poses, visual_scene

STATIC = Path(__file__).parent / "static"


@web.middleware
async def isolation(request, handler):
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        exc.headers.update(
            {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp"}
        )
        raise
    response.headers.update(
        {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
        }
    )
    return response


class ResetApplication:
    def __init__(
        self, roots, studio_python, port, *, scene="so101", model=None, model_signature=None, state_spec=None
    ):
        if scene not in ("so101", "cube-trial", "wowrobo-components", "wowrobo-layout"):
            raise ValueError(f"Unknown viewer scene: {scene}")
        if model is None or model_signature is None:
            if scene == "cube-trial":
                from assembly_policy.cube_trial.core import model as make_model
                from assembly_policy.cube_trial.core import signature
            else:
                from assembly_policy.so101.resets import signature
                from assembly_policy.so101.scene import model as make_model
            if model is None:
                model = make_model()
            if model_signature is None:
                model_signature = signature()
        self.model = model
        self.scene_name = scene
        self.state_spec = (
            state_spec
            if state_spec is not None
            else (
                mujoco.mjtState.mjSTATE_INTEGRATION
                if scene == "cube-trial"
                else mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
        )
        self.data = mujoco.MjData(self.model)
        self.scene = visual_scene(self.model)
        self.bank_roots = list(roots)
        self.model_signature = model_signature
        self.catalog = ResetCatalog(
            self.bank_roots,
            model_signature,
            mujoco.mj_stateSize(self.model, self.state_spec),
            state_spec=self.state_spec,
        )
        self.studio_python = Path(studio_python).expanduser().absolute()
        self.port = port
        self.point_cloud = PolicyPointCloud(self.model, self.scene_name)
        self.reward_regions = RewardRegions(self.model, self.model_signature)
        self.version = 0
        self.snapshot = None
        self.studio_process = None
        self.studio_port = None
        self.temp = tempfile.TemporaryDirectory(prefix="so101-viewer-")
        self.log = None
        self.start_lock = asyncio.Lock()
        if self.catalog.entries:
            self.select(self.catalog.entries[0]["id"])

    def select(self, entry_id):
        entry, state = self.catalog.state(entry_id)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, self.state_spec)
        # Kinematics only: saved contacts/velocities are never advanced or settled.
        mujoco.mj_kinematics(self.model, self.data)
        self.version += 1
        self.snapshot = dict(
            version=self.version,
            point_cloud=self.point_cloud.snapshot(self.data, entry),
            reward_region=self.reward_regions.snapshot(self.data, entry),
            entry=entry,
            poses=scene_poses(self.model, self.data, self.scene),
            time=float(self.data.time),
            qpos=self.data.qpos.tolist(),
            qvel=self.data.qvel.tolist(),
            focus=entry.get("focus"),
            qualification=self.catalog.qualifications.get(entry_id),
            variation=self.catalog.variations.get(entry_id),
        )
        return self.snapshot

    async def catalog_handler(self, request):
        return web.json_response(
            dict(
                banks=self.catalog.banks,
                entries=self.catalog.entries,
                studio_installed=self.studio_python.is_file(),
                scene=self.scene_name,
                default_bank=self.catalog.default_bank,
            )
        )

    def refresh_catalog(self):
        """Reindex atomically published banks without changing the loaded model."""
        selected = self.snapshot["entry"]["id"] if self.snapshot else None
        self.catalog = ResetCatalog(
            self.bank_roots,
            self.model_signature,
            mujoco.mj_stateSize(self.model, self.state_spec),
            state_spec=self.state_spec,
        )
        if self.catalog.entries:
            self.select(selected if selected in self.catalog.by_id else self.catalog.entries[0]["id"])
        else:
            self.snapshot = None

    async def refresh_handler(self, request):
        self.check_origin(request)
        self.refresh_catalog()
        return await self.catalog_handler(request)

    async def scene_handler(self, request):
        return web.json_response(self.scene)

    async def state_handler(self, request):
        if self.snapshot is None:
            raise web.HTTPNotFound(text="No compatible reset states were discovered")
        return web.json_response(self.snapshot)

    async def select_handler(self, request):
        self.check_origin(request)
        try:
            data = await request.json()
            result = self.select(data["id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="Unknown reset ID") from None
        return web.json_response(result)

    async def reward_tolerance_handler(self, request):
        self.check_origin(request)
        try:
            values = await request.json()
            self.reward_regions.set_tolerances(values["position_mm"], values["orientation_degrees"])
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(text="Position must be 0.01–5 mm; orientation 0.1–10 degrees") from None
        return web.json_response(dict(updated=True))

    @staticmethod
    def check_origin(request):
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            raise web.HTTPForbidden(text="Cross-origin control requests are disabled")

    async def studio_handler(self, request):
        self.check_origin(request)
        async with self.start_lock:
            if not self.studio_python.is_file():
                raise web.HTTPServiceUnavailable(
                    text="Install the viewer extra: uv pip install -e '.[viewer]'"
                )
            if self.snapshot is None:
                raise web.HTTPConflict(text="Select a compatible reset first")
            if self.studio_process is None or self.studio_process.poll() is not None:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    self.studio_port = sock.getsockname()[1]
                display = Path(self.temp.name) / "display.xml"
                display.write_text(display_xml(self.scene))
                if self.log:
                    self.log.close()
                self.log = (Path(self.temp.name) / "studio.log").open("w")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                self.studio_process = subprocess.Popen(
                    [
                        str(self.studio_python),
                        "-m",
                        "assembly_policy.viewer.studio",
                        "--model",
                        str(display),
                        "--source",
                        f"http://127.0.0.1:{self.port}/api/state",
                        "--port",
                        str(self.studio_port),
                    ],
                    env=env,
                    stdout=self.log,
                    stderr=subprocess.STDOUT,
                )
            for _ in range(300):
                if self.studio_process.poll() is not None:
                    log = (Path(self.temp.name) / "studio.log").read_text()[-1500:]
                    raise web.HTTPServiceUnavailable(text=f"WebViewer exited: {log}")
                try:
                    async with self.client.get(
                        f"http://127.0.0.1:{self.studio_port}/", timeout=1
                    ) as response:
                        if response.status == 200:
                            return web.json_response(dict(url="/mujoco/"))
                except (OSError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.1)
            raise web.HTTPServiceUnavailable(text="WebViewer is still starting; try again")

    async def proxy(self, request):
        if self.studio_process is None or self.studio_process.poll() is not None:
            raise web.HTTPServiceUnavailable(text="Start the MuJoCo view first")
        self.check_origin(request)
        path = request.path.removeprefix("/mujoco") or "/"
        if request.query_string:
            path += "?" + request.query_string
        # Studio serves literal asset filenames, including the variable font's
        # square brackets; aiohttp's default URL canonicalization escapes them.
        url = URL(f"http://127.0.0.1:{self.studio_port}{path}", encoded=True)
        if request.headers.get("Upgrade", "").lower() == "websocket":
            # Reset inspection has no model uploads or simulator mutation channel.
            if request.path.endswith("/drop"):
                raise web.HTTPForbidden(text="Model uploads are disabled in reset inspection")
            async with self.client.ws_connect(url, max_msg_size=64 * 1024 * 1024) as remote:
                local = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
                local.headers.update(
                    {
                        "Cross-Origin-Opener-Policy": "same-origin",
                        "Cross-Origin-Embedder-Policy": "require-corp",
                    }
                )
                await local.prepare(request)

                async def relay(source, target):
                    async for message in source:
                        if message.type == WSMsgType.BINARY:
                            await target.send_bytes(message.data)
                        elif message.type == WSMsgType.TEXT:
                            await target.send_str(message.data)
                        elif message.type == WSMsgType.ERROR:
                            break

                tasks = [asyncio.create_task(relay(local, remote)), asyncio.create_task(relay(remote, local))]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await local.close()
                return local
        async with self.client.get(url) as response:
            body = await response.read()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            return web.Response(body=body, status=response.status, headers={"Content-Type": content_type})

    async def static_handler(self, request):
        path = (STATIC / request.match_info.get("path", "index.html")).resolve()
        if not path.is_relative_to(STATIC.resolve()) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def lifecycle(self, app):
        self.client = ClientSession()
        try:
            yield
        finally:
            if self.studio_process and self.studio_process.poll() is None:
                self.studio_process.terminate()
                try:
                    await asyncio.to_thread(self.studio_process.wait, 5)
                except subprocess.TimeoutExpired:
                    self.studio_process.kill()
                    await asyncio.to_thread(self.studio_process.wait)
            await self.client.close()
            if self.log:
                self.log.close()
            self.temp.cleanup()

    def app(self):
        app = web.Application(middlewares=[isolation], client_max_size=64 * 1024)
        app.cleanup_ctx.append(self.lifecycle)
        app.router.add_get("/api/catalog", self.catalog_handler)
        app.router.add_post("/api/refresh", self.refresh_handler)
        app.router.add_get("/api/scene", self.scene_handler)
        app.router.add_get("/api/state", self.state_handler)
        app.router.add_post("/api/select", self.select_handler)
        app.router.add_post("/api/reward-tolerance", self.reward_tolerance_handler)
        app.router.add_post("/api/studio", self.studio_handler)
        app.router.add_get("/mujoco/{path:.*}", self.proxy)
        for path in ("/ui", "/state", "/model", "/drop"):
            app.router.add_get(path, self.proxy)
        app.router.add_get("/", self.static_handler)
        app.router.add_get("/{path:.*}", self.static_handler)
        return app
