"""Official WebViewer process. Never loads training physics plugins."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    from mujoco.studio import launch_web, messages, plugin, viewer_protocol
    from mujoco.studio.web import web_server

    # Studio 4.0 has no public host setting. Constrain its otherwise-public bind
    # to loopback before launching; the application proxies its HTTP/WebSockets.
    bind = web_server.bind_public_socket
    web_server.bind_public_socket = lambda host, port=0: bind("127.0.0.1", port)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    class InspectionCamera(plugin.ViewerPlugin):
        focus = None
        last_focus = None

        @messages.handler
        def on_update(self, event: messages.UpdateEvent):
            if self.focus and self.focus != self.last_focus:
                camera = self.viewer.camera
                camera.lookat[:] = self.focus["target"]
                camera.distance = self.focus["radius"] * 2.8
                camera.azimuth = 110
                camera.elevation = -30
                self.last_focus = self.focus

        @messages.handler
        def on_model(self, event: messages.PostModelEvent):
            self.last_focus = None
            self.viewer.camera.lookat[:] = [0, 0.025, 0.12]
            self.viewer.camera.distance = 1.15
            self.viewer.camera.azimuth = 110
            self.viewer.camera.elevation = -30

    config = viewer_protocol.ViewerConfig(title="SO-101 · Reset inspection", gfx="web", http_port=args.port)
    inspection_camera = InspectionCamera()
    with launch_web.launch(config, plugins=[inspection_camera]) as handle:
        handle.set_model(model, data=data)
        version = -1
        while handle.is_running():
            with urllib.request.urlopen(args.source, timeout=5) as response:
                snapshot = json.load(response)
            if snapshot["version"] != version:
                poses = np.asarray(snapshot["poses"])
                data.mocap_pos[:] = poses[:, :3]
                data.mocap_quat[:] = poses[:, 3:]
                mujoco.mj_forward(model, data)
                inspection_camera.focus = snapshot.get("focus")
                version = snapshot["version"]
            handle.sync(model, data)
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
