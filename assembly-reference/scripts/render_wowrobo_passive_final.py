"""Render the saved force-driven trajectory without changing simulated states."""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from assembly_policy.so101.wowrobo_contact_backend import load_backend

p = argparse.ArgumentParser()
p.add_argument("folder")
a = p.parse_args()
out = Path(a.folder)
info = json.loads((out / "result.json").read_text())
load_backend(info["grid_manifest"])
m = mujoco.MjModel.from_xml_path(info["scene"])
d = mujoco.MjData(m)
z = np.load(out / "trajectory.npz")
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, -0.08, 0.16]
cam.distance = 0.68
cam.azimuth = 140
cam.elevation = -25
w, h = 960, 640
fps = 30
proc = subprocess.Popen(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(out / "simulation.mp4"),
    ],
    stdin=subprocess.PIPE,
)
opt = mujoco.MjvOption()
opt.geomgroup[1] = 0
opt.geomgroup[3] = 0
with mujoco.Renderer(m, height=h, width=w) as renderer:
    for t in np.arange(0, z["time"][-1], 1 / fps):
        i = int(np.argmin(abs(z["time"] - t)))
        d.qpos[:] = z["qpos"][i]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, cam, scene_option=opt)
        g = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.004, 0, 0]),
            d.geom("servo_3_case").xpos,
            np.eye(3).ravel(),
            np.array([1, 0.45, 0.05, 1.0]),
        )
        renderer.scene.ngeom += 1
        frame = renderer.render()
        im = Image.fromarray(frame)
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, w, 65), fill="white")
        dr.text(
            (15, 8),
            f"Force-held motor / passive hinge / fixed receiver / gravity ON | t={t:.2f}s",
            fill="black",
        )
        dr.text(
            (15, 30),
            f"Bracket angle {np.degrees(z['hinge_angle'][i]):.1f} deg | own contacts {z['own_contacts'][i]} | receiving contacts {z['receiver_contacts'][i]}",
            fill="black",
        )
        proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close()
assert proc.wait() == 0
