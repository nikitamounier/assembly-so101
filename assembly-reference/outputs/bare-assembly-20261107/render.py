"""Render saved real pilot qpos; no reenactment or interpolation of physics."""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

out = Path(sys.argv[1])
z = np.load(out / "after-eval.npz")
m = mujoco.MjModel.from_binary_path(str(out / "model.mjb"))
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 360, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.12]
cam.distance = 1.1
cam.azimuth = 100
cam.elevation = -40
option = mujoco.MjvOption()
option.geomgroup[3] = 0
index = 0
with imageio.get_writer(
    out / "same-task-pilot.mp4", fps=25, codec="libx264", quality=7
) as video:
    for qpos in z["qpos"][:, index]:
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam, scene_option=option)
        video.append_data(renderer.render())
renderer.close()
print(out / "same-task-pilot.mp4")
