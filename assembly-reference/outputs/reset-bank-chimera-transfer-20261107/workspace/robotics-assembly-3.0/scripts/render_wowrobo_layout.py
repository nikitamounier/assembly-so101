"""Render the saved initial layout with its measured geometric references."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folder", type=Path, default=ROOT / "assets_local/wowrobo/initial-layout-v2"
    )
    a = p.parse_args()
    m = mujoco.MjModel.from_xml_path(str(a.folder / "scene.xml"))
    d = mujoco.MjData(m)
    archive = np.load(a.folder / "layouts/initial_scene_train.npz")
    font = ImageFont.load_default(size=22)
    with mujoco.Renderer(m, height=1000, width=1500) as renderer:
        for seed in [0, 17]:
            mujoco.mj_setState(
                m, d, archive["states"][seed], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            mujoco.mj_forward(m, d)
            for name, azimuth, elevation, distance in [
                ("perspective", 110, -45, 1.15),
                ("top", 90, -90, 1.12),
            ]:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0, 0, 0.06]
                camera.azimuth = azimuth
                camera.elevation = elevation
                camera.distance = distance
                renderer.update_scene(d, camera=camera)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 1500, 70), fill=(245, 247, 249))
                draw.text(
                    (20, 10),
                    f"Initial workcell | seed {seed} | rear spacing 720 mm | assembly rear offset 300 mm",
                    font=font,
                    fill=(25, 37, 48),
                )
                draw.text(
                    (20, 38),
                    "Blue grip spans: left arm   |   Orange grip spans: right arm   |   tangent at each local grasp center",
                    font=font,
                    fill=(25, 37, 48),
                )
                im.save(a.folder / f"{name}-{seed}.png")
