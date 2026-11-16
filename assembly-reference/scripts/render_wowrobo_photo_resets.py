"""Render every photo reset candidate with arm roles and evidence labels."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assembly_policy.so101.wowrobo_layout import ROOT

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folder", type=Path, default=ROOT / "assets_local/wowrobo/photo-resets-v3"
    )
    a = p.parse_args()
    m = mujoco.MjModel.from_xml_path(str(a.folder / "scene.xml"))
    receipts = json.loads((a.folder / "stage-receipts.json").read_text())
    d = mujoco.MjData(m)
    font = ImageFont.load_default(size=17)
    thumbs = []
    with mujoco.Renderer(m, height=700, width=1000) as renderer:
        for receipt in receipts:
            stage = receipt["definition"]
            name = stage["id"]
            state = np.load(a.folder / "stages" / f"{name}_train.npz")["states"][0]
            mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, 0, 0.10]
            cam.distance = 1.05
            cam.azimuth = 90
            cam.elevation = -38
            renderer.update_scene(d, camera=cam)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 1000, 95), fill="#f4f6f8")
            for y, t in [
                (7, name + " | UNQUALIFIED kinematic candidate"),
                (
                    30,
                    "LEFT: " + stage["left_role"] + " | RIGHT: " + stage["right_role"],
                ),
                (53, stage["evidence"] + " | " + ", ".join(stage["reference_images"])),
                (
                    76,
                    "Terminal reference only"
                    if stage["terminal"]
                    else "Training admission pending contacts/clearance/reward qualification",
                ),
            ]:
                draw.text((12, y), t, font=font, fill="#263644")
            im.save(a.folder / (name + ".png"))
            thumbs.append(im.resize((400, 280)))
    sheet = Image.new("RGB", (1600, 280 * ((len(thumbs) + 3) // 4)), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 4) * 400, (i // 4) * 280))
    sheet.save(a.folder / "contact-sheet.jpg")
