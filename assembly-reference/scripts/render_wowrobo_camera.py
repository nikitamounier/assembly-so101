"""Render the photo-comparable camera-on-gripper assembly in a fixed local pose."""

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folder", type=Path, default=ROOT / "assets_local/wowrobo/initial-layout-v2"
    )
    a = p.parse_args()
    root = ET.parse(a.folder / "scene.xml").getroot()
    gripper = copy.deepcopy(root.find('.//body[@name="left_gripper"]'))
    for j in list(gripper.findall("joint")):
        gripper.remove(j)
    rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0.0]])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, rotation.ravel())
    gripper.set("pos", "0 0 .055")
    gripper.set("quat", " ".join(map(str, q)))
    world = root.find("worldbody")
    for item in list(world):
        if (
            item.tag not in ["light", "geom"]
            or item.tag == "geom"
            and item.get("name") != "review_table"
        ):
            world.remove(item)
    world.append(gripper)
    acts = root.find("actuator")
    acts.clear()
    k = root.find("keyframe")
    if k is not None:
        root.remove(k)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[model.joint("left_gripper").qposadr[0]] = 0.65
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=900, width=1300) as renderer:
        for name, az, el in [("camera-closeup", 120, -22), ("camera-side", 90, -8)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.025, 0, 0.065]
            camera.distance = 0.24
            camera.azimuth = az
            camera.elevation = el
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(a.folder / (name + ".png"))
