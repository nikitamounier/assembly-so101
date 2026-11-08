"""Compile/render the independent five-component CAD review scene.

Uses native MuJoCo; geometry is visual-only, with gravity disabled. This is an
articulated asset review, not a contact/insertion/training environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def add(parent, tag, **kwargs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kwargs.items()})


def camera_axes(eye, target):
    z = np.array(eye, dtype=float) - np.array(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    return np.r_[x, np.cross(z, x)]


def review_xml(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    root = ET.Element("mujoco", model=manifest["profile"])
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep=".002", gravity="0 0 0", integrator="implicitfast")
    default = add(root, "default")
    add(default, "geom", contype="0", conaffinity="0", group="2")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1600", offheight="1200")
    add(visual, "headlight", ambient=".6 .6 .6", diffuse=".6 .6 .6")
    assets = add(root, "asset")
    add(
        assets,
        "texture",
        name="review_background",
        type="skybox",
        builtin="flat",
        rgb1=".96 .97 .98",
        width="64",
        height="64",
    )
    world = add(root, "worldbody")
    acts = add(root, "actuator")
    add(
        world,
        "geom",
        name="review_table",
        type="plane",
        size=".8 .6 .02",
        rgba=".87 .89 .91 1",
    )
    add(world, "light", pos="0 0 1.4", diffuse=".8 .8 .8")
    for side, x, yaw in [("left", -0.48, 0), ("right", 0.48, np.pi)]:
        helper = folder.parent / "helper"
        lock = json.loads((ROOT / "cad/wowrobo/helper-lock.json").read_text())
        for item in lock["files"]:
            p = helper / item["path"]
            if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Helper hash mismatch: {p}")
        src = ET.parse(helper / "so101_new_calib.xml").getroot()
        # Retain one render geom per original part, not duplicate collision geoms.
        for b in src.findall(".//body"):
            for g in list(b.findall("geom")):
                if g.get("class") == "collision":
                    b.remove(g)
        for m in src.findall("asset/mesh"):
            m.set("name", Path(m.get("file")).stem)
            m.set("file", str((helper / "assets" / m.get("file")).resolve()))
        for element in src.iter():
            for key in [
                "name",
                "class",
                "childclass",
                "mesh",
                "material",
                "joint",
                "body",
                "site",
            ]:
                if element.get(key):
                    element.set(key, side + "_" + element.get(key))
        for parent in src.findall("default"):
            for item in parent:
                default.append(copy.deepcopy(item))
        for item in src.find("asset"):
            assets.append(copy.deepcopy(item))
        for material in assets.findall("material"):
            if "so101" in material.get("name", ""):
                material.set("rgba", ".19 .23 .28 1")
        body = copy.deepcopy(src.find("worldbody/body"))
        body.set("pos", numbers([x, 0, 0.0024]))
        body.set("quat", numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        world.append(body)
        for item in src.find("actuator"):
            acts.append(copy.deepcopy(item))
        gripper = body.find(f".//body[@name='{side}_gripper']")
        # Use the camera-equipped model's registered attachment, rather than
        # guessing which adapter face holds the PCB. Its wrist frame matches
        # our pinned follower exactly. Intrinsics remain uncalibrated for this kit.
        reference_dir = folder.parent / "camera-reference"
        camera_lock = json.loads(
            (ROOT / "cad/wowrobo/camera-source-lock.json").read_text()
        )
        for entry in camera_lock["files"]:
            path = reference_dir / entry["local"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(urlopen(entry["url"], timeout=60).read())
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Camera source hash mismatch: {path}")
        camera_source = ET.parse(reference_dir / "so101_new_calib_camera.xml").getroot()
        reference_body = camera_source.find(".//body[@name='wrist_camera_mount']")
        cam = copy.deepcopy(reference_body)
        for parent in cam.iter("body"):
            for geom in list(parent.findall("geom")):
                if geom.get("class") == "collision":
                    parent.remove(geom)
                else:
                    geom.attrib.pop("class", None)
                    geom.attrib.pop("material", None)
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    geom.set("group", "2")
                    geom.set("rgba", ".13 .15 .18 1")
        for element in cam.iter():
            for key in ["name", "mesh"]:
                if element.get(key):
                    element.set(key, side + "_" + element.get(key))
        cam.set("name", side + "_camera_rig")
        cam.find("geom").set("name", side + "_camera_mount")
        module = cam.find("body")
        module.set("name", side + "_camera_module")
        module.find("geom").set("name", side + "_camera_module_geom")
        optical = module.find("camera")
        # Keep source lens position/optical axis; do not infer this kit's lens
        # model or sensor dimensions merely from visual similarity.
        for key in ["resolution", "sensorsize", "focal"]:
            optical.attrib.pop(key, None)
        optical.set("name", side + "_wrist_camera")
        optical.set("fovy", "75")
        for mesh_name, scale in [
            ("wrist_camera_mount_so101_v1", ".001 .001 .001"),
            ("wrist_camera_so101_v1", "1 1 1"),
        ]:
            add(
                assets,
                "mesh",
                name=side + "_" + mesh_name,
                file=str((reference_dir / (mesh_name + ".stl")).resolve()),
                scale=scale,
            )
        # Photo-observed gold PCB perimeter, cosmetic only. Dimensions remain
        # the reference32mm module size until measured on the real camera.
        for edge, position, size in [
            ("a", [0, -0.015, 0.0018], [0.0155, 0.00045, 0.00015]),
            ("b", [0, 0.015, 0.0018], [0.0155, 0.00045, 0.00015]),
            ("c", [-0.015, 0, 0.0018], [0.00045, 0.0155, 0.00015]),
            ("d", [0.015, 0, 0.0018], [0.00045, 0.0155, 0.00015]),
        ]:
            add(
                module,
                "geom",
                name=side + "_camera_pcb_edge_" + edge,
                type="box",
                pos=numbers(position),
                size=numbers(size),
                rgba=".64 .48 .22 1",
                group="2",
            )
        gripper.append(cam)
    colors = {
        "printed": ".28 .38 .48 1",
        "servo_case": ".11 .13 .16 1",
        "servo_horns": ".72 .76 .81 1",
        "pin": ".88 .57 .13 1",
    }
    for i, (name, piece) in enumerate(manifest["pieces"].items()):
        center = np.array(piece["center_m"])
        body = add(
            world,
            "body",
            name=name,
            pos=numbers(np.array(piece["review_position_m"]) - center),
        )
        add(body, "freejoint", name=name + "_free")
        add(
            body,
            "inertial",
            pos=numbers(center),
            mass=".1",
            diaginertia=".0001 .0001 .0001",
        )
        pivot = {}
        for servo in piece["servos"]:
            axis = manifest["servo_axes"][str(servo)]
            child = add(body, "body", name=f"target_servo_{servo}_pivot")
            # All constituent meshes remain in the common CAD assembly frame.
            # Cases rotate at motors2/3 (horns fastened to yokes), horns at motors1/4.
            add(
                child,
                "joint",
                name=f"target_servo_{servo}_shaft",
                type="hinge",
                pos=numbers(axis["position_m"]),
                axis=numbers(axis["axis"]),
                range="-1.7 1.7",
                damping=".02",
            )
            add(
                child,
                "inertial",
                pos=numbers(axis["position_m"]),
                mass=".02",
                diaginertia=".00001 .00001 .00001",
            )
            pivot[servo] = child
        for key in piece["constituents"]:
            item = manifest["constituents"][key]
            parent = body
            if key.startswith("servo_"):
                servo = int(key.split("_")[1])
                moving = manifest["moving_servo_parts"][str(servo)]
                if item["kind"] == moving:
                    parent = pivot[servo]
            add(assets, "mesh", name=key, file=str((folder / item["mesh"]).resolve()))
            add(
                parent,
                "geom",
                name=key,
                type="mesh",
                mesh=key,
                rgba=colors[item["kind"]],
            )
    return root


def build_review(folder):
    root = review_xml(folder)
    path = folder / "review.xml"
    ET.indent(root)
    for item in root.findall("asset/mesh"):
        item.set("file", os.path.relpath(item.get("file"), folder))
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    # Review pose for the helpers; no action commands or sim-to-real claims.
    for side in ["left", "right"]:
        for joint, value in [
            ("shoulder_pan", 0),
            ("shoulder_lift", -0.7),
            ("elbow_flex", 1.0),
            ("wrist_flex", 0.65),
            ("wrist_roll", 0),
            ("gripper", 0.5),
        ]:
            j = model.joint(side + "_" + joint)
            data.qpos[j.qposadr[0]] = value
        for k in range(model.nu):
            j = model.actuator_trnid[k, 0]
            data.ctrl[k] = data.qpos[model.jnt_qposadr[j]]
    mujoco.mj_forward(model, data)
    np.savez(folder / "review_pose.npz", qpos=data.qpos, ctrl=data.ctrl)
    outputs = []
    with mujoco.Renderer(model, height=1000, width=1500) as renderer:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, -0.04, 0.12]
        cam.distance = 1.30
        cam.azimuth = 90
        cam.elevation = -52
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(folder / "workcell.png")
        for side in ["left", "right"]:
            renderer.update_scene(data, camera=side + "_wrist_camera")
            Image.fromarray(renderer.render()).save(folder / (side + "_wrist.png"))
        for name, piece in json.loads((folder / "manifest.json").read_text())[
            "pieces"
        ].items():
            body = model.body(name)
            center = np.array(piece["center_m"])
            cam.lookat[:] = data.xpos[body.id] + center
            dims = np.ptp(np.array(piece["bounds_m"]), axis=0)
            cam.distance = max(0.10, float(np.max(dims)) * 1.6)
            cam.azimuth = 120
            cam.elevation = -35
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[2] = 1
            renderer.update_scene(data, camera=cam, scene_option=opt)
            keep = {model.geom(key).id for key in piece["constituents"]}
            for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid not in keep:
                    geom.rgba[3] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            image = Image.fromarray(renderer.render())
            image.save(folder / f"{name}.png")
            outputs.append((piece["label"], image))
    sheet = Image.new("RGB", (1500, 2 * 520), (245, 247, 249))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for i, (label, image) in enumerate(outputs):
        image.thumbnail((500, 470))
        x = i % 3 * 500
        y = i // 3 * 520
        sheet.paste(image, (x, y + 35))
        draw.text((x + 18, y + 12), label, fill=(25, 37, 48), font=font)
    sheet.save(folder / "components.png")
    return model, data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--folder", type=Path, default=ROOT / "assets_local/wowrobo/components-v2"
    )
    args = ap.parse_args()
    model, data = build_review(args.folder)
    print(
        json.dumps(
            {
                "bodies": model.nbody,
                "joints": model.njnt,
                "cameras": model.ncam,
                "helper_actuators": model.nu,
                "review": str(args.folder / "review.xml"),
            }
        )
    )
