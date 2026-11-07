"""A renderer-neutral visual scene derived from the native compiled model."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def visual_scene(model):
    geoms = []
    meshes = {}
    for i in range(model.ngeom):
        kind = int(model.geom_type[i])
        if model.geom_group[i] > 2 or kind == int(mujoco.mjtGeom.mjGEOM_SDF):
            continue
        rgba = model.geom_rgba[i].copy()
        material = int(model.geom_matid[i])
        if material >= 0:
            rgba *= model.mat_rgba[material]
        if rgba[3] <= 0:
            continue
        geom = dict(
            id=i,
            name=model.geom(i).name or f"geom_{i}",
            type=kind,
            size=model.geom_size[i].tolist(),
            rgba=rgba.tolist(),
        )
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[i])
            key = str(mesh)
            if key not in meshes:
                va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                meshes[key] = dict(
                    vertices=model.mesh_vert[va : va + vn].ravel().tolist(),
                    faces=model.mesh_face[fa : fa + fn].ravel().tolist(),
                )
            geom["mesh"] = key
        geoms.append(geom)
    return dict(geoms=geoms, meshes=meshes, up="z", units="metres")


def scene_poses(model, data, scene):
    poses = []
    for geom in scene["geoms"]:
        i = geom["id"]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
        poses.append([*data.geom_xpos[i].tolist(), *quat.tolist()])
    return poses


def display_xml(scene):
    """Build a non-simulating display model; native collision plugins stay in the host."""
    root = ET.Element("mujoco", model="SO-101 reset inspection")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient=".6 .6 .6", diffuse=".8 .8 .8")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=".65 .72 .73",
        rgb2=".9 .92 .9",
        width="256",
        height="256",
    )
    numbers = lambda values: " ".join(format(v, ".9g") for v in values)
    for key, mesh in scene["meshes"].items():
        ET.SubElement(
            asset, "mesh", name=f"mesh_{key}", vertex=numbers(mesh["vertices"]), face=numbers(mesh["faces"])
        )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse=".9 .9 .9")
    types = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
    for i, geom in enumerate(scene["geoms"]):
        body = ET.SubElement(world, "body", name=f"display_{i}", mocap="true")
        attrs = dict(
            name=f"visual_{i}",
            type=types[geom["type"]],
            size=numbers(geom["size"]),
            rgba=numbers(geom["rgba"]),
            contype="0",
            conaffinity="0",
            group="0",
        )
        if "mesh" in geom:
            attrs["mesh"] = "mesh_" + geom["mesh"]
        ET.SubElement(body, "geom", **attrs)
    return ET.tostring(root, encoding="unicode")
