"""Bake an explicit one-axis rigid insertion fixture from a reviewed pose.

This is a virtual-gripper test fixture. It locks helper and receiver poses and
incoming articulation, then leaves only force-driven axial travel free. It is
not production robot physics and cannot establish grasp retention.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _numbers(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_guide(source_scene, alignment_config, output_scene, piece):
    """Preserve every source geom at q=0; permit only rigid axial travel.

    The SDF backend must be loaded before calling this function. The returned
    JSON-ready metadata explicitly records every locked degree of freedom.
    """
    source_scene = Path(source_scene).resolve()
    output_scene = Path(output_scene).resolve()
    config_path = (
        Path(alignment_config).resolve()
        if not isinstance(alignment_config, dict)
        else None
    )
    alignment = json.loads(config_path.read_text()) if config_path else alignment_config
    m = mujoco.MjModel.from_xml_path(str(source_scene))
    d = mujoco.MjData(m)
    qpos = np.asarray(alignment["start_qpos"], dtype=float)
    if qpos.shape != (m.nq,):
        raise ValueError(f"Expected {m.nq} start coordinates; got {qpos.shape}")
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    original_positions = d.geom_xpos.copy()
    original_rotations = d.geom_xmat.copy()
    axis = np.asarray(alignment["axis_world"], dtype=float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("Invalid insertion axis")
    axis /= np.linalg.norm(axis)
    active = m.body(piece).id
    local_axis = d.xmat[active].reshape(3, 3).T @ axis
    root = ET.parse(source_scene).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("fusestatic", "false")
    removed_joint_names = []
    for body in root.iter("body"):
        bid = m.body(body.get("name")).id
        parent = int(m.body_parentid[bid])
        parent_r = d.xmat[parent].reshape(3, 3)
        local_r = parent_r.T @ d.xmat[bid].reshape(3, 3)
        local_p = parent_r.T @ (d.xpos[bid] - d.xpos[parent])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, local_r.ravel())
        for attr in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            body.attrib.pop(attr, None)
        body.set("pos", _numbers(local_p))
        body.set("quat", _numbers(quat))
        for child in list(body):
            if child.tag in ("joint", "freejoint"):
                removed_joint_names.append(child.get("name"))
                body.remove(child)
    for tag in ("actuator", "tendon", "equality", "keyframe", "sensor"):
        for element in list(root.findall(tag)):
            root.remove(element)
    active_element = root.find(f".//body[@name='{piece}']")
    ET.SubElement(
        active_element,
        "joint",
        name="aligned_insertion_slide",
        type="slide",
        axis=_numbers(local_axis),
        pos="0 0 0",
        damping="0",
        frictionloss="0",
        armature="0",
        stiffness="0",
        limited="false",
    )
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene, encoding="unicode")
    guided = mujoco.MjModel.from_xml_path(str(output_scene))
    data = mujoco.MjData(guided)
    mujoco.mj_forward(guided, data)
    if guided.nq != 1 or guided.nv != 1:
        raise RuntimeError("Fixture must have exactly one slide coordinate")
    position_errors = []
    rotation_errors = []
    for gid in range(m.ngeom):
        name = m.geom(gid).name
        new_id = guided.geom(name).id if name else gid
        position_errors.append(
            float(np.max(np.abs(data.geom_xpos[new_id] - original_positions[gid])))
        )
        rotation_errors.append(
            float(np.max(np.abs(data.geom_xmat[new_id] - original_rotations[gid])))
        )
    if (
        max(position_errors, default=0) > 1e-12
        or max(rotation_errors, default=0) > 1e-12
    ):
        raise RuntimeError(
            f"Baked geometry moved: {max(position_errors)}m/{max(rotation_errors)}rotation entries"
        )
    travel = float(alignment.get("authored_axial_travel_m", 0.025))
    qadr = guided.joint("aligned_insertion_slide").qposadr[0]
    initial_positions = data.geom_xpos.copy()
    initial_rotations = data.geom_xmat.copy()
    data.qpos[qadr] = travel
    mujoco.mj_forward(guided, data)
    active_names = set()
    for bid in range(m.nbody):
        ancestor = bid
        while ancestor > 0 and ancestor != active:
            ancestor = int(m.body_parentid[ancestor])
        if ancestor == active:
            active_names.add(m.body(bid).name)
    endpoint_error = 0.0
    endpoint_rotation_error = 0.0
    for gid in range(guided.ngeom):
        moving = guided.body(guided.geom_bodyid[gid]).name in active_names
        expected = initial_positions[gid] + (axis * travel if moving else 0)
        endpoint_error = max(
            endpoint_error, float(np.max(np.abs(data.geom_xpos[gid] - expected)))
        )
        endpoint_rotation_error = max(
            endpoint_rotation_error,
            float(np.max(np.abs(data.geom_xmat[gid] - initial_rotations[gid]))),
        )
    if endpoint_error > 1e-12 or endpoint_rotation_error > 1e-12:
        raise RuntimeError("Slide failed rigid axial geometry check")
    result = {
        "scene": str(output_scene),
        "source_scene": str(source_scene),
        "source_scene_sha256": _hash(source_scene),
        "scene_sha256": _hash(output_scene),
        "alignment_config": str(config_path) if config_path else None,
        "alignment_sha256": _hash(config_path) if config_path else None,
        "piece": piece,
        "joint": "aligned_insertion_slide",
        "axis_world": axis.tolist(),
        "axis_piece_local": local_axis.tolist(),
        "authored_axial_travel_m": travel,
        "original_nq": m.nq,
        "original_nv": m.nv,
        "nq": guided.nq,
        "nv": guided.nv,
        "plugin_count": guided.nplugin,
        "removed_joint_names": removed_joint_names,
        "initial_max_geom_position_error_m": max(position_errors, default=0),
        "initial_max_geom_rotation_entry_error": max(rotation_errors, default=0),
        "endpoint_max_geom_translation_error_m": endpoint_error,
        "endpoint_max_geom_rotation_entry_error": endpoint_rotation_error,
        "fixture": "EXPLICIT one-axis virtual-gripper fixture. All helper/receiver DOFs and active internal hinges locked at aligned start. Only axial slide is force integrated; no trajectory interpolation. Slide is unlimited, with no artificial seated stop. Original meshes, body inertias, SDF backend and sparse options retained. This does not demonstrate production robot insertion or grasp retention.",
    }
    output_scene.with_suffix(".guide.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
