"""Opt-in stock-MuJoCo cuboids preserving sampled native material voids.

Offline construction may query the original field. Runtime uses only primitives,
not field plugins, and never merges material across an unoccupied voxel.
"""

import itertools
import xml.etree.ElementTree as ET

import numpy as np


def field_cuboids(bounds, query, *, pitch=0.00075, corner_allowance=0.00015):
    """Partition occupied cells; reject cells extending into sampled voids.

    Centers must be inside the source and all eight corners no farther than
    `corner_allowance` outside. This is sampled geometry, not an error bound.
    """
    bounds = np.asarray(bounds, float)
    counts = np.maximum(1, np.ceil((bounds[1] - bounds[0]) / pitch).astype(int))
    if np.prod(counts) > 4_000_000:
        raise ValueError("Field cuboid budget exceeded")
    delta = (bounds[1] - bounds[0]) / counts
    centers = [
        bounds[0, k] + (np.arange(n) + 0.5) * delta[k] for k, n in enumerate(counts)
    ]
    points = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.asarray(query(points))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite construction oracle")
    inside = values <= 0
    candidate = points[inside]
    valid = np.ones(len(candidate), bool)
    for signs in itertools.product((-1, 1), repeat=3):
        distance = np.asarray(query(candidate + np.asarray(signs) * delta / 2))
        valid &= np.isfinite(distance) & (distance <= corner_allowance + 1e-12)
    occupied = np.zeros(len(points), bool)
    occupied[np.flatnonzero(inside)] = valid
    occupied = occupied.reshape(tuple(counts))
    material_cells = int(occupied.sum())
    flat = occupied.ravel()
    cursor = 0
    boxes = []
    while cursor < len(flat):
        end = min(cursor + 4096, len(flat))
        found = np.flatnonzero(flat[cursor:end])
        if not len(found):
            cursor = end
            continue
        index = cursor + int(found[0])
        cursor = index + 1
        x, y, z = map(int, np.unravel_index(index, tuple(counts)))
        xe, ye, ze = x + 1, y + 1, z + 1
        while xe < counts[0] and occupied[xe, y, z]:
            xe += 1
        while ye < counts[1] and occupied[x:xe, ye, z].all():
            ye += 1
        while ze < counts[2] and occupied[x:xe, y:ye, ze].all():
            ze += 1
        occupied[x:xe, y:ye, z:ze] = False
        lo = bounds[0] + np.array([x, y, z]) * delta
        hi = bounds[0] + np.array([xe, ye, ze]) * delta
        boxes.append(
            {
                "position": ((lo + hi) / 2).tolist(),
                "half_size": ((hi - lo) / 2).tolist(),
            }
        )
    if not boxes:
        raise ValueError("No material survived primitive construction")
    return boxes, {
        "pitch_m": pitch,
        "corner_allowance_m": corner_allowance,
        "grid_shape": counts.tolist(),
        "material_cells": material_cells,
        "boxes": len(boxes),
    }


def apply_cuboids(root, source, replacements):
    """Replace selected collision geoms; restore interactions across assemblies."""
    import mujoco

    numbers = lambda v: " ".join(format(float(x), ".17g") for x in v)
    mapping = {}
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name")
            if name not in replacements:
                continue
            gid = source.geom(name).id
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, source.geom_quat[gid])
            rotation = rotation.reshape(3, 3)
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            names = []
            finger = name.startswith("physical_helper_")
            for i, box in enumerate(replacements[name]):
                attrs = dict(geom.attrib)
                attrs.update(
                    name=name + f"__cell_{i}",
                    type="box",
                    size=numbers(box["half_size"]),
                    pos=numbers(source.geom_pos[gid] + rotation @ box["position"]),
                    quat=numbers(source.geom_quat[gid]),
                    contype="4" if finger else "2",
                    conaffinity="3" if finger else "7",
                    mass="0",
                    group="3",
                    margin="0",
                    rgba=".1 .7 .4 1" if finger else ".9 .5 .15 1",
                )
                if "quaternion" in box:
                    local_rotation = np.empty(9)
                    mujoco.mju_quat2Mat(local_rotation, np.asarray(box["quaternion"]))
                    composed = np.empty(4)
                    mujoco.mju_mat2Quat(
                        composed, (rotation @ local_rotation.reshape(3, 3)).ravel()
                    )
                    attrs["quat"] = numbers(composed)
                attrs["margin"] = str(box.get("margin", 0))
                ET.SubElement(body, "geom", attrs)
                names.append(attrs["name"])
            mapping[name] = names
    # Servo pivots within one preassembled component must not collide with its
    # own housing. Contacts between different assembly components stay enabled.
    contact = ET.SubElement(root, "contact")
    groups = {}
    roots = {
        "base_servo",
        "base_side_cover",
        "shoulder_connector",
        "assembly_pin",
        "upper_arm_module",
        "forearm_module",
    }
    for bid in range(1, source.nbody):
        cursor = bid
        while cursor and source.body(cursor).name not in roots:
            cursor = int(source.body_parentid[cursor])
        if cursor:
            groups.setdefault(source.body(cursor).name, []).append(
                source.body(bid).name
            )
    for bodies in groups.values():
        for a, b in itertools.combinations(bodies, 2):
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    return mapping, groups
