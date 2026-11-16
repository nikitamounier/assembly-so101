"""Native finger-span calibration for photo-directed cover, connector and pin grips.

Geometric pad registration only; no force, friction or loaded-grasp claim.
The source gripperframe's Z axis spans the jaws (Y is finger breadth).
"""

import mujoco
import numpy as np
import trimesh
from scipy.optimize import brentq

from assembly_policy.so101.wowrobo_layout import mesh_vertices


def outer_wall_contacts(layout, name, center=None, keys=None):
    center = (
        layout.grips[name]["center"].copy()
        if center is None
        else np.asarray(center).copy()
    )
    pieces = [
        trimesh.load_mesh(
            layout.assets / layout.manifest["constituents"][k]["mesh"], process=False
        )
        for k in (keys or layout.manifest["pieces"][name]["constituents"])
    ]
    mesh = trimesh.util.concatenate(pieces)
    segments = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, center[1], 0])
    xs = []
    for a, b in segments:
        if abs(b[2] - a[2]) < 1e-10:
            continue
        t = (center[2] - a[2]) / (b[2] - a[2])
        if 0 <= t <= 1:
            xs.append(float((a + t * (b - a))[0]))
    if len(xs) < 2:
        raise ValueError(f"No opposed exterior wall section: {name}")
    low, high = min(xs), max(xs)
    center[0] = (low + high) / 2
    contacts = np.tile(center, (2, 1))
    contacts[:, 0] = [low, high]
    return center, contacts


def native_jaw_grasp(model, width, depth, half_length=None):
    half_length = (
        (0.006 if width < 0.020 else 0.022) if half_length is None else half_length
    )
    data = mujoco.MjData(model)
    site = model.site("left_gripperframe").id
    joint = model.joint("left_gripper")
    address = joint.qposadr[0]
    gids = {}
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name
        if model.geom_type[i] == 7 and body in [
            "left_gripper",
            "left_moving_jaw_so101_v1",
        ]:
            gids[body] = i

    def inner_faces(opening):
        data.qpos[address] = opening
        mujoco.mj_forward(model, data)
        faces = []
        for body in ["left_gripper", "left_moving_jaw_so101_v1"]:
            gid = gids[body]
            mid = model.geom_dataid[gid]
            vertices = (
                mesh_vertices(model, data, gid) - data.site_xpos[site]
            ) @ data.site_xmat[site].reshape(3, 3)
            start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            mesh = trimesh.Trimesh(
                vertices, model.mesh_face[start : start + count], process=False
            )
            # Fit the complete occupied finger length, not a single plane.
            # Otherwise a converging jaw can clear the midpoint but cut into the
            # opposite end of the cylinder/case/printed wall.
            band = (vertices[:, 0] >= -depth - half_length) & (
                vertices[:, 0] <= -depth + half_length
            )
            points = [vertices[band]]
            for x in [-depth - half_length, -depth + half_length]:
                seg = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [x, 0, 0])
                if len(seg):
                    points.append(seg.reshape(-1, 3))
            points = np.vstack(points)
            if not len(points):
                raise ValueError("No finger geometry in grip envelope")
            z = points[:, 2]
            faces.append(float(z.max() if body == "left_gripper" else z.min()))
        return np.array(faces)

    # 0.3mm visual clearance per wall; physical pinch force remains unqualified.
    target = width + 0.0006
    last = None
    opening = None
    for candidate in np.linspace(
        float(joint.range[0]) + 0.0001, min(float(joint.range[1]), 1.3), 70
    ):
        try:
            value = float(np.diff(inner_faces(candidate))[0] - target)
        except ValueError:
            continue
        if last is not None and last[1] * value <= 0:
            opening = brentq(
                lambda q: np.diff(inner_faces(q))[0] - target,
                last[0],
                candidate,
                xtol=1e-10,
            )
            break
        last = (candidate, value)
    if opening is None:
        raise ValueError(
            f"No reachable native jaw envelope for width={width}, depth={depth}"
        )
    faces = inner_faces(opening)
    return {
        "span_site_axis": 2,
        "occupied_half_length_m": half_length,
        "center_offset_site": [-depth, 0, float(faces.mean())],
        "opening_radians": float(opening),
        "pad_inner_faces_site_z_m": faces.tolist(),
        "object_span_m": float(width),
        "pad_gap_m": float(np.diff(faces)[0]),
        "qualification": "mesh-section exterior-face fit only; physical grasp unqualified",
    }
