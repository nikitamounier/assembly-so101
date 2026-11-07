"""Independent sampled CAD overlap checks; never used to move a live part."""

from functools import lru_cache
from itertools import combinations

import mujoco
import numpy as np


@lru_cache(maxsize=6)
def _part(path):
    with np.load(path) as data:
        vertices, faces = data["vertices"].copy(), data["faces"].copy()
    return vertices, faces, np.vstack((vertices, vertices[faces].mean(axis=1)))


@lru_cache(maxsize=6)
def _trees(path):
    import igl

    vertices, faces, _ = _part(path)
    distance = igl.AABB()
    distance.init(vertices, faces)
    winding = igl.FastWindingNumberBVH()
    winding.init(vertices, faces)
    return distance, winding


def sampled_pair_overlap(data, first, second, asset_root):
    """Bidirectional original-surface penetration, with immutable mesh trees."""
    worst = 0.0
    for name, other in ((first, second), (second, first)):
        source = str(asset_root / "parts" / f"{name}.npz")
        target = str(asset_root / "parts" / f"{other}.npz")
        body, receiver = data.body(name), data.body(other)
        points = (
            _part(source)[2] @ body.xmat.reshape(3, 3).T + body.xpos - receiver.xpos
        ) @ receiver.xmat.reshape(3, 3)
        vertices, faces, _ = _part(target)
        points = points[np.all((points >= vertices.min(0)) & (points <= vertices.max(0)), axis=1)]
        if not len(points):
            continue
        distance, winding = _trees(target)
        inside = winding.winding_number(points) > 0.5
        if inside.any():
            squared = distance.squared_distance(vertices, faces, points[inside])[0]
            worst = max(worst, float(np.sqrt(np.maximum(squared, 0)).max()))
    return worst


def table_overlaps(data, names, asset_root):
    """Exact original-triangle depth against the shared table plane at this pose."""
    table = data.model.geom("table")
    if int(table.type[0]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise ValueError("Original-material table audit requires a plane")
    plane = data.geom("table")
    normal = plane.xmat.reshape(3, 3)[:, 2]
    result = {}
    for name in names:
        if name == "screw":
            continue
        body = data.body(name)
        vertices = _part(str(asset_root / "parts" / f"{name}.npz"))[0]
        # A linear plane-distance function attains each triangle's minimum at
        # a vertex, so this check has no within-triangle sampling uncertainty.
        heights = vertices @ (body.xmat.reshape(3, 3).T @ normal) + (body.xpos - plane.xpos) @ normal
        result[name + "/table"] = max(0.0, -float(heights.min()))
    return result


def sampled_assembly_overlaps(data, names, asset_root):
    """Regular predecessor pairs and exact table depth; analytic screw is separate."""
    names = [name for name in names if name != "screw"]
    result = {a + "/" + b: sampled_pair_overlap(data, a, b, asset_root) for a, b in combinations(names, 2)}
    result.update(table_overlaps(data, names, asset_root))
    return result


def sampled_motor_base_overlap(model, qpos, asset_root):
    """Return sampled material overlap in metres, distinct from contact.dist."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return sampled_pair_overlap(data, "motor", "base", asset_root)


def finger_surface_clearance(model, data, side):
    """Sample original visual triangles at the distal fingers, not stop proxies."""
    import igl

    palm = data.body(side + "_gripper")
    rotation = palm.xmat.reshape(3, 3)
    surfaces = []
    for body, suffix in (
        (side + "_gripper", "wrist_roll_follower_so101_v1"),
        (side + "_moving_jaw_so101_v1", "moving_jaw_so101_v1"),
    ):
        geom = next(
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.body(body).id
            and model.geom_group[g] == 2
            and model.mesh(model.geom_dataid[g]).name == side + "_" + suffix
        )
        mesh = model.geom_dataid[geom]
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        f, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = (
            model.mesh_vert[a : a + n] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
            - palm.xpos
        ) @ rotation
        faces = model.mesh_face[f : f + nf]
        samples = np.vstack((vertices, vertices[faces].mean(axis=1)))
        surfaces.append((vertices, faces, samples[samples[:, 2] < -0.065]))
    gap = float("inf")
    overlap = 0.0
    for index in (0, 1):
        points = surfaces[index][2]
        vertices, faces, _ = surfaces[1 - index]
        dist = np.sqrt(igl.point_mesh_squared_distance(points, vertices, faces)[0])
        inside = igl.fast_winding_number(vertices, faces, points) > 0.5
        gap = min(gap, float(dist.min()))
        overlap = max(overlap, float(np.where(inside, dist, 0).max()))
    return dict(sampled_surface_gap_m=gap, max_sampled_material_overlap_m=overlap)
