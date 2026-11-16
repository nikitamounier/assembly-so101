"""Small source-material contact patches, fitted offline to native witnesses."""

import itertools

import mujoco
import numpy as np


def patch_from_contact(model, data, gid, contact, side, query):
    normal = np.asarray(contact["normal"], float)
    normal /= np.linalg.norm(normal)
    outward = normal if side == 0 else -normal
    surface = (
        np.asarray(contact["pos"])
        + (-0.5 if side == 0 else 0.5) * contact["dist"] * normal
    )
    field = float(query(gid, surface[None])[0])
    surface -= outward * max(0, field - 0.00005)
    axis = (
        np.array([1.0, 0.0, 0.0])
        if abs(outward[0]) < 0.8
        else np.array([0.0, 1.0, 0.0])
    )
    tangent = np.cross(outward, axis)
    tangent /= np.linalg.norm(tangent)
    frame = np.column_stack([tangent, np.cross(outward, tangent), outward])
    depth = 0.0005
    center = surface - outward * depth / 2
    for radius in (0.0015, 0.00075, 0.00035, 0.00015):
        half = np.array([radius, radius, depth / 2])
        corners = np.array(list(itertools.product((-1, 1), repeat=3))) * half
        world = corners @ frame.T + center
        values = query(gid, world)
        if np.isfinite(values).all() and values.max() <= 0.00015:
            rotation = data.geom_xmat[gid].reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rotation.T @ frame).ravel())
            return {
                "position": ((center - data.geom_xpos[gid]) @ rotation).tolist(),
                "half_size": half.tolist(),
                "quaternion": quat.tolist(),
                "margin": 0.00015,
                "source": "native contact witness; 8 corners checked against original material",
            }
    return None
