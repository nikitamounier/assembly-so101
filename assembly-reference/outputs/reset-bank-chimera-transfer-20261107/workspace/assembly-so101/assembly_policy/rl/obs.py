"""Reusable surface sampling and batched quaternion transforms."""

import numpy as np
import torch


def sample_surface(verts, faces, n, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _box_surface(size, n, rng):
    hx, hy, hz = size
    areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy])
    face = rng.choice(6, n, p=areas / areas.sum())
    p = rng.uniform(-1, 1, (n, 3)) * np.array([hx, hy, hz])
    axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
    p[np.arange(n), axis] = sign * np.array([hx, hy, hz])[axis]
    return p


def quat_to_mat_torch(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(q.shape[:-1] + (3, 3))
