"""Register far-side resting poses against actual receiver mesh surfaces.

Pose authoring only: no load-bearing contact or reward qualification.
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def vertical_surface(mesh, x, y, upper=True):
    t = mesh.triangles
    a = t[:, 0]
    u = t[:, 1] - a
    v = t[:, 2] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-14
    d = np.array([x, y]) - a[:, :2]
    safe = np.where(valid, det, 1)
    s = (d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) / safe
    q = (u[:, 0] * d[:, 1] - u[:, 1] * d[:, 0]) / safe
    valid &= (s >= -1e-8) & (q >= -1e-8) & (s + q <= 1 + 1e-8)
    if not np.any(valid):
        return None
    z = a[:, 2] + s * u[:, 2] + q * v[:, 2]
    return float(z[valid].max() if upper else z[valid].min())


def supported_pose(
    source_mesh, receiver_mesh, center, pivot, axis, far_y, height, sign, cache=None
):
    from assembly_policy.so101.wowrobo_grasp_refinement import MeshField

    cache = Path(cache or (Path(__file__).resolve().parents[2] / "outputs/portable-rest-fields"))
    cache.mkdir(parents=True, exist_ok=True)
    field = MeshField(receiver_mesh, cache)
    points = np.vstack([source_mesh.vertices, source_mesh.triangles_center])
    points = points[points[:, 1] > 0.035] if far_y > 0 else points[points[:, 1] < 0.010]
    if len(points) > 1024:
        points = points[np.linspace(0, len(points) - 1, 1024, dtype=int)]
    center = np.asarray(center)
    pivot = np.asarray(pivot)
    axis = np.asarray(axis)
    target = center + np.array([0, 0, height])
    direction = -1 if far_y > 0 else 1

    exact_points = None

    def evaluate(angle, exact=False):
        r = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        hinge = Rotation.from_rotvec(axis * angle).as_matrix()
        moved_center = pivot + hinge @ (center - pivot)
        pos = target - r @ moved_center
        sampled = exact_points if exact and exact_points is not None else points
        world = sampled @ r.T + pos
        if exact:
            nearest, dist, tri = trimesh.proximity.closest_point(receiver_mesh, world)
            dot = np.einsum(
                "ij,ij->i", world - nearest, receiver_mesh.face_normals[tri]
            )
            gaps = np.where(dot >= 0, dist, -dist)
        else:
            gaps = field.query(world) - 0.0005
        i = int(np.argmin(gaps))
        return float(gaps[i]), r, pos, world[i], hinge, sampled[i]

    previous = None
    angle = None
    for a in np.linspace(-direction * 1.25, direction * 1.25, 160):
        gap = evaluate(a)[0]
        if previous and previous[1] > 0 and gap <= 0:
            angle = brentq(
                lambda x: evaluate(x)[0],
                min(previous[0], a),
                max(previous[0], a),
                xtol=1e-10,
            )
            break
        previous = (a, gap)
    if angle is None:
        raise ValueError(f"No first far-side contact at height {height}")
    # Refine the coarse field contact against real triangles. The first contact
    # uses the complete far-end module, including its hanging servo casing.
    _, r0, p0, *_ = evaluate(angle)
    exact_points = points[np.argsort(field.query(points @ r0.T + p0))[:64]]
    lo, hi = angle - direction * 0.055, angle + direction * 0.055
    flo, fhi = evaluate(lo, True)[0], evaluate(hi, True)[0]
    if flo * fhi <= 0:
        angle = brentq(
            lambda x: evaluate(x, True)[0], min(lo, hi), max(lo, hi), xtol=1e-10
        )
    gap, r, pos, contact, hinge, far = evaluate(angle, True)
    return {
        "rotation": r,
        "position": pos,
        "shaft_angle": angle,
        "far_point": far,
        "receiver_contact": contact,
        "far_contact_gap_m": float(gap),
        "case_center": target,
        "case_rotation": r @ hinge,
    }
