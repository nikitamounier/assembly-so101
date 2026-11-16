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


def _calculate_supported_pose(
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


# This cache contains deterministic pose-authoring outputs, never native contact
# qualification. Each generated reset must still pass its independent simulation.
_SUPPORTED_CACHE_SCHEMA = 1
_SUPPORTED_ALGORITHM_SHA = "cfff8bd20d7e6ff0062d6d1e3201b89fe4fe52944463c9310a25af7dd1764930"


def _supported_pose_cache_key(source_mesh, receiver_mesh, center, pivot, axis, far_y, height, sign):
    import hashlib
    import json
    import scipy

    digest = hashlib.sha256()
    digest.update(f"supported-pose:{_SUPPORTED_CACHE_SCHEMA}:{_SUPPORTED_ALGORITHM_SHA}".encode())
    refinement = Path(__file__).with_name("wowrobo_grasp_refinement.py")
    if refinement.is_file():
        digest.update(hashlib.sha256(refinement.read_bytes()).digest())
    digest.update(json.dumps([np.__version__, scipy.__version__, trimesh.__version__]).encode())
    def add_array(value):
        if value is None:
            digest.update(b"none;")
            return
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError("Pose cache inputs must be exact numeric arrays")
        digest.update(json.dumps([array.dtype.str, array.shape]).encode())
        digest.update(memoryview(array).cast("B"))
    for mesh in (source_mesh, receiver_mesh):
        add_array(mesh.vertices)
        add_array(mesh.faces)
    for value in (center, pivot, axis, far_y, height, sign):
        add_array(value)
    return digest.hexdigest()


def _supported_pose_cache_load(path):
    import json

    with np.load(path, allow_pickle=False) as saved:
        schema = json.loads(str(saved["__meta__"].item()))
        if schema["version"] != _SUPPORTED_CACHE_SCHEMA:
            raise ValueError("Unsupported authoring-result cache schema")
        result = {}
        for key, kind in schema["fields"].items():
            value = saved[key].copy()
            if kind == "array":
                result[key] = value
            elif kind == "numpy_scalar":
                result[key] = value[()]
            elif kind == "python_scalar":
                result[key] = value.item()
            else:
                raise ValueError("Unexpected pose-cache field type")
    return result


def _supported_pose_cache_save(path, result):
    import json
    import os
    import tempfile

    fields, arrays = {}, {}
    for key, value in result.items():
        if key == "__meta__":
            raise ValueError("Reserved pose-cache field")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Pose cache outputs must be numeric")
        arrays[key] = array
        fields[key] = ("array" if isinstance(value, np.ndarray) else
                       "numpy_scalar" if isinstance(value, np.generic) else "python_scalar")
    arrays["__meta__"] = np.array(json.dumps({"version": _SUPPORTED_CACHE_SCHEMA, "fields": fields}))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def supported_pose(source_mesh, receiver_mesh, center, pivot, axis, far_y, height, sign, cache=None):
    import fcntl

    directory = Path(cache or (Path(__file__).resolve().parents[2] / "outputs/portable-rest-fields"))
    directory.mkdir(parents=True, exist_ok=True)
    key = _supported_pose_cache_key(source_mesh, receiver_mesh, center, pivot, axis, far_y, height, sign)
    path = directory / ("authoring-result-" + key + ".npz")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return _supported_pose_cache_load(path)
        result = _calculate_supported_pose(source_mesh, receiver_mesh, center, pivot, axis, far_y, height, sign, cache=directory)
        _supported_pose_cache_save(path, result)
        return result
