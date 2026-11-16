"""Optional convex-piece compaction with a sampled native-material oracle.

The oracle is a guard, not a global error certificate. Dense occupancy and
functional opening/grasp checks remain necessary for each candidate release.
"""

import hashlib
import heapq
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def grid_zero_bounds(path):
    """Bounds of a trilinear grid's zero set, including outside extension."""
    with Path(path).open("rb") as stream:
        nx, ny, nz, ox, oy, oz, h = struct.unpack("<3i4d", stream.read(44))
    grid = np.asarray(
        np.memmap(path, dtype="<f2", mode="r", offset=44, shape=(nz, ny, nx)),
        dtype=float,
    )
    if not np.isfinite(grid).all() or h <= 0:
        raise ValueError("Invalid native distance grid")
    origin = np.array([ox, oy, oz])
    bounds = np.array([[np.inf] * 3, [-np.inf] * 3])
    for axis, size in enumerate([nx, ny, nz]):
        dimension = 2 - axis
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[dimension] = slice(0, -1)
        right[dimension] = slice(1, None)
        a, b = grid[tuple(left)], grid[tuple(right)]
        mask = ((a <= 0) & (b >= 0)) | ((a >= 0) & (b <= 0))
        if mask.any():
            indices = np.nonzero(mask)[dimension]
            av, bv = a[mask], b[mask]
            t = np.divide(av, av - bv, out=np.zeros_like(av), where=av != bv)
            positions = origin[axis] + (indices + t) * h
            bounds[0, axis] = positions.min()
            bounds[1, axis] = positions.max()
            both_zero = (av == 0) & (bv == 0)
            if both_zero.any():
                bounds[1, axis] = max(
                    bounds[1, axis], (origin[axis] + (indices[both_zero] + 1) * h).max()
                )
        # grid_eval adds nonnegative Euclidean distance outside the clamped
        # domain. A negative boundary value can extend its zero beyond it.
        low = float(np.take(grid, 0, axis=dimension).min())
        high = float(np.take(grid, -1, axis=dimension).min())
        if low <= 0:
            bounds[0, axis] = min(bounds[0, axis], origin[axis] + low)
        if high <= 0:
            bounds[1, axis] = max(bounds[1, axis], origin[axis] + (size - 1) * h - high)
    if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Native case has no bounded zero surface")
    return bounds


def _piece(vertices):
    h = ConvexHull(vertices)
    vertices = np.asarray(vertices)[h.vertices]
    h = ConvexHull(vertices)
    return {"v": vertices, "h": h, "center": vertices.mean(0), "volume": h.volume}


def _read(path):
    return np.array(
        [
            [float(x) for x in line.split()[1:4]]
            for line in Path(path).read_text().splitlines()
            if line.startswith("v ")
        ]
    )


def _write(path, piece):
    v, h = piece["v"], piece["h"]
    faces = h.simplices.copy()
    for i, face in enumerate(faces):
        if (
            np.cross(v[face[1]] - v[face[0]], v[face[2]] - v[face[0]])
            @ h.equations[i, :3]
            < 0
        ):
            faces[i, [1, 2]] = faces[i, [2, 1]]
    with path.open("w") as f:
        for point in v:
            f.write("v " + " ".join(format(float(x), ".17g") for x in point) + "\n")
        for face in faces:
            f.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")


def compact_parts(
    record, output, query, oracle_id, *, target=64, added_material_guard=0.0005
):
    """Merge cuts, rejecting sampled additions farther than guard into void.

    `query` receives points in the same frame as the input OBJ files. The
    original convex union remains covered; compaction can only add material.
    The target is soft: preserve the guard when no accepted merge remains.
    """
    if record["count"] <= target:
        return record
    settings = {
        "target": target,
        "added_material_guard": added_material_guard,
        "oracle_id": oracle_id,
        "source_key": record["key"],
        "seed": 41827,
        "algorithm": "volume-priority-native-added-material-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = Path(output) / key
    receipt = directory / "receipt.json"
    if receipt.exists():
        return {**record, **json.loads(receipt.read_text())}
    directory.mkdir(parents=True, exist_ok=True)
    active = {i: _piece(_read(path)) for i, path in enumerate(record["paths"])}
    heap, seen = [], set()

    def add_pair(a, b):
        if a == b:
            return
        a, b = sorted((int(a), int(b)))
        if (a, b) in seen:
            return
        seen.add((a, b))
        try:
            merged = _piece(np.concatenate([active[a]["v"], active[b]["v"]]))
        except Exception as exc:
            raise RuntimeError(f"Cannot construct convex merge {a}/{b}") from exc
        cost = max(0.0, merged["volume"] - active[a]["volume"] - active[b]["volume"])
        heapq.heappush(heap, (cost, a, b))

    def neighbors(identifier):
        ids = list(active)
        tree = cKDTree(np.array([active[i]["center"] for i in ids]))
        _, indices = tree.query(active[identifier]["center"], k=min(13, len(ids)))
        for j in np.atleast_1d(indices):
            add_pair(identifier, ids[int(j)])

    for identifier in list(active):
        neighbors(identifier)
    rng = np.random.default_rng(41827)
    next_id, accepted, rejected, oracle_samples = len(active), 0, 0, 0
    while len(active) > target and heap:
        _, a, b = heapq.heappop(heap)
        if a not in active or b not in active:
            continue
        first, second = active[a], active[b]
        merged = _piece(np.concatenate([first["v"], second["v"]]))
        v, h = merged["v"], merged["h"]
        faces = v[h.simplices]
        # Boundary, interior and bridge probes focus on material that the two
        # original pieces did not contain. Existing approximation errors are
        # assessed separately, rather than counted as new merge material.
        chosen = faces[rng.integers(0, len(faces), 64)]
        barycentric = rng.dirichlet(np.ones(4), 64)
        interior = (chosen * barycentric[:, :3, None]).sum(1) + merged[
            "center"
        ] * barycentric[:, 3, None]
        probes = np.concatenate([v, faces.mean(1), interior, merged["center"][None]])
        in_first = np.all(
            probes @ first["h"].equations[:, :3].T + first["h"].equations[:, 3] <= 1e-8,
            axis=1,
        )
        in_second = np.all(
            probes @ second["h"].equations[:, :3].T + second["h"].equations[:, 3]
            <= 1e-8,
            axis=1,
        )
        added = probes[~in_first & ~in_second]
        if len(added):
            distances = np.asarray(query(added))
            oracle_samples += len(added)
            if not np.isfinite(distances).all() or np.any(
                distances > added_material_guard
            ):
                rejected += 1
                continue
        del active[a], active[b]
        active[next_id] = merged
        neighbors(next_id)
        next_id += 1
        accepted += 1
    paths = []
    for i, piece in enumerate(active.values()):
        path = directory / f"hull-{i:03d}.obj"
        _write(path, piece)
        paths.append(str(path.resolve()))
    result = {
        "key": key,
        "source_key": record["key"],
        "paths": paths,
        "count": len(paths),
        "original_count": record["count"],
        "compaction": {
            "settings": settings,
            "accepted": accepted,
            "rejected": rejected,
            "oracle_samples": oracle_samples,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return {**record, **result}


def voxel_parts(bounds, output, query, oracle_id, *, pitch=0.0005):
    """Greedily merged boxes sampled from the native field, not capped CAD.

    This is reserved for known open vendor case meshes. Bounds are derived from
    their existing native zero surface, retaining outer gripping dimensions.
    """
    bounds = np.asarray(bounds, float)
    settings = {
        "algorithm": "native-field-greedy-boxes-v1",
        "bounds": bounds.tolist(),
        "pitch": pitch,
        "oracle_id": oracle_id,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = Path(output) / key
    receipt = directory / "receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    counts = np.maximum(1, np.ceil((bounds[1] - bounds[0]) / pitch).astype(int))
    if np.prod(counts) > 12_000_000:
        raise ValueError("Native-case voxel budget exceeded")
    edges = [np.linspace(bounds[0, k], bounds[1, k], counts[k] + 1) for k in range(3)]
    centers = [(a[:-1] + a[1:]) / 2 for a in edges]
    points = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 3)
    distances = np.asarray(query(points))
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite native case distance")
    occupied = (distances <= 0).reshape(tuple(counts))
    material_cells = int(occupied.sum())
    paths = []
    flat = occupied.ravel()
    cursor = 0
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
        v = np.array(
            [
                [edges[0][a], edges[1][b], edges[2][c]]
                for a in [x, xe]
                for b in [y, ye]
                for c in [z, ze]
            ]
        )
        path = directory / f"box-{len(paths):04d}.obj"
        _write(path, _piece(v))
        paths.append(str(path.resolve()))
    if not paths:
        raise ValueError("Native field produced no case material")
    result = {
        "key": key,
        "paths": paths,
        "count": len(paths),
        "proxy_source": "native-SDF voxel boxes",
        "settings": settings,
        "material_cells": material_cells,
        "voxel_samples": len(points),
    }
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return result
