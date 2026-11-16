"""Compaction must keep solid coverage without bridging functional openings."""

import importlib.util
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

path = (
    Path(__file__).resolve().parents[1]
    / "assembly_policy/so101/wowrobo_collision_compaction.py"
)
spec = importlib.util.spec_from_file_location("collision_compaction", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(tmp_path, boxes, key):
    paths = []
    for i, (center, half) in enumerate(boxes):
        mesh = trimesh.creation.box(extents=2 * np.array(half))
        mesh.apply_translation(center)
        p = tmp_path / f"{key}-{i}.obj"
        mesh.export(p)
        paths.append(str(p))
    return {"key": key, "count": len(paths), "paths": paths}


def box_distance(points, center, half):
    q = np.abs(points - np.array(center)) - np.array(half)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def test_solid_partition_merges_to_one_without_losing_extent(tmp_path):
    boxes = [([x, 0, 0], [0.005, 0.01, 0.01]) for x in [-0.005, 0.005]]
    r = record(tmp_path, boxes, "solid")
    result = module.compact_parts(
        r,
        tmp_path / "out",
        lambda p: box_distance(p, [0, 0, 0], [0.01, 0.01, 0.01]),
        "solid",
        target=1,
    )
    assert result["count"] == 1
    vertices = module._read(result["paths"][0])
    np.testing.assert_allclose(vertices.min(0), [-0.01] * 3, atol=1e-12)
    np.testing.assert_allclose(vertices.max(0), [0.01] * 3, atol=1e-12)


def test_native_void_guard_keeps_ring_open_even_when_target_is_one(tmp_path):
    boxes = [([0, y, 0], [0.02, 0.005, 0.005]) for y in [-0.015, 0.015]] + [
        ([x, 0, 0], [0.005, 0.01, 0.005]) for x in [-0.015, 0.015]
    ]
    r = record(tmp_path, boxes, "ring")

    def oracle(p):
        return np.min([box_distance(p, c, h) for c, h in boxes], axis=0)

    result = module.compact_parts(
        r, tmp_path / "out", oracle, "ring", target=1, added_material_guard=0.0001
    )
    assert result["count"] > 1
    assert result["compaction"]["rejected"] > 0
    points = np.array([[0, 0, z] for z in [-0.004, 0, 0.004]])
    for path in result["paths"]:
        h = ConvexHull(module._read(path))
        assert not np.any(
            np.all(points @ h.equations[:, :3].T + h.equations[:, 3] <= 1e-10, axis=1)
        )


def test_native_grid_bounds_and_box_voxels_preserve_outer_planes(tmp_path):
    import struct

    coordinates = np.linspace(-0.01, 0.01, 21)
    points = np.stack(
        np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), -1
    )
    half = np.array([0.004, 0.006, 0.008])
    values = box_distance(points.reshape(-1, 3), [0, 0, 0], half).reshape(21, 21, 21)
    path = tmp_path / "grid.bin"
    path.write_bytes(
        struct.pack("<3i4d", 21, 21, 21, -0.01, -0.01, -0.01, 0.001)
        + values.transpose(2, 1, 0).astype("<f2").tobytes()
    )
    bounds = module.grid_zero_bounds(path)
    np.testing.assert_allclose(bounds, [-half, half], atol=1e-9)
    result = module.voxel_parts(
        bounds,
        tmp_path / "boxes",
        lambda p: box_distance(p, [0, 0, 0], half),
        "box",
        pitch=0.001,
    )
    assert result["count"] == 1
    v = module._read(result["paths"][0])
    np.testing.assert_allclose(v.min(0), -half, atol=1e-9)
    np.testing.assert_allclose(v.max(0), half, atol=1e-9)


def test_voxel_boxes_keep_a_square_insertion_channel_open(tmp_path):
    boxes = [([0, y, 0], [0.02, 0.005, 0.005]) for y in [-0.015, 0.015]] + [
        ([x, 0, 0], [0.005, 0.01, 0.005]) for x in [-0.015, 0.015]
    ]

    def oracle(p):
        return np.min([box_distance(p, c, h) for c, h in boxes], axis=0)

    result = module.voxel_parts(
        [[-0.02, -0.02, -0.005], [0.02, 0.02, 0.005]],
        tmp_path / "voxel-ring",
        oracle,
        "ring",
        pitch=0.002,
    )
    volume = 0
    for path in result["paths"]:
        h = ConvexHull(module._read(path))
        volume += h.volume
        assert not np.all(h.equations[:, 3] <= 1e-10)
    np.testing.assert_allclose(volume, (0.04**2 - 0.02**2) * 0.01, rtol=1e-12)
