"""Compare the optimized supplement against its original exhaustive oracle."""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.scene import model
from assembly_policy.so101.task_scenes import geometric_scene
from assembly_policy.so101.workcell import load_contact_search


@pytest.fixture(scope="module")
def plant():
    return model(contact_monitor=True)


def replay(m, qpos, mode, steps=24):
    native = load_contact_search()
    previous = native.so101_surface_mode(mode)
    try:
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        result = []
        for _ in range(steps):
            mujoco.mj_step(m, d)
            result.append(
                (
                    d.qpos.copy(),
                    d.qvel.copy(),
                    np.array([np.r_[c.dist, c.pos, c.frame.ravel(), c.geom] for c in d.contact]),
                )
            )
        return result
    finally:
        native.so101_surface_mode(previous)


def assert_same(a, b):
    for left, right in zip(a, b, strict=True):
        for x, y in zip(left, right, strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("task", ["motor", "horn", "yoke", "retainer", "screw"])
def test_contact_positive_perturbations_match_exhaustive_scan(plant, task):
    d = geometric_scene(plant, task)
    rng = np.random.default_rng(20489)
    for _ in range(6):
        qpos = d.qpos.copy()
        # Perturb all free objects around geometric mating poses, in either
        # direction across material boundaries, without claiming valid resets.
        for name in ["base", "motor", "horn", "yoke", "retainer", "screw"]:
            address = int(plant.joint(name + "_free").qposadr[0])
            qpos[address : address + 3] += rng.uniform(-0.003, 0.003, 3)
        assert_same(replay(plant, qpos, 0), replay(plant, qpos, 1))


def test_shared_model_parallel_threads_match_exhaustive_scan(plant):
    poses = [geometric_scene(plant, task).qpos.copy() for task in ["motor", "horn", "screw"]]
    expected = [replay(plant, q, 0, 40) for q in poses]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda q: replay(plant, q, 1, 40), poses))
    for a, b in zip(expected, actual, strict=True):
        assert_same(a, b)


def test_grid_bounds_never_reject_a_sample_below_contact_margin(plant):
    """Independent trilinear oracle includes clamped cells outside the grid."""
    import ctypes
    import struct
    from pathlib import Path

    from assembly_policy.so101.workcell import ROOT

    sdf = ctypes.CDLL(str(ROOT / "plugin/build/libassemblenist_sdf.so"))
    library = ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob("libmujoco*"))))
    config = library.mj_getPluginConfig
    config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
    config.restype = ctypes.c_char_p
    query = sdf.assemblenist_grid_box_separated
    ptr = ctypes.POINTER(ctypes.c_double)
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ptr, ptr, ctypes.c_double]
    query.restype = ctypes.c_int
    d = mujoco.MjData(plant)
    rng = np.random.default_rng(2911)
    rejected = 0
    for part in ["base", "motor", "horn"]:
        target = plant.geom(part + "_contact").id
        instance = int(plant.geom_plugin[target])
        grid_id = config(plant._address, instance, b"id").decode()
        path = ROOT / f"assets_local/sdf/grid_{grid_id}.bin"
        header = struct.unpack("<3i4d", path.read_bytes()[:44])
        n, origin, h = np.array(header[:3]), np.array(header[3:6]), header[6]
        grid = np.memmap(path, dtype="<f2", mode="r", offset=44, shape=tuple(n[::-1]))
        for _ in range(150):
            center = origin + rng.uniform(-0.1, 1.1, 3) * (n - 1) * h
            radius = rng.uniform(0, 8, 3) * h
            lo, hi = np.ascontiguousarray(center - radius), np.ascontiguousarray(center + radius)
            threshold = float(rng.choice([-0.0001, 0, 0.0001]))
            if not query(
                plant._address, d._address, target, lo.ctypes.data_as(ptr), hi.ctypes.data_as(ptr), threshold
            ):
                continue
            rejected += 1
            points = np.vstack((lo, hi, rng.uniform(lo, hi, (64, 3))))
            f = (points - origin) / h
            c = np.clip(f, 0, n - 1.000001)
            i = np.floor(c).astype(int)
            t = c - i

            def at(x, y, z):
                return grid[i[:, 2] + z, i[:, 1] + y, i[:, 0] + x].astype(float)

            c00 = at(0, 0, 0) + (at(1, 0, 0) - at(0, 0, 0)) * t[:, 0]
            c10 = at(0, 1, 0) + (at(1, 1, 0) - at(0, 1, 0)) * t[:, 0]
            c01 = at(0, 0, 1) + (at(1, 0, 1) - at(0, 0, 1)) * t[:, 0]
            c11 = at(0, 1, 1) + (at(1, 1, 1) - at(0, 1, 1)) * t[:, 0]
            c0, c1 = c00 + (c10 - c00) * t[:, 1], c01 + (c11 - c01) * t[:, 1]
            distance = c0 + (c1 - c0) * t[:, 2] + np.linalg.norm((f - c) * h, axis=1)
            assert np.all(distance > threshold)
    assert rejected > 30, "The panel must exercise actual positive rejection"
