import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location(
        "supported_pose_cache_test", HERE / "wowrobo_supported_poses.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def mesh():
    return SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )


def test_exact_results_and_no_mutable_cache_alias(module, monkeypatch, tmp_path):
    calls = []

    def calculate(*args, **kwargs):
        calls.append(True)
        return {
            "rotation": np.eye(3),
            "position": np.array([1, 2, 3], dtype=np.float32),
            "shaft_angle": 0.12345678901234567,
            "numpy_scalar": np.float32(0.234),
            "far_contact_gap_m": -1e-8,
        }

    monkeypatch.setattr(module, "_calculate_supported_pose", calculate)
    source, receiver = mesh(), mesh()
    args = (
        source,
        receiver,
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.06,
        0.025,
        None,
    )
    first = module.supported_pose(*args, cache=tmp_path)
    expected = {
        k: v.copy() if isinstance(v, np.ndarray) else v for k, v in first.items()
    }
    first["rotation"][0, 0] = 123.0
    second = module.supported_pose(*args, cache=tmp_path)
    assert len(calls) == 1
    for key, value in expected.items():
        np.testing.assert_array_equal(second[key], value)
        assert type(second[key]) is type(value)
        if isinstance(value, np.ndarray):
            assert second[key].dtype == value.dtype
    second["position"][0] = 99.0
    third = module.supported_pose(*args, cache=tmp_path)
    np.testing.assert_array_equal(third["position"], expected["position"])
    assert len(calls) == 1


def test_every_numeric_input_and_mesh_bytes_invalidates(module, monkeypatch, tmp_path):
    calls = []

    def calculate(*args, **kwargs):
        calls.append(True)
        return {"value": 1.0}

    monkeypatch.setattr(module, "_calculate_supported_pose", calculate)
    base = [
        mesh(),
        mesh(),
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.06,
        0.025,
        None,
    ]
    module.supported_pose(*base, cache=tmp_path)
    for which in range(8):
        args = [
            mesh(),
            mesh(),
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.06,
            0.025,
            None,
        ]
        if which < 2:
            args[which].vertices[0, 0] = 1e-15
        elif which < 5:
            args[which][0] += 1e-15
        elif which < 7:
            args[which] += 1e-15
        else:
            args[which] = -1
        module.supported_pose(*args, cache=tmp_path)
    assert len(calls) == 9
    args = [mesh(), mesh(), *base[2:]]
    args[0].faces[0] = [1, 0, 2]
    module.supported_pose(*args, cache=tmp_path)
    assert len(calls) == 10


def test_algorithm_fingerprint_and_corruption_are_not_ignored(
    module, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        module,
        "_calculate_supported_pose",
        lambda *a, **k: calls.append(True) or {"value": 1.0},
    )
    args = (
        mesh(),
        mesh(),
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.06,
        0.025,
        -1,
    )
    module.supported_pose(*args, cache=tmp_path)
    monkeypatch.setattr(module, "_SUPPORTED_ALGORITHM_SHA", "changed")
    module.supported_pose(*args, cache=tmp_path)
    assert len(calls) == 2
    key = module._supported_pose_cache_key(*args)
    (tmp_path / f"authoring-result-{key}.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        module.supported_pose(*args, cache=tmp_path)
    assert len(calls) == 2
