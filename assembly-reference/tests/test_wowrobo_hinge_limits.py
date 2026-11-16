import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from assembly_policy.so101.wowrobo_hinge_limits import DEFAULT_NOISE_RAD, sample_noise


def test_noise_is_small_reproducible_and_keeps_reset_inside_limits():
    bounds = np.array([[-0.625, 3.766], [-3.766, 0.624]])
    q = np.array([-0.6249, 0.6239])
    a, b = np.random.default_rng(125), np.random.default_rng(125)
    for _ in range(100):
        x, y = sample_noise(a, bounds, q), sample_noise(b, bounds, q)
        np.testing.assert_array_equal(x, y)
        assert np.max(np.abs(x)) <= DEFAULT_NOISE_RAD
        perturbed = bounds + x.reshape(2, 2)
        assert np.all(q >= perturbed[:, 0]) and np.all(q <= perturbed[:, 1])
    with pytest.raises(ValueError):
        sample_noise(a, bounds, [-1.0, 0.0])


def test_native_noisy_limits_match_independent_models_on_two_threads(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("hinge_limit_probe.py")),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            PYTHONPATH=str(root),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
