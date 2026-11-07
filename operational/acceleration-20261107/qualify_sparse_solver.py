"""Run the physical acceptance panel with only Jacobian storage changed."""

import functools

import pytest

from assembly_policy.so101 import scene

original = scene.model


@functools.wraps(original)
def sparse_model(*args, **kwargs):
    m = original(*args, **kwargs)
    m.opt.jacobian = 1
    assert int(m.opt.jacobian) == 1
    return m


scene.model = sparse_model
raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/test_native_motor_walls.py",
            "tests/test_native_contact_stability.py",
            "tests/test_native_table_impacts.py",
            "tests/test_native_saved_states.py",
            "tests/test_native_gripper_closure.py",
            "tests/test_so101_contact_coverage.py",
            "tests/test_so101_rewards.py",
        ]
    )
)
