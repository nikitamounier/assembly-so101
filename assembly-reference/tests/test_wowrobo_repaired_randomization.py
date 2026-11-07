"""Repaired reset sampling: model routing and real support articulation."""

from pathlib import Path

import numpy as np
import pytest

from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/wowrobo_randomization_repaired.json"
BANK = ROOT / "assets_local/wowrobo/photo-resets-repaired-v1"


@pytest.fixture(scope="module")
def sampler():
    if not (BANK / "scene.xml").exists():
        pytest.skip("Build repaired local assets/reference bank first")
    return WowRoboRandomizer(config=CONFIG)


def test_repaired_configuration_routes_matching_assets_and_bank(sampler):
    assert sampler.assets.name == "components-repaired-v1"
    assert sampler.nominal_bank == BANK
    for stage in sampler.definitions:
        nominal = sampler.restore(stage)
        sampled, receipt = sampler.sample(stage, 17, difficulty=0, solve_arms=False)
        np.testing.assert_allclose(sampled.qpos, nominal.qpos, atol=1e-10, rtol=0)
        assert receipt["cad_revision"] == "final-common-v1"
        assert not receipt.get("candidate_rejected", False)


def test_mismatched_old_geometry_is_rejected(sampler):
    with pytest.raises(ValueError, match="CAD assets do not match"):
        WowRoboRandomizer(
            assets=ROOT / "assets_local/wowrobo/components-v2", config=CONFIG
        )


@pytest.mark.parametrize(
    "stage", ["22_upper_half", "23_upper_near", "30_final_half", "31_final_near"]
)
def test_supported_sample_preserves_registered_case_orientation(sampler, stage):
    nominal = sampler.restore(stage)
    d, receipt = sampler.sample(stage, 0, difficulty=0.5, solve_arms=False)
    fit = receipt["resting_contact"]["case_support_fit"]
    assert fit["case_position_error_m"] < 1e-10
    assert fit["case_rotation_matrix_error"] < 1e-10
    assert receipt["resting_contact"]["projection_distance_m"] == 0
    assert receipt["resting_contact"]["method"].startswith("registered case path")
    servo = 2 if stage.startswith(("22", "23")) else 3
    case = sampler.model.body(f"target_servo_{servo}_pivot").id
    swivel = next(t for t in receipt["transforms"] if t["kind"] == "connector_swivel")
    expected = np.asarray(swivel["rotation"]) @ nominal.xmat[case].reshape(3, 3)
    np.testing.assert_allclose(d.xmat[case].reshape(3, 3), expected, atol=1e-10, rtol=0)
    axis = np.asarray(receipt["insertion_frame"]["axis_receiver_local"])
    assert np.linalg.norm(axis) == pytest.approx(1, abs=1e-10)
