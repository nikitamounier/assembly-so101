"""The corrected supporting hand grasps the installed cover, continuously."""

from pathlib import Path

import numpy as np
import pytest

from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "assets_local/wowrobo/photo-resets-repaired-v1"
NEW = ROOT / "assets_local/wowrobo/photo-resets-repaired-cover-support-v2"
AFFECTED = [
    "17_left_support",
    "18_upper_floor_grasp",
    "20_upper_transport",
    "21_upper_far_side",
    "22_upper_half",
    "23_upper_near",
    "25_right_support",
]


@pytest.fixture(scope="module")
def samplers():
    if not (NEW / "scene.xml").exists():
        pytest.skip("Build reviewed cover-support bank")
    return (
        WowRoboRandomizer(config=ROOT / "configs/wowrobo_randomization_repaired.json"),
        WowRoboRandomizer(
            config=ROOT / "configs/wowrobo_randomization_repaired_cover_support.json"
        ),
    )


def test_only_left_reference_coordinates_change_and_hold_continues(samplers):
    old, new = samplers
    m = new.model
    left = [
        m.joint("left_" + n).qposadr[0]
        for n in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    ]
    mask = np.ones(m.nq, bool)
    mask[left] = False
    first = None
    for stage in AFFECTED:
        before, after = old.restore(stage), new.restore(stage)
        np.testing.assert_array_equal(before.qpos[mask], after.qpos[mask])
        if first is None:
            first = after.qpos[left].copy()
        np.testing.assert_array_equal(after.qpos[left], first)
        assert (
            new.nominal_receipts[stage]["support_grasp_by_side"]["left"]["piece"]
            == "base_side_cover"
        )
    for stage in set(new.definitions) - set(AFFECTED):
        np.testing.assert_array_equal(old.restore(stage).qpos, new.restore(stage).qpos)


@pytest.mark.parametrize("stage", ["18_upper_floor_grasp", "21_upper_far_side"])
def test_active_grasp_noise_does_not_jitter_the_cover_support(samplers, stage):
    old, new = samplers
    m = new.model
    left = [
        m.joint("left_" + n).qposadr[0]
        for n in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    ]
    nominal = new.restore(stage)
    before, _unused = old.sample(stage, 0, difficulty=0.5)
    after, receipt = new.sample(stage, 0, difficulty=0.5)
    # Native IK keeps a 1e-7 rad safety margin from the wrist joint limit.
    np.testing.assert_allclose(after.qpos[left], nominal.qpos[left], atol=2e-7, rtol=0)
    mask = np.ones(m.nq, bool)
    mask[left] = False
    np.testing.assert_allclose(after.qpos[mask], before.qpos[mask], atol=1e-10, rtol=0)
    assert receipt["support_grasp_by_side"]["left"]["stable_support"] is True
