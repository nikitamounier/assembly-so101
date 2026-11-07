"""Displayed tolerance boxes and binary seating reward use the same bounds."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_reward import RewardRegions


@pytest.fixture
def reward():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="receiver"><freejoint/><geom type="box" size=".01 .01 .01"/></body><body name="part"><freejoint/><geom name="part_geom" type="box" size=".02 .01 .01"/></body></worldbody></mujoco>'
    )
    config = dict(
        model_signature="test",
        position_tolerance_m=0.0005,
        orientation_tolerance_degrees=1.0,
        definition="Complete seating within position and angle tolerance",
        tasks=dict(
            part=dict(
                receiver="receiver",
                geom="part_geom",
                mesh_center_local=[0, 0, 0],
                center_receiver_local=[0, 0, 0],
                rotation_receiver_local=np.eye(3).tolist(),
                size_m=[0.04, 0.02, 0.02],
            )
        ),
    )
    return m, mujoco.MjData(m), RewardRegions(m, "test", config)


def read(m, d, regions):
    mujoco.mj_kinematics(m, d)
    return regions.snapshot(d, dict(task="part"))


def test_position_and_rotation_limits_match_displayed_box(reward):
    m, d, r = reward
    o = read(m, d, r)
    assert o["reward"] == 1 and o["noise_box"]["size"] == [0.001] * 3
    d.qpos[m.joint(1).qposadr[0]] = 0.000499
    assert read(m, d, r)["reward"] == 1
    d.qpos[m.joint(1).qposadr[0]] = 0.000501
    assert read(m, d, r)["reward"] == 0
    d.qpos[m.joint(1).qposadr[0]] = 0
    adr = m.joint(1).qposadr[0] + 3
    for degrees, expected in [(0.9, 1), (1.1, 0)]:
        angle = np.radians(degrees) / 2
        d.qpos[adr : adr + 4] = [np.cos(angle), 0, 0, np.sin(angle)]
        assert read(m, d, r)["reward"] == expected


def test_region_follows_receiver_and_tolerance_updates(reward):
    m, d, r = reward
    for joint in range(2):
        adr = m.joint(joint).qposadr[0]
        d.qpos[adr : adr + 3] = [0.2, 0.3, 0.4]
        d.qpos[adr + 3 : adr + 7] = [np.cos(0.4), 0, 0, np.sin(0.4)]
    o = read(m, d, r)
    np.testing.assert_allclose(o["noise_box"]["center"], [0.2, 0.3, 0.4])
    assert o["reward"] == 1
    r.set_tolerances(1.2, 2)
    assert read(m, d, r)["noise_box"]["size"] == [0.0024] * 3
    assert r.reward(d, "part") == 1
    with pytest.raises(ValueError):
        r.set_tolerances(float("nan"), 1)
    with pytest.raises(ValueError):
        r.set_tolerances(0.5, 11)
