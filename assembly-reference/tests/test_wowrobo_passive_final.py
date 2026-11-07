"""Native fixture structure and mechanical stop regression."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_passive_final import FROZEN, build_fixture


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    if not (FROZEN / "scene.xml").is_file():
        pytest.skip("Local repaired vendor geometry is required")
    info = build_fixture(tmp_path_factory.mktemp("passive-final"))
    return info, mujoco.MjModel.from_xml_path(info["scene"])


def test_only_module_root_and_passive_motor_hinge_remain_free(fixture):
    info, m = fixture
    assert (m.nq, m.nv, m.nu) == (8, 7, 0)
    assert [m.joint(i).name for i in range(m.njnt)] == [
        "forearm_module_free",
        "target_servo_3_shaft",
    ]
    assert info["receiver_fixed"] and not info["hand_present"]
    np.testing.assert_array_equal(m.opt.gravity, [0, 0, -9.81])
    assert ET.parse(info["scene"]).find(".//equality") is None


def test_own_motor_stop_is_contact_geometry_not_a_prescribed_angle(fixture):
    info, m = fixture
    d = mujoco.MjData(m)
    qa = int(m.joint("target_servo_3_shaft").qposadr[0])
    counts = []
    for angle in [0, 8, 10]:
        d.qpos[qa] = np.deg2rad(angle)
        mujoco.mj_forward(m, d)
        counts.append(
            sum(
                "servo_3_case__contact" in [m.geom(int(g)).name for g in c.geom]
                and any(
                    m.geom(int(g)).name.startswith("forearm_module_printed")
                    for g in c.geom
                )
                for c in d.contact
            )
        )
    assert counts[:2] == [0, 0] and counts[2] > 0
    assert not info["hinge_actuation"] and not info["prescribed_hinge_angle"]
    assert float(m.joint("target_servo_3_shaft").stiffness[0]) == 0.0
    assert float(m.joint("target_servo_3_shaft").range[1]) > np.deg2rad(60)


def test_own_stop_pair_keeps_isotropic_source_friction(fixture):
    _info, m = fixture
    for name in ("own_motor_stop_0", "own_motor_stop_1"):
        pair = m.pair(name)
        source = m.geom("servo_3_case__contact").friction
        np.testing.assert_allclose(
            pair.friction, [source[0], source[0], source[1], source[2], source[2]]
        )
