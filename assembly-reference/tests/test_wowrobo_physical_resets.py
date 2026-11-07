"""Physical admission mechanisms; never a certificate for all generated states."""

import json

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_physical_resets import (
    ROOT,
    WowRoboPhysicalValidator,
    physical_split_step,
)


def test_implicit_friction_split_matches_native_and_rebuilds_force_cone():
    from assembly_policy.so101.wowrobo_friction import CATEGORIES

    m = mujoco.MjModel.from_xml_string("""<mujoco>
    <option integrator="implicitfast" timestep=".00025" cone="elliptic"/>
    <worldbody><geom name="table" type="plane" size="1 1 .1"/>
    <body name="base_servo" pos="0 0 .0099"><freejoint/>
    <geom type="box" size=".01 .01 .01" mass=".1" friction=".5 .005 .0001"/>
    </body></worldbody></mujoco>""")
    native, split, noisy = (mujoco.MjData(m) for _ in range(3))
    for d in (native, split, noisy):
        d.qvel[0] = 0.2
    unity = dict.fromkeys(CATEGORIES, 1.0)
    for _ in range(100):
        mujoco.mj_step(m, native)
        physical_split_step(m, split, unity)
    np.testing.assert_allclose(split.qpos, native.qpos, atol=1e-12)
    np.testing.assert_allclose(split.qvel, native.qvel, atol=1e-12)
    physical_split_step(m, noisy, dict.fromkeys(CATEGORIES, 0.5))
    assert noisy.ncon > 0
    np.testing.assert_allclose(noisy.contact.friction[: noisy.ncon, 0], 0.5)
    assert np.isfinite(noisy.efc_R[: noisy.nefc]).all()
    unity_first = mujoco.MjData(m)
    unity_first.qvel[0] = 0.2
    physical_split_step(m, unity_first, unity)
    np.testing.assert_allclose(unity_first.contact.friction[: unity_first.ncon, 0], 1.0)
    assert not np.allclose(noisy.qvel, unity_first.qvel)


def pad_model():
    return mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 -9.81" timestep=".001"/>
    <worldbody><body name="right_gripper"><site name="right_gripperframe"/>
    <geom name="palm__contact" type="plane" size="1 1 .01"/></body>
    <body name="right_moving_jaw_so101_v1" pos="2 0 0"/>
    <body name="base_servo" pos="0 0 .009"><freejoint/><geom type="box" size=".008 .008 .01" mass=".1"/></body>
    </worldbody></mujoco>""")


def test_positive_palm_contact_cannot_count_as_fixed_finger_load():
    m = pad_model()
    d = mujoco.MjData(m)
    for _ in range(10):
        mujoco.mj_step(m, d)
    assert d.ncon > 0
    v = object.__new__(WowRoboPhysicalValidator)
    v.model = m
    v.grid_records = {"palm": {"source_mesh_name": "right_sts3215_03a_v1"}}
    assert v._loaded_pads(d, [("right", "base_servo")])["right"]["fixed"] == 0
    v.grid_records = {
        "palm": {"source_mesh_name": "right_wrist_roll_follower_so101_v1"}
    }
    assert v._loaded_pads(d, [("right", "base_servo")])["right"]["fixed"] > 0


def test_nonterminal_uses_actual_position_and_orientation_not_stage_label():
    m = pad_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    v = object.__new__(WowRoboPhysicalValidator)
    v.model = m
    v.definitions = {"05_cover_near": {"task": "base_servo"}}
    b = m.body("base_servo").id
    v.goal_anchors = {
        "base_servo": {
            "body": b,
            "feature": np.zeros(3),
            "position": d.xpos[b].copy() + [0, 0, 0.002],
            "rotation": np.eye(3),
            "source": "test known feature",
            "goal_geometry_certified": True,
        }
    }
    assert not v._completion("05_cover_near", d, {})["is_terminal"]
    v.goal_anchors["base_servo"]["position"] = d.xpos[b].copy() + [0, 0, 0.0001]
    assert v._completion("05_cover_near", d, {})["is_terminal"]


@pytest.fixture(scope="module")
def physical():
    folder = ROOT / "assets_local/wowrobo/physical-resets-v6"
    if not (folder / "profile.json").exists():
        pytest.skip("Prepare source-SDF physical profile first")
    return WowRoboPhysicalValidator(
        output_dir=folder,
        pin_mesh=ROOT / "assets_local/wowrobo/fit-pin-v2/pin.obj",
        settle_seconds=0.001,
        hold_seconds=0.001,
    )


def test_native_profile_has_real_gravity_contacts_and_no_added_fixtures(physical):
    m = physical.model
    original = physical.nominal_model
    np.testing.assert_array_equal(m.opt.gravity, [0, 0, -9.81])
    assert m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.nplugin >= 54 and m.nq == original.nq and m.nv == original.nv
    assert m.neq == original.neq + 3
    np.testing.assert_allclose(m.body_mass, original.body_mass, atol=1e-10)
    assert np.count_nonzero(m.geom_contype) > 50
    assert m.actuator_forcelimited.all()
    assert np.all(m.actuator_forcerange[:, 1] <= 3.35 + 1e-10)


def test_grasp_roles_require_case_not_surrounding_print(physical):
    assert ("right", "target_servo_2_pivot") in physical._grip_pairs(
        "20_upper_transport"
    )
    assert ("left", "target_servo_3_pivot") in physical._grip_pairs(
        "28_final_transport"
    )
    assert ("right", "upper_arm_module") in physical._grip_pairs("28_final_transport")


def test_installed_interfaces_preserve_output_hinge_and_integration_state(physical):
    v = physical
    m, original = v.model, v.nominal_model
    assert m.neq == original.neq + 3
    assert m.nq == original.nq and m.nv == original.nv
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for i in range(m.neq):
        assert not m.eq_active0[i]
        names = [m.body(int(b)).name for b in (m.eq_obj1id[i], m.eq_obj2id[i])]
        assert not any("left" in name or "right" in name for name in names)
    eq = m.equality("installed_connector_to_output_horns").id
    assert m.body(int(m.eq_obj1id[eq])).name == "target_servo_1_pivot"
    assert m.joint("target_servo_1_shaft").type == mujoco.mjtJoint.mjJNT_HINGE
    d, replay = mujoco.MjData(m), mujoco.MjData(m)
    d.eq_active[eq] = True
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    mujoco.mj_setState(m, replay, state, spec)
    np.testing.assert_array_equal(d.eq_active, replay.eq_active)


def test_target_grid_binding_rejects_wrong_geometry_scale_and_grid(tmp_path):
    from assembly_policy.so101.wowrobo_physical_resets import (
        _hash,
        _target_grid_matches_asset,
    )

    mesh, other, grid = (
        tmp_path / name for name in ("source.obj", "other.obj", "field.bin")
    )
    mesh.write_bytes(b"original source")
    other.write_bytes(b"different geometry")
    grid.write_bytes(b"field")
    record = {
        "source_file": str(mesh),
        "source_sha256": _hash(mesh),
        "grid_file": str(grid),
        "grid_sha256": _hash(grid),
    }
    _target_grid_matches_asset(record, mesh, np.ones(3))
    with pytest.raises(ValueError, match="visual source"):
        _target_grid_matches_asset(record, other, np.ones(3))
    with pytest.raises(ValueError, match="unscaled"):
        _target_grid_matches_asset(record, mesh, np.array([0.001] * 3))
    grid.write_bytes(b"changed field")
    with pytest.raises(ValueError, match="grid file identity"):
        _target_grid_matches_asset(record, mesh, np.ones(3))


def test_new_completed_anchor_selection_never_falls_back_when_explicit(tmp_path):
    from assembly_policy.so101.wowrobo_physical_resets import _anchor_folder

    nominal = tmp_path / "new-nominal"
    (nominal / "completed").mkdir(parents=True)
    assert _anchor_folder(nominal) == (nominal / "completed").resolve()
    chosen = tmp_path / "chosen-completed"
    assert _anchor_folder(nominal, chosen) == chosen.resolve()


def test_full_mating_profile_is_the_unique_union_of_original_per_step_laws():
    import xml.etree.ElementTree as ET

    from assembly_policy.so101.wowrobo_mating_profile import CONFIG
    from assembly_policy.so101.wowrobo_physical_resets import _apply_all_mating_pairs

    config = json.loads(CONFIG.read_text())
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    names = {
        row[k]
        for rows in config["steps"].values()
        for row in rows
        for k in ("geom1", "geom2")
    }
    for name in names:
        ET.SubElement(world, "geom", name=name)
    receipt = _apply_all_mating_pairs(root)
    assert len(receipt["pairs"]) == len(root.findall("contact/pair")) == 14
    for pair, record in zip(root.findall("contact/pair"), receipt["pairs"]):
        for key, value in record["law"].items():
            if isinstance(value, list):
                np.testing.assert_array_equal(
                    np.fromstring(pair.get(key), sep=" "), value
                )
            else:
                assert pair.get(key) == str(value)
    assert not root.findall("equality/*")


def test_registered_pin_input_is_not_repositioned_or_regripped():
    validator = WowRoboPhysicalValidator.__new__(WowRoboPhysicalValidator)
    validator.pin_registration = {"head_gap_m": 0.0001}
    validator.profile = {
        "pin_pose_registered": True,
        "pin_registration_path": "verified-registration.json",
        "pin_registration_sha256": "bound-sha",
    }
    data = type("UnchangedData", (), {"qpos": np.arange(12, dtype=float)})()
    before = data.qpos.copy()
    receipt = validator._register_pin("14_pin_half", data, {})
    np.testing.assert_array_equal(before, data.qpos)
    assert not receipt["pose_overlay_applied"]
    assert receipt["noise_preserved"]
    assert receipt["left_gripper_ik"] is None


def test_installed_attachments_use_new_completed_states(physical, tmp_path):
    import shutil
    import xml.etree.ElementTree as ET

    from assembly_policy.so101.wowrobo_physical_resets import (
        ASSEMBLY_ATTACHMENTS,
        add_assembly_attachments,
    )

    nominal = tmp_path / "new-bank"
    completed = nominal / "completed"
    completed.mkdir(parents=True)
    shutil.copyfile(physical.nominal_bank / "scene.xml", nominal / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str(nominal / "scene.xml"))
    d = mujoco.MjData(m)
    d.qpos[m.joint("base_side_cover_free").qposadr[0]] = 0.031234
    d.qpos[m.joint("shoulder_connector_free").qposadr[0] + 1] = 0.04321
    d.qpos[m.joint("upper_arm_module_free").qposadr[0] + 2] = 0.212345
    d.qpos[m.joint("target_servo_2_shaft").qposadr[0]] = 0.1234
    mujoco.mj_forward(m, d)
    state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for _, _, _, anchor, _ in ASSEMBLY_ATTACHMENTS:
        np.savez_compressed(completed / (anchor + "_train.npz"), states=[state])
    root = ET.parse(nominal / "scene.xml").getroot()
    add_assembly_attachments(root, nominal, completed)
    for name, parent, child, _, _ in ASSEMBLY_ATTACHMENTS:
        weld = root.find(f"equality/weld[@name='{name}']")
        actual = np.fromstring(weld.get("relpose"), sep=" ")
        a, b = m.body(parent).id, m.body(child).id
        expected = d.xmat[a].reshape(3, 3).T @ (d.xpos[b] - d.xpos[a])
        np.testing.assert_allclose(actual[:3], expected, atol=1e-12)


def test_support_contact_target_can_override_lower_base_housing():
    validator = object.__new__(WowRoboPhysicalValidator)
    validator.definitions = {
        "18_upper_floor_grasp": {
            "task": "upper_arm_module",
            "left_role": "hold_base_assembly",
            "right_role": "grasp_active",
        }
    }
    validator.support_grasp_by_stage = {
        "18_upper_floor_grasp": {"left": {"piece": "base_side_cover"}}
    }
    assert validator._grip_pairs("18_upper_floor_grasp") == [
        ("left", "base_side_cover"),
        ("right", "target_servo_2_pivot"),
    ]


def test_passive_motor_stops_bypass_parent_filter_without_locking_hinge():
    import xml.etree.ElementTree as ET

    from assembly_policy.so101.wowrobo_physical_resets import (
        add_passive_bracket_contacts,
    )

    root = ET.fromstring("""<mujoco><option gravity="0 0 0"/>
    <worldbody><body name="upper_arm_module"><freejoint/><geom name="upper_arm_module_printed_0__contact" type="box" size=".05 .05 .05"/>
    <body><joint name="test_hinge" type="hinge"/>
    <geom name="servo_2_case__contact" type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body></body>
    <body pos="1 0 0"><freejoint/><geom name="forearm_module_printed_0__contact" type="box" size=".05 .05 .05"/>
    <geom name="forearm_module_printed_1__contact" type="box" pos="0 1 0" size=".05 .05 .05"/>
    <body><joint type="hinge"/><geom name="servo_3_case__contact" type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body>
    </body></worldbody></mujoco>""")
    no_stop = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    no_data = mujoco.MjData(no_stop)
    mujoco.mj_forward(no_stop, no_data)
    assert no_data.ncon == 0  # Native parent filtering omits these contacts.
    receipt = add_passive_bracket_contacts(root)
    with_stop = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(with_stop)
    mujoco.mj_forward(with_stop, data)
    assert data.ncon == 2
    assert with_stop.nq == no_stop.nq and with_stop.neq == no_stop.neq == 0
    assert with_stop.nu == 0
    assert not with_stop.jnt_stiffness.any()
    assert len(receipt["pairs"]) == 3
    assert not receipt["hinge_angle_target"]
    with pytest.raises(ValueError, match="already has"):
        add_passive_bracket_contacts(root)


@pytest.mark.parametrize("locking", ["actuator", "spring", "equality", "weld"])
def test_passive_target_shafts_reject_prescribed_angles(locking):
    from assembly_policy.so101.wowrobo_physical_resets import (
        check_passive_target_shafts,
    )

    body = "".join(
        f'<body name="parent{i}" pos="{i} 0 0"><body name="pivot{i}">'
        f'<joint name="target_servo_{i}_shaft" type="hinge" stiffness="{int(locking == "spring" and i == 2)}"/>'
        '<geom type="sphere" size=".01" mass=".1"/></body></body>'
        for i in range(1, 5)
    )
    extra = {
        "actuator": '<actuator><position joint="target_servo_2_shaft" kp="10"/></actuator>',
        "spring": "",
        "equality": '<equality><joint joint1="target_servo_2_shaft"/></equality>',
        "weld": '<equality><weld body1="parent2" body2="pivot2"/></equality>',
    }[locking]
    m = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{body}</worldbody>{extra}</mujoco>"
    )
    with pytest.raises(ValueError, match="shaft"):
        check_passive_target_shafts(m)


def test_internal_bearing_exclusions_preserve_explicit_bracket_stops():
    import xml.etree.ElementTree as ET

    from assembly_policy.so101.wowrobo_physical_resets import (
        add_passive_bracket_contacts,
        exclude_internal_servo_bearings,
    )

    root = ET.fromstring("""<mujoco><option gravity="0 0 0"/>
    <worldbody>
    <body name="left_base" pos="-1 0 0"><geom type="box" size=".05 .05 .05"/>
      <body name="left_shoulder"><joint type="hinge"/><geom type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body></body>
    <body name="right_base" pos="-2 0 0"><geom type="box" size=".05 .05 .05"/>
      <body name="right_shoulder"><joint type="hinge"/><geom type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body></body>
    <body name="base_servo" pos="2 0 0"><geom name="servo_1_case__contact" type="box" size=".05 .05 .05"/>
      <body name="target_servo_1_pivot"><joint type="hinge"/><geom name="servo_1_horns__contact" type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body></body>
    <body name="upper_arm_module"><freejoint/><geom name="upper_arm_module_printed_0__contact" type="box" size=".05 .05 .05"/>
      <body name="target_servo_2_pivot"><joint type="hinge"/><geom name="servo_2_case__contact" type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body></body>
    <body name="forearm_module" pos="1 0 0"><freejoint/><geom name="forearm_module_printed_0__contact" type="box" size=".05 .05 .05"/>
      <geom name="forearm_module_printed_1__contact" type="box" pos="0 1 0" size=".05 .05 .05"/>
      <body name="target_servo_3_pivot"><joint type="hinge"/><geom name="servo_3_case__contact" type="sphere" size=".02" pos=".06 0 0" mass=".1"/></body>
      <body name="target_servo_4_pivot" pos="0 2 0"><joint type="hinge"/><geom type="sphere" size=".02" mass=".1"/></body>
    </body></worldbody></mujoco>""")
    add_passive_bracket_contacts(root)
    before = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(before)
    mujoco.mj_forward(before, data)
    assert data.ncon == 5  # Includes target and both helper base bearings.
    receipt = exclude_internal_servo_bearings(root)
    after = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(after)
    mujoco.mj_forward(after, data)
    assert data.ncon == 2  # Both explicit bracket stops survive the exclusions.
    assert after.nq == before.nq and after.nu == before.nu == 0
    assert after.neq == before.neq == 0
    assert receipt["explicit_bracket_stops_preserved"]
    assert len(receipt["exclusions"]) == 6
    assert all(
        "servo_1_horns" not in after.geom(int(g)).name
        for c in data.contact
        for g in c.geom
    )
