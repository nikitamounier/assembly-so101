"""Effective friction, constraint rebuilding, replay and per-data isolation."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_friction import (
    apply_contact_friction,
    apply_friction_and_rebuild_constraints,
    infer_geom_roles,
    sample_friction_multipliers,
)


def model():
    return mujoco.MjModel.from_xml_string("""<mujoco>
      <option integrator="Euler" cone="pyramidal" jacobian="dense"/>
      <worldbody><geom name="review_table" type="plane" size="1 1 .1" friction="1.2 .005 .0001"/>
        <body name="base_servo" pos="0 0 .009"><freejoint/>
          <geom name="part" type="sphere" size=".01" mass="1" friction=".8 .003 .0002"/></body>
        <body name="right_gripper" pos=".019 0 .009">
          <geom name="pad" type="sphere" size=".01" friction=".9 .006 .0003"/></body>
      </worldbody></mujoco>""")


def test_seeded_draws_small_positive_and_complete():
    a = sample_friction_multipliers(np.random.default_rng(100))
    assert a == sample_friction_multipliers(np.random.default_rng(100))
    assert a != sample_friction_multipliers(np.random.default_rng(101))
    assert set(a) == {"finger_part", "part_table", "part_part"}
    assert all(0.97 <= v <= 1.03 for v in a.values())
    with pytest.raises(ValueError):
        sample_friction_multipliers(np.random.default_rng(), 1)


def test_effective_contact_coefficients_and_shared_model_isolation():
    m = model()
    a, b = mujoco.MjData(m), mujoco.MjData(m)
    mujoco.mj_step1(m, a)
    mujoco.mj_step1(m, b)
    old_model, old_pair = m.geom_friction.copy(), m.pair_friction.copy()
    other = b.contact.friction.copy()
    baseline = apply_contact_friction(
        m, a, {"finger_part": 1.03, "part_table": 0.97, "part_part": 1}
    )
    roles = infer_geom_roles(m)
    seen = set()
    for i, pair in enumerate(a.contact.geom):
        pair_roles = {roles.get(int(g)) for g in pair}
        factor = (
            1.03
            if pair_roles == {"finger", "part"}
            else 0.97
            if pair_roles == {"part", "table"}
            else 1
        )
        if factor != 1:
            seen.add(factor)
        np.testing.assert_allclose(a.contact.friction[i, :2], baseline[i, :2] * factor)
        np.testing.assert_array_equal(a.contact.friction[i, 2:], baseline[i, 2:])
    assert seen == {1.03, 0.97}
    np.testing.assert_array_equal(b.contact.friction, other)
    np.testing.assert_array_equal(m.geom_friction, old_model)
    np.testing.assert_array_equal(m.pair_friction, old_pair)
    first = a.contact.friction.copy()
    apply_contact_friction(
        m,
        a,
        {"finger_part": 1.03, "part_table": 0.97, "part_part": 1},
        baseline_friction=baseline,
    )
    np.testing.assert_array_equal(a.contact.friction, first)


def test_rebuild_changes_actual_friction_constraint_jacobian():
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_step1(m, d)
    before = d.efc_J.copy()
    multipliers = {"finger_part": 1.03, "part_table": 0.97, "part_part": 1}
    baseline = apply_contact_friction(m, d, multipliers)
    np.testing.assert_array_equal(d.efc_J, before)  # Contact edits alone do not work.
    apply_friction_and_rebuild_constraints(
        m, d, multipliers, baseline_friction=baseline
    )
    assert np.max(np.abs(d.efc_J - before)) > 1e-3
    mujoco.mj_step2(m, d)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qacc).all()
    assert d.time == pytest.approx(m.opt.timestep)


def test_pair_coefficients_and_non_euler_guard():
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_step1(m, d)
    original = d.contact.friction.copy()
    apply_contact_friction(m, d, {"finger_part": 1, "part_table": 1, "part_part": 1})
    np.testing.assert_array_equal(d.contact.friction, original)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError, match="Euler"):
        apply_friction_and_rebuild_constraints(
            m, d, {"finger_part": 1, "part_table": 1, "part_part": 1}
        )
    with pytest.raises(ValueError):
        apply_contact_friction(m, d, {"part_table": 1})


def test_explicit_pair_baseline_is_scaled_after_mixing():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="review_table" type="plane" size="1 1 .1" friction="1.2 .005 .0001"/>
      <body name="base_servo" pos="0 0 .009"><freejoint/>
        <geom name="part" type="sphere" size=".01" mass="1" friction=".8 .003 .0002"/>
      </body></worldbody><contact>
        <pair geom1="review_table" geom2="part" condim="4" friction=".4 .5 .002 .0002 .0003"/>
      </contact></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_step1(m, d)
    assert d.ncon == 1
    baseline = d.contact.friction.copy()
    np.testing.assert_allclose(baseline[0, :2], [0.4, 0.5])
    apply_contact_friction(m, d, {"finger_part": 1, "part_table": 0.97, "part_part": 1})
    np.testing.assert_allclose(d.contact.friction[0, :2], [0.388, 0.485])
    np.testing.assert_array_equal(d.contact.friction[0, 2:], baseline[0, 2:])
    np.testing.assert_allclose(m.pair_friction[0, :2], [0.4, 0.5])


def test_part_part_category_and_unknown_material_unchanged():
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_step1(m, d)
    roles = {m.geom("part").id: "part", m.geom("pad").id: "part"}
    before = d.contact.friction.copy()
    apply_contact_friction(
        m, d, {"finger_part": 1, "part_table": 1, "part_part": 1.02}, geom_roles=roles
    )
    changed = 0
    for i, pair in enumerate(d.contact.geom):
        factor = 1.02 if all(roles.get(int(g)) == "part" for g in pair) else 1
        np.testing.assert_allclose(d.contact.friction[i, :2], before[i, :2] * factor)
        changed += factor != 1
    assert changed == 1
