"""Reported overlapping states must remain visibly invalid."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from tests.native import ROOT, fixture_state, load_scene


@pytest.mark.parametrize("name", ["motor_seated_pose.json", "motor_side_pose.json"])
def test_overlapping_saved_states_are_invalid(name):
    m, _ = load_scene()
    saved = json.loads(Path(__file__).with_name(name).read_text())
    q, _ = fixture_state(m, saved)
    assert sampled_motor_base_overlap(m, q, ROOT / "assets_local") > 0.0005


def test_free_motor_moves_without_sleep_or_welds():
    m, desc = load_scene()
    d = mujoco.MjData(m)
    d.qpos[:] = desc["home_qpos"]
    d.ctrl[:] = desc["home_ctrl"]
    mujoco.mj_step(m, d, nstep=4000)
    before = d.body("motor").xpos.copy()
    d.xfrc_applied[m.body("motor").id, 2] = 0.8
    mujoco.mj_step(m, d, nstep=200)
    assert np.linalg.norm(d.body("motor").xpos - before) > 0.005
    assert not m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    assert m.neq == 0 and d.body_awake[m.body("motor").id] == 1 and np.isfinite(d.qpos).all()
