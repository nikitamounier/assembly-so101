"""The actual fingers stop by native contact and reopen, without CAD overlap."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.contact_audit import finger_surface_clearance
from assembly_policy.so101.workcell import configuration
from tests.native import load_scene


@pytest.fixture(scope="module")
def plant():
    return load_scene()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("initial", [0.65, 0.1])
def test_empty_fingers_touch_by_contact_before_joint_limit_and_reopen(plant, side, initial):
    m, description = plant
    d = mujoco.MjData(m)
    d.qpos[:] = description["home_qpos"]
    d.ctrl[:] = description["home_ctrl"]
    q = int(m.joint(side + "_gripper").qposadr[0])
    a = m.actuator(side + "_gripper").id
    d.qpos[q] = initial
    d.ctrl[a] = configuration()["gripper"]["closed_command_rad"]
    mujoco.mj_step(m, d, nstep=2400)
    clearance = finger_surface_clearance(m, d, side)
    assert clearance["sampled_surface_gap_m"] < 0.00005, clearance
    assert clearance["max_sampled_material_overlap_m"] < 0.0002, clearance
    assert d.qpos[q] > m.joint(side + "_gripper").range[0] + 0.02
    forces = []
    for i, c in enumerate(d.contact):
        if any(m.geom(int(g)).name == side + "_distal_finger_contact" for g in c.geom):
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            forces.append(f[0])
    assert sum(forces) > 1.0
    assert not d.warning.number.any() and m.neq == 0
    d.ctrl[a] = configuration()["gripper"]["open_command_rad"]
    mujoco.mj_step(m, d, nstep=2400)
    assert finger_surface_clearance(m, d, side)["sampled_surface_gap_m"] > 0.01
