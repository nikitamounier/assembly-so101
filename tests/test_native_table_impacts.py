"""Fast-corner table impacts at several sub-step phases, using native physics."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101 import scene
from assembly_policy.so101.contact_audit import table_overlaps
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.task_scenes import task_scene


@pytest.fixture(scope="module")
def preimpact():
    model = scene.model(contact_monitor=True)
    prepared = task_scene(model, "yoke")
    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "ctrl"):
        getattr(data, name)[:] = prepared[name]
    q = int(model.joint("yoke_free").qposadr[0])
    v = int(model.joint("yoke_free").dofadr[0])
    data.qpos[q : q + 3] += [0.002, 0, 0.06]
    delta = np.zeros(4)
    mujoco.mju_axisAngle2Quat(delta, np.array([1.0, 0, 0]), np.deg2rad(2.0))
    quat = data.qpos[q + 3 : q + 7].copy()
    mujoco.mju_mulQuat(data.qpos[q + 3 : q + 7], delta, quat)
    data.qvel[v : v + 6] = 0
    mujoco.mj_forward(model, data)
    visual = model.geom("yoke_visual").id
    mesh = model.geom_dataid[visual]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count].astype(float)
    audit = mujoco.MjData(model)

    def height(d):
        # The rendered mesh provides an independent coordinate path from the
        # original-CAD table guard used by training and the review recorder.
        geom = d.geom("yoke_visual")
        plane = d.geom("table")
        normal = plane.xmat.reshape(3, 3)[:, 2]
        return float(
            (vertices @ (geom.xmat.reshape(3, 3).T @ normal) + (geom.xpos - plane.xpos) @ normal).min()
        )

    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        audit.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, audit)
        if 0.003 < height(audit) < 0.006:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, data, state, spec)
            return model, state, q, height
    pytest.fail("The released yoke did not reach the pre-impact window")


@pytest.mark.parametrize("height_offset", [0.0, 0.00025, 0.0005])
def test_fast_yoke_corner_respects_table_at_different_impact_phases(preimpact, height_offset):
    model, state, q, height = preimpact
    data = mujoco.MjData(model)
    audit = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    # Each trial starts from the same physically reached state with one small
    # initialization perturbation. No pose or velocity overrides are applied during motion.
    data.qpos[q + 2] += height_offset
    mujoco.mj_forward(model, data)
    assert height(data) > 0.002
    ids = {model.geom("table").id, model.geom("yoke_contact").id}
    native = visual = original = 0.0
    for _ in range(round(0.15 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        native = max(native, max([0.0, *[-float(c.dist) for c in data.contact if set(c.geom) == ids]]))
        audit.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, audit)
        visual = max(visual, -height(audit))
        original = max(original, table_overlaps(audit, ["yoke"], scene.ASSETS)["yoke/table"])
    assert native < PENETRATION_LIMIT_M and visual < PENETRATION_LIMIT_M and original < PENETRATION_LIMIT_M, (
        native,
        visual,
        original,
    )
    assert abs(visual - original) < 1e-7
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and not data.warning.number.any()
    assert model.neq == 0 and not model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
