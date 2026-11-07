"""Force-driven wall approaches, independent CAD-overlap checks, no pose actuation."""

import json
import struct
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from tests.native import ROOT, load_scene


@pytest.fixture(scope="module")
def plant():
    return load_scene()


@pytest.mark.parametrize("axis_index", [0, 1, 2])
@pytest.mark.parametrize("approach_seconds", [0.8, 0.25])
def test_force_pusher_cannot_drive_motor_through_material(plant, axis_index, approach_seconds):
    m, desc = plant
    s = json.loads(Path(__file__).with_name("motor_side_pose.json").read_text())
    from tests.native import fixture_state

    qpos, qvel = fixture_state(m, s)
    s = dict(s, qpos=qpos.tolist(), qvel=qvel.tolist())
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    d.ctrl[:] = s["reference"]
    mujoco.mj_forward(m, d)
    base = m.body("base").id
    motor = m.body("motor").id
    axis = d.body("base").xmat.reshape(3, 3)[:, axis_index].copy()
    axis *= 1.0 if axis[2] >= 0 else -1.0
    q = int(m.joint("motor_free").qposadr[0])
    # Fixture initialization starts outside the base. All later movement is force-driven.
    d.qpos[q : q + 3] += axis * 0.10
    mujoco.mj_forward(m, d)
    assert sampled_motor_base_overlap(m, d.qpos, ROOT / "assets_local") < 1e-6
    bp = d.body("base").xpos.copy()
    br = d.body("base").xmat.reshape(3, 3).copy()
    mp = d.body("motor").xpos.copy()
    mr = d.body("motor").xmat.reshape(3, 3).copy()
    raw = np.load(ROOT / "assets_local/parts/motor.npz")
    points = np.vstack((raw["vertices"], raw["vertices"][raw["faces"]].mean(1)))
    grid_path = ROOT / "assets_local/sdf/grid_101.bin"
    with grid_path.open("rb") as f:
        header = struct.unpack("<3i4d", f.read(44))
    n = np.array(header[:3])
    origin = np.array(header[3:6])
    h = header[6]
    grid = np.memmap(grid_path, dtype="<f2", mode="r", offset=44, shape=tuple(n[::-1]))

    def overlap():
        local = (
            points @ d.body("motor").xmat.reshape(3, 3).T + d.body("motor").xpos - d.body("base").xpos
        ) @ d.body("base").xmat.reshape(3, 3)
        f = (local - origin) / h
        f = f[np.all((f >= 0) & (f < n - 1), axis=1)]
        i = np.floor(f).astype(int)
        t = f - i
        values = np.zeros(len(i))
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    values += grid[i[:, 2] + z, i[:, 1] + y, i[:, 0] + x].astype(float) * np.prod(
                        np.where([x, y, z], t, 1 - t), axis=1
                    )
        return max(0.0, -float(values.min())) if len(values) else 0.0

    peak = 0.0
    peak_pose = d.qpos.copy()
    for k in range(2400):
        time = k * m.opt.timestep
        for body, p0, r0, kp, kd, kr, dr, cap, tc in [
            (base, bp, br, 30000.0, 90.0, 15.0, 0.12, 100.0, 3.0),
            (
                motor,
                mp - axis * 0.10 * np.clip((time - 0.1) / approach_seconds, 0, 1),
                mr,
                3000.0,
                25.0,
                2.0,
                0.012,
                20.0,
                1.0,
            ),
        ]:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            force = kp * (p0 - d.body(body).xpos) - kd * velocity[3:] + [0, 0, m.body_mass[body] * 9.81]
            torque = (
                kr * Rotation.from_matrix(r0 @ d.body(body).xmat.reshape(3, 3).T).as_rotvec()
                - dr * velocity[:3]
            )
            d.xfrc_applied[body, :3] = force * min(1.0, cap / max(np.linalg.norm(force), 1e-12))
            d.xfrc_applied[body, 3:] = torque * min(1.0, tc / max(np.linalg.norm(torque), 1e-12))
        mujoco.mj_step(m, d)
        if k % 40 == 0:
            depth = overlap()
            if depth > peak:
                peak = depth
                peak_pose = d.qpos.copy()
    assert peak < PENETRATION_LIMIT_M, peak
    assert sampled_motor_base_overlap(m, peak_pose, ROOT / "assets_local") < PENETRATION_LIMIT_M
    assert not m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    assert m.neq == 0 and not d.warning.number.any() and np.isfinite(d.qpos).all()
    if axis_index == 0:
        # The commanded endpoint is the reported bad wall pose; contact must refuse it.
        final = (d.body("motor").xpos - d.body("base").xpos) @ d.body("base").xmat.reshape(3, 3)
        original = (np.array(s["qpos"][q : q + 3]) - bp) @ br
        assert final[0] - original[0] > 0.005
