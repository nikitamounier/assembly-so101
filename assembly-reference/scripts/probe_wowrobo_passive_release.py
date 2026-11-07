"""Independent gravity release of an actual force-held endpoint, with no controller."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--seconds", type=float, default=1.0)
a = p.parse_args()
out = Path(a.folder)
info = json.loads((out / "result.json").read_text())
load_backend(info["grid_manifest"])
set_surface_mode(1)
m = mujoco.MjModel.from_xml_path(info["scene"])
d = mujoco.MjData(m)
z = np.load(out / "trajectory.npz")
d.qpos[:] = z["qpos"][-1]
d.qvel[:] = z["qvel"][-1]
mujoco.mj_forward(m, d)
initial = d.geom("servo_3_case").xpos.copy()
rotation = d.geom("servo_3_case").xmat.reshape(3, 3).copy()
angle = float(d.joint("target_servo_3_shaft").qpos[0])
rows = []
maxdepth = 0.0
for i in range(int(a.seconds / m.opt.timestep) + 1):
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    if i % 20 == 0:
        rows.append(
            (
                float(d.time),
                d.qpos.copy(),
                d.qvel.copy(),
                d.geom("servo_3_case").xpos.copy(),
                float(d.joint("target_servo_3_shaft").qpos[0]),
            )
        )
    for c in d.contact:
        maxdepth = min(maxdepth, float(c.dist))
    mujoco.mj_step(m, d)
rot = d.geom("servo_3_case").xmat.reshape(3, 3)
rotdeg = float(
    np.degrees(np.arccos(np.clip((np.trace(rotation.T @ rot) - 1) / 2, -1, 1)))
)
result = {
    "seconds": a.seconds,
    "all_applied_forces_zero": True,
    "receiver_fixed": True,
    "motor_drift_m": float(np.linalg.norm(d.geom("servo_3_case").xpos - initial)),
    "motor_rotation_drift_deg": rotdeg,
    "hinge_drift_deg": float(
        np.degrees(d.joint("target_servo_3_shaft").qpos[0] - angle)
    ),
    "max_qvel": float(np.max(np.abs(d.qvel))),
    "minimum_contact_distance_m": maxdepth,
    "initial_endpoint_source": str(out / "trajectory.npz"),
    "status": "bounded_retention_diagnostic_not_training_admission",
}
(out / "release.json").write_text(json.dumps(result, indent=2) + "\n")
np.savez_compressed(
    out / "release.npz",
    time=[r[0] for r in rows],
    qpos=[r[1] for r in rows],
    qvel=[r[2] for r in rows],
    motor_position=[r[3] for r in rows],
    hinge_angle=[r[4] for r in rows],
)
print(json.dumps(result, indent=2))
