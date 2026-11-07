"""Native workcell consistency and force-driven insertion bench; no hardware calibration."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.robot import home, ik
from assembly_policy.so101.scene import ROOT, metadata, model
from assembly_policy.so101.workcell import environment_identity


def press(force_cap, timestep, contact_time=None):
    m = model(contact_monitor=True)
    m.opt.timestep = timestep
    if contact_time is not None:
        for name in ("motor_base_volume", "motor_base_seating"):
            m.pair_solref[m.pair(name).id, 0] = contact_time
    d = mujoco.MjData(m)
    home(m, d)
    mujoco.mj_step(m, d, nstep=round(0.5 / timestep))
    base = m.body("base").id
    motor = m.body("motor").id
    q = int(m.joint("motor_free").qposadr[0])
    target = origins(metadata())
    offset = target["motor"] - target["base"]
    rotation = d.body("base").xmat.reshape(3, 3).copy()
    quat = d.body("base").xquat.copy()
    d.qpos[q : q + 3] = d.body("base").xpos + rotation @ (offset + [0, -0.075, 0])
    d.qpos[q + 3 : q + 7] = quat
    v = int(m.joint("motor_free").dofadr[0])
    d.qvel[v : v + 6] = 0
    mujoco.mj_forward(m, d)
    initial_overlap = sampled_motor_base_overlap(m, d.qpos, ROOT / "assets_local")
    assert initial_overlap < 1e-6
    samples = []
    peak_overlap = 0.0
    peak_force = 0.0
    foreign_contacts = set()
    for step in range(round(2.5 / timestep)):
        t = step * timestep
        base_pose = d.body("base")
        R = base_pose.xmat.reshape(3, 3)
        goal = base_pose.xpos + R @ (offset + [0, -0.075 * (1 - np.clip((t - 0.1) / 1.2, 0, 1)), 0])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, motor, velocity, 0)
        force = 10000 * (goal - d.body("motor").xpos) - 45 * velocity[3:] + [0, 0, m.body_mass[motor] * 9.81]
        force *= min(1.0, force_cap / max(np.linalg.norm(force), 1e-12))
        torque = (
            20 * Rotation.from_matrix(R @ d.body("motor").xmat.reshape(3, 3).T).as_rotvec()
            - 0.03 * velocity[:3]
        )
        torque *= min(1.0, 1 / max(np.linalg.norm(torque), 1e-12))
        d.xfrc_applied[motor, :3] = force
        d.xfrc_applied[motor, 3:] = torque
        peak_force = max(peak_force, float(np.linalg.norm(force)))
        mujoco.mj_step(m, d)
        if step % round(0.1 / timestep) == 0:
            depth = sampled_motor_base_overlap(m, d.qpos, ROOT / "assets_local")
            peak_overlap = max(peak_overlap, depth)
            samples.append(
                dict(t=t, error_m=float(np.linalg.norm(d.body("motor").xpos - goal)), overlap_m=depth)
            )
        for c in d.contact:
            bodies = set(m.geom_bodyid[c.geom])
            if motor in bodies:
                for body in bodies - {motor, base, 0}:
                    foreign_contacts.add(m.body(body).name)
    driven_error = float(
        np.linalg.norm(
            d.body("motor").xpos - (d.body("base").xpos + d.body("base").xmat.reshape(3, 3) @ offset)
        )
    )
    d.xfrc_applied[:] = 0
    mujoco.mj_step(m, d, nstep=round(0.5 / timestep))
    error = float(
        np.linalg.norm(
            d.body("motor").xpos - (d.body("base").xpos + d.body("base").xmat.reshape(3, 3) @ offset)
        )
    )
    overlap = sampled_motor_base_overlap(m, d.qpos, ROOT / "assets_local")
    angle = float(
        Rotation.from_matrix(
            d.body("base").xmat.reshape(3, 3).T @ d.body("motor").xmat.reshape(3, 3)
        ).magnitude()
    )
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, motor, velocity, 0)
    return dict(
        force_cap_N=force_cap,
        timestep_s=timestep,
        diagnostic_contact_time_override_s=contact_time,
        initial_overlap_m=initial_overlap,
        driven_position_error_m=driven_error,
        released_position_error_m=error,
        peak_sampled_overlap_m=max(peak_overlap, overlap),
        final_overlap_m=overlap,
        peak_force_N=peak_force,
        warnings=d.warning.number.tolist(),
        released_rotation_error_rad=angle,
        released_linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
        other_motor_contacts=sorted(foreign_contacts),
        samples=samples,
        inserted_and_released=bool(
            error < 0.001
            and angle < 0.03
            and overlap < PENETRATION_LIMIT_M
            and np.linalg.norm(velocity[3:]) < 0.01
            and not d.warning.number.any()
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts_local/shared_workcell")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = model(contact_monitor=True)
    d = mujoco.MjData(m)
    home(m, d)
    d.qpos.copy()
    mujoco.mj_step(m, d, nstep=4000)
    settling = {}
    for name in ("base", "motor", "horn", "screw", "retainer", "yoke"):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body(name).id, velocity, 0)
        settling[name] = dict(
            position=d.body(name).xpos.tolist(),
            linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
            angular_speed_rad_s=float(np.linalg.norm(velocity[:3])),
        )
    reaches = {}
    for side in ("left", "right"):
        for part in ("base", "motor"):
            point = d.body(part).xpos + [0, 0, 0.05]
            try:
                ik(m, d.qpos, side, point)
                reaches[side + "_" + part] = "position-only approach reachable"
            except ValueError as e:
                reaches[side + "_" + part] = str(e)
    renderer = mujoco.Renderer(m, height=540, width=960)
    try:
        import PIL.Image

        renderer.update_scene(d, camera="workspace_camera")
        PIL.Image.fromarray(renderer.render()).save(args.out / "layout.png")
    finally:
        renderer.close()
    report = dict(
        environment=environment_identity(),
        scope="Native simulation consistency and parameter sensitivity; physical dimensions, forces and friction have not been measured.",
        settling=settling,
        warnings=d.warning.number.tolist(),
        approach_reach=reaches,
        insertion_axis_world=(d.body("base").xmat.reshape(3, 3) @ np.array([0, 1, 0])).tolist(),
        loose_overlap_m=sampled_motor_base_overlap(m, d.qpos, ROOT / "assets_local"),
        presses=[],
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for force, dt in ((5.0, 0.0005), (20.0, 0.0005), (5.0, 0.00025), (20.0, 0.00025)):
        result = press(force, dt)
        report["presses"].append(result)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
