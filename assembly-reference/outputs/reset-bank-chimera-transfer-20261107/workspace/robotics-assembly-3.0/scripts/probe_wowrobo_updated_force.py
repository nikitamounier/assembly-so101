"""Force-driven WowRobo probes using accelerated grid-SDF contacts and sparse solving."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assembly_policy.so101.wowrobo_contact_backend import (
    load_backend,
    set_surface_mode,
    surface_stats,
)

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "cover": (
        "03_cover_aligned",
        "06_cover_seated",
        "base_side_cover",
        "base_side_cover",
        ["base_servo"],
        [0, 0, -1],
    ),
    "connector": (
        "08_connector_aligned",
        "11_connector_seated",
        "shoulder_connector",
        "shoulder_connector",
        ["base_servo", "base_side_cover"],
        [-1, 0, 0],
    ),
    "pin": (
        "13_pin_aligned",
        "16_pin_seated",
        "assembly_pin",
        "assembly_pin",
        ["base_servo", "base_side_cover", "shoulder_connector"],
        [0, 0, -1],
    ),
    "upper": (
        "21_upper_far_side",
        "24_upper_seated",
        "upper_arm_module",
        "target_servo_2_pivot",
        ["base_servo", "base_side_cover", "shoulder_connector", "assembly_pin"],
        [0, 0, -1],
    ),
    "final": (
        "29_final_far_side",
        "32_final_seated",
        "forearm_module",
        "target_servo_3_pivot",
        [
            "base_servo",
            "base_side_cover",
            "shoulder_connector",
            "assembly_pin",
            "upper_arm_module",
        ],
        None,
    ),
}


def descendants(m, root):
    result = {m.body(root).id}
    for i in range(m.nbody):
        if int(m.body_parentid[i]) in result:
            result.add(i)
    return result


def load_state(m, folder, stage):
    d = mujoco.MjData(m)
    mujoco.mj_setState(
        m,
        d,
        np.load(folder / "stages" / f"{stage}_train.npz")["states"][0],
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(m, d)
    return d


def run(args):
    out = args.output or ROOT / "outputs/linear-insertion-updated-20261107" / args.step
    out.mkdir(parents=True, exist_ok=True)
    folder = args.source
    start, end, piece, held, receivers, axis = SPECS[args.step]
    backend_info = load_backend(args.grid_manifest)
    set_surface_mode(args.surface_mode)
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    assert mujoco.mj_versionString() == "4.0.0"
    assert m.nplugin > 0 and m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE
    m.opt.gravity[:] = 0
    active = descendants(m, piece)
    fixtures = set().union(*(descendants(m, x) for x in receivers))
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    active_geoms = []
    receiver_geoms = []
    for gid in range(m.ngeom):
        if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SDF:
            continue
        bid = int(m.geom_bodyid[gid])
        if bid in active:
            m.geom_contype[gid] = 1
            m.geom_conaffinity[gid] = 2
            active_geoms.append(gid)
        elif bid in fixtures:
            m.geom_contype[gid] = 2
            m.geom_conaffinity[gid] = 1
            receiver_geoms.append(gid)
    # Runtime broad-phase aggregate masks must agree with geometry masks.
    m.body_contype[:] = 0
    m.body_conaffinity[:] = 0
    for gid in active_geoms + receiver_geoms:
        bid = int(m.geom_bodyid[gid])
        m.body_contype[bid] |= m.geom_contype[gid]
        m.body_conaffinity[bid] |= m.geom_conaffinity[gid]
    d = load_state(m, folder, start)
    ref = load_state(m, folder, end)
    if axis is None:
        # Final module inserts along the installed upper receiver's local -Z.
        axis = -d.body("upper_arm_module").xmat.reshape(3, 3)[:, 2]
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    bid = m.body(held).id
    # Force at case geom center for articulated servo, otherwise body COM.
    point_geom = {"upper": "servo_2_case", "final": "servo_3_case"}.get(args.step)

    def point(data):
        return (
            data.geom(point_geom).xpos.copy() if point_geom else data.xipos[bid].copy()
        )

    origin = point(d)
    refpoint = point(ref)
    travel = float(np.dot(refpoint - origin, axis))
    q0 = d.qpos.copy()
    frozen_q = np.ones(m.nq, dtype=bool)
    frozen_v = np.ones(m.nv, dtype=bool)
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j]) not in active:
            continue
        qa = int(m.jnt_qposadr[j])
        va = int(m.jnt_dofadr[j])
        free = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        frozen_q[qa : qa + (7 if free else 1)] = False
        frozen_v[va : va + (6 if free else 1)] = False
        if free:
            m.dof_damping[va : va + 3] = args.damping
            m.dof_damping[va + 3 : va + 6] = 0.03
    mujoco.mj_forward(m, d)
    initial_contacts = int(d.ncon)
    surface_stats(reset=True)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    width, height = 960, 640
    fps = 24
    frames = int(args.seconds * fps)
    ffmpeg = "/Users/nikitamounier/.local/bin/ffmpeg"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "19",
        "-movflags",
        "+faststart",
        str(out / "insertion.mp4"),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (origin + refpoint) / 2
    cam.lookat[2] = max(cam.lookat[2], 0.08)
    cam.distance = 0.53
    cam.azimuth = 125 if args.step != "connector" else 55
    cam.elevation = -24
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0, 0, 0.1]
    overview.distance = 1.1
    overview.azimuth = 90
    overview.elevation = -35
    rows = []
    stopped = None
    peak_contacts = 0
    min_dist = 0.0
    step = 0
    opts = mujoco.MjvOption()
    opts.geomgroup[1] = 0
    opts.geomgroup[3] = 0
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for frame in range(frames):
            target = (frame + 1) / fps
            force_mag = 0.0
            while step * m.opt.timestep < target - 1e-9 and stopped is None:
                d.qpos[frozen_q] = q0[frozen_q]
                d.qvel[frozen_v] = 0
                mujoco.mj_forward(m, d)
                force_mag = args.force * min(1.0, step * m.opt.timestep / 0.8)
                force = force_mag * axis
                d.qfrc_applied[:] = 0
                mujoco.mj_applyFT(
                    m, d, force, np.zeros(3), point(d), bid, d.qfrc_applied
                )
                mujoco.mj_step(m, d)
                d.qpos[frozen_q] = q0[frozen_q]
                d.qvel[frozen_v] = 0
                mujoco.mj_forward(m, d)
                step += 1
                displacement = point(d) - origin
                axial = float(displacement @ axis)
                lateral = float(np.linalg.norm(displacement - axial * axis))
                peak_contacts = max(peak_contacts, int(d.ncon))
                if d.ncon:
                    min_dist = min(min_dist, float(np.min(d.contact.dist[: d.ncon])))
                if not np.all(np.isfinite(d.qpos)):
                    stopped = "nonfinite state"
                elif axial >= max(travel, 0):
                    stopped = "Reference depth reached (seating unverified)"
                elif np.linalg.norm(displacement) > max(abs(travel) + 0.005, 0.03):
                    stopped = "travel safety limit"
                elif lateral > 0.035:
                    stopped = "lateral safety limit"
                elif np.linalg.norm(d.qvel[~frozen_v]) > 12:
                    stopped = "velocity safety limit"
            displacement = point(d) - origin
            axial = float(displacement @ axis)
            current = point(d)
            renderer.update_scene(d, camera=cam, scene_option=opts)
            # World-space yellow force arrow, length is an illustrative glyph.
            scene = renderer.scene
            if stopped is not None:
                force_mag = 0.0
                d.qfrc_applied[:] = 0
            if stopped is None and scene.ngeom < scene.maxgeom:
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.array([1.0, 0.55, 0.05, 1.0]),
                )
                mujoco.mjv_connector(
                    g,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.004,
                    current - axis * 0.065,
                    current - axis * 0.006,
                )
                scene.ngeom += 1
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, width, 110), fill="#f2f5f8")
            draw.text(
                (15, 8),
                f"{args.step.upper()} | LINEAR FORCE PROBE | {start}",
                font=font,
                fill="#172e41",
            )
            draw.text(
                (15, 34),
                f"t={min(target, step * m.opt.timestep):.2f}s   force={force_mag:.2f} N   travel={axial * 1000:+.2f} mm   contacts={d.ncon}",
                font=font,
                fill="#172e41",
            )
            draw.text(
                (15, 62),
                "MuJoCo 4 | grid SDF + surface tree | SPARSE | receiver/helpers fixed",
                font=small,
                fill="#8b3d16",
            )
            draw.text(
                (15, 83),
                "Gravity OFF | direct part force | current poses/inertias; NOT qualified robot insertion",
                font=small,
                fill="#8b3d16",
            )
            draw.rectangle((0, height - 52, width, height), fill="#f2f5f8")
            draw.text(
                (15, height - 43),
                ("Force removed / frame held: " + stopped)
                if stopped
                else "Native force integration: active root + servo hinge remain free",
                font=font,
                fill="#172e41",
            )
            draw.text(
                (15, height - 21),
                f"Authored reference axial travel {travel * 1000:.2f} mm; source-mesh cavities retained in SDF grids",
                font=small,
                fill="#172e41",
            )
            # overview inset shows frozen helper roles without obscuring close view
            renderer.update_scene(d, camera=overview, scene_option=opts)
            inset = Image.fromarray(renderer.render()).resize((240, 160))
            im.paste(inset, (width - 250, height - 222))
            proc.stdin.write(np.asarray(im).tobytes())
            if frame in [0, frames // 2, frames - 1]:
                im.save(out / f"frame-{frame:03d}.png")
            row = {
                "time_s": float(d.time),
                "force_N": (force_mag * axis).tolist(),
                "held_point_m": current.tolist(),
                "axial_mm": axial * 1000,
                "contacts": int(d.ncon),
                "min_contact_distance_m": float(np.min(d.contact.dist[: d.ncon]))
                if d.ncon
                else None,
            }
            rows.append(row)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    np.savez_compressed(out / "final-state.npz", qpos=d.qpos, qvel=d.qvel)
    receipt = {
        "step": args.step,
        "contact_mode": "accelerated_grid_sdf",
        "start": start,
        "reference": end,
        "source_scene_sha256": hashlib.sha256(
            (folder / "scene.xml").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "force_max_N": args.force,
        "axis_world": axis.tolist(),
        "damping_Ns_per_m": args.damping,
        "initial_contacts": initial_contacts,
        "peak_contacts": peak_contacts,
        "min_contact_distance_m": min_dist,
        "final_axial_mm": rows[-1]["axial_mm"],
        "authored_reference_axial_mm": travel * 1000,
        "stopped": stopped,
        "physical_success": False,
        "qualification": "UNQUALIFIED accelerated grid-SDF force fixture probe; no seating or grasp-retention success asserted",
        "fixtures": "All helper joints and nonactive target bodies restored each substep. Active component has free root and internal hinges. No active qpos prescription.",
        "gravity": [0, 0, 0],
        "direct_force_body": held,
        "active_contact_geoms": [m.geom(x).name for x in active_geoms],
        "receiver_contact_geoms": [m.geom(x).name for x in receiver_geoms],
        "trajectory": rows,
        "simulated_scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "backend": backend_info,
        "surface_mode": args.surface_mode,
        "surface_stats": surface_stats(reset=False),
        "mujoco_version": mujoco.mj_versionString(),
        "plugin_count": m.nplugin,
        "final_lateral_mm": float(
            np.linalg.norm(
                (point(d) - origin) - float((point(d) - origin) @ axis) * axis
            )
        )
        * 1000,
        "reference_orientation_error_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (
                            np.trace(
                                (
                                    d.geom(point_geom).xmat.reshape(3, 3)
                                    if point_geom
                                    else d.body(held).xmat.reshape(3, 3)
                                ).T
                                @ (
                                    ref.geom(point_geom).xmat.reshape(3, 3)
                                    if point_geom
                                    else ref.body(held).xmat.reshape(3, 3)
                                )
                            )
                            - 1
                        )
                        / 2,
                        -1,
                        1,
                    )
                )
            )
        ),
        "solver": {
            "jacobian": int(m.opt.jacobian),
            "timestep": m.opt.timestep,
            "iterations": m.opt.iterations,
            "ls_iterations": m.opt.ls_iterations,
            "tolerance": m.opt.tolerance,
            "noslip_iterations": m.opt.noslip_iterations,
            "noslip_tolerance": m.opt.noslip_tolerance,
            "sdf_initpoints": m.opt.sdf_initpoints,
            "sdf_iterations": m.opt.sdf_iterations,
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "trajectory"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", choices=SPECS, required=True)
    p.add_argument(
        "--source",
        "--folder",
        dest="source",
        type=Path,
        default=ROOT / "outputs/linear-insertion-20261107/source-snapshot",
    )
    p.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "outputs/linear-insertion-updated-20261107/scene.xml",
    )
    p.add_argument(
        "--grid-manifest",
        type=Path,
        default=ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json",
    )
    p.add_argument("--surface-mode", type=int, choices=[0, 1], default=1)
    p.add_argument("--output", type=Path)
    p.add_argument("--force", type=float, default=0.5)
    p.add_argument("--damping", type=float, default=10.0)
    p.add_argument("--seconds", type=float, default=4.0)
    run(p.parse_args())
