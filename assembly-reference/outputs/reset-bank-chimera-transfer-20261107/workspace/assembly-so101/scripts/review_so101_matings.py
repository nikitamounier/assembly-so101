"""Record native mating targets and unassisted, misaligned free drops.

Initialization assigns a pose once. Subsequent motion uses the unchanged shared
model, gravity and contact only. Saved traces separate simulation from rendering;
replaying a trace is a recording, not actuation or a learned policy.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from assembly_policy.so101 import scene
from assembly_policy.so101.contact_audit import sampled_assembly_overlaps, table_overlaps
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.goals import placement_errors
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.robot import home
from assembly_policy.so101.task_scenes import task_scene
from assembly_policy.so101.workcell import configuration, environment_identity

CASES = {
    "motor-right": ("motor", [3.0, 0.0], 60.0, [0.0, 0.0, 0.0]),
    "motor-left": ("motor", [-3.0, 2.0], 60.0, [0.0, 0.0, 0.0]),
    "motor-tilted": ("motor", [2.0, -1.0], 60.0, [2.0, -2.0, 3.0]),
    "motor-outside": ("motor", [0.0, -25.0], 60.0, [0.0, 0.0, 0.0]),
    "horn": ("horn", [1.0, 0.0], 10.0, [2.0, 0.0, 0.0]),
    "screw": ("screw", [1.0, 0.0], 12.0, [2.0, 0.0, 0.0]),
    "retainer": ("retainer", [2.0, 0.0], 50.0, [2.0, 0.0, 0.0]),
    "yoke": ("yoke", [2.0, 0.0], 60.0, [2.0, 0.0, 0.0]),
}


def default_output():
    return scene.ROOT / "artifacts_local" / ("mating-review-" + configuration()["profile"])


def identity(model):
    binary = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    return dict(
        environment=environment_identity(),
        qualification_limits=dict(penetration_m=PENETRATION_LIMIT_M),
        native_model_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
    )


def restore(model, prepared):
    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "ctrl"):
        getattr(data, name)[:] = prepared[name]
    mujoco.mj_forward(model, data)
    return data


def record(case, output, seconds, timestep=None):
    task, xy, height, tilt = CASES[case]
    model = scene.model(contact_monitor=True)
    if timestep is not None:
        model.opt.timestep = timestep
    # Motor cases reproduce the existing clear-drop regression exactly. Other
    # cases start from a qualified completed prefix with its active part lifted.
    if task == "motor":
        data = mujoco.MjData(model)
        home(model, data)
        target = origins(scene.metadata())
        q = int(model.joint("motor_free").qposadr[0])
        base = data.body("base")
        data.qpos[q : q + 3] = base.xpos + base.xmat.reshape(3, 3) @ (target["motor"] - target["base"])
        data.qpos[q + 3 : q + 7] = base.xquat.copy()
    else:
        data = restore(model, task_scene(model, task))
        q = int(model.joint(task + "_free").qposadr[0])
    data.qpos[q : q + 3] += np.r_[xy, height] / 1000.0
    delta = Rotation.from_euler("xyz", tilt, degrees=True).as_quat(scalar_first=True)
    quat = data.qpos[q + 3 : q + 7].copy()
    mujoco.mju_mulQuat(data.qpos[q + 3 : q + 7], delta, quat)
    v = int(model.joint(task + "_free").dofadr[0])
    data.qvel[v : v + 6] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)
    body = model.body(task).id
    names = [p for p in scene.PARTS if p != "screw"]
    initial_pairs = sampled_assembly_overlaps(data, names, scene.ASSETS)
    initial_overlap = max(initial_pairs.values(), default=0.0)
    active_overlap = max(
        (value for pair, value in initial_pairs.items() if task in pair.split("/")), default=0.0
    )
    active_depth = max([0.0, *[-float(c.dist) for c in data.contact if body in model.geom_bodyid[c.geom]]])
    if active_depth > 1e-6 or active_overlap > 1e-6:
        raise ValueError(
            f"Overlapping drop initialization: {case}, native={active_depth:g}, material={active_overlap:g} m"
        )
    times, poses, velocities, samples = [0.0], [data.qpos.copy()], [data.qvel.copy()], []
    peak_depth, peak_speed, peak_angular = 0.0, 0.0, 0.0
    peak_contact = None
    peak_contact_pose = data.qpos.copy()
    peak_material, pairs_peak = initial_overlap, {}
    floor_data = mujoco.MjData(model)
    peak_floor = max(table_overlaps(data, names, scene.ASSETS).values(), default=0.0)
    peak_floor_pose = data.qpos.copy()
    peak_floor_contact = dict(time_s=0.0, depth_m=peak_floor)
    stride = max(1, round(0.01 / model.opt.timestep))
    start = time.monotonic()
    for step in range(1, round(seconds / model.opt.timestep) + 1):
        # mj_step contacts describe the pre-integration pose; material sampling
        # below refreshes kinematics for the saved current qpos explicitly.
        previous_pose = data.qpos.copy()
        mujoco.mj_step(model, data)
        # Audit every integrated pose in a separate data buffer. This adds no
        # force or state update to the simulated world.
        floor_data.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, floor_data)
        floor_depths = table_overlaps(floor_data, names, scene.ASSETS)
        floor_depth = max(floor_depths.values(), default=0.0)
        if floor_depth > peak_floor:
            peak_floor = floor_depth
            peak_floor_pose = data.qpos.copy()
            peak_floor_contact = dict(
                time_s=float(data.time), depth_m=floor_depth, pair=max(floor_depths, key=floor_depths.get)
            )
        peak_material = max(peak_material, floor_depth)
        for pair, value in floor_depths.items():
            pairs_peak[pair] = max(value, pairs_peak.get(pair, 0.0))
        depth = max(0.0, -min(data.contact.dist, default=0.0))
        if depth > peak_depth:
            contact = min(data.contact, key=lambda c: c.dist)
            peak_depth = depth
            peak_contact_pose = previous_pose
            peak_contact = dict(
                time_s=float(data.time - model.opt.timestep),
                geoms=[model.geom(int(g)).name for g in contact.geom],
                position_m=contact.pos.tolist(),
                depth_m=depth,
            )
        peak_speed = max(peak_speed, float(np.linalg.norm(data.qvel[v : v + 3])))
        peak_angular = max(peak_angular, float(np.linalg.norm(data.qvel[v + 3 : v + 6])))
        if step % stride == 0:
            mujoco.mj_kinematics(model, data)
            pairs = sampled_assembly_overlaps(data, names, scene.ASSETS)
            material = max(pairs.values(), default=0.0)
            peak_material = max(peak_material, material)
            for pair, value in pairs.items():
                pairs_peak[pair] = max(value, pairs_peak.get(pair, 0.0))
            samples.append(
                dict(
                    time_s=float(data.time),
                    material_overlap_m=material,
                    speed_m_s=float(np.linalg.norm(data.qvel[v : v + 3])),
                )
            )
            times.append(float(data.time))
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
        if step % round(1 / model.opt.timestep) == 0:
            print(
                f"{case}: {data.time:.1f}s, peak material {peak_material * 1000:.3f} mm, native {peak_depth * 1000:.3f} mm",
                flush=True,
            )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.warning.number.any():
            break
    audit = mujoco.MjData(model)
    audit.qpos[:] = peak_contact_pose
    mujoco.mj_kinematics(model, audit)
    peak_pose_pairs = sampled_assembly_overlaps(audit, names, scene.ASSETS)
    peak_pose_material = max(peak_pose_pairs.values(), default=0.0)
    peak_material = max(peak_material, peak_pose_material)
    for pair, value in peak_pose_pairs.items():
        pairs_peak[pair] = max(value, pairs_peak.get(pair, 0.0))
    mujoco.mj_forward(model, data)
    final_speed = float(np.linalg.norm(data.qvel[v : v + 3]))
    settling_peak = max(
        (s["speed_m_s"] for s in samples if s["time_s"] >= data.time - 0.4 - 1e-9), default=final_speed
    )
    report = dict(
        case=case,
        task=task,
        **identity(model),
        timestep_s=float(model.opt.timestep),
        xy_offset_world_mm=xy,
        lift_world_mm=height,
        tilt_world_xyz_deg=tilt,
        initial_active_native_depth_m=active_depth,
        initial_active_material_overlap_m=active_overlap,
        initial_material_overlap_m=initial_overlap,
        simulated_seconds=float(data.time),
        computation_seconds=time.monotonic() - start,
        peak_native_depth_m=peak_depth,
        peak_native_contact=peak_contact,
        peak_native_pose_qpos=peak_contact_pose.tolist(),
        peak_native_pose_material_m=peak_pose_material,
        peak_original_floor_overlap_m=peak_floor,
        floor_sample_interval_s=float(model.opt.timestep),
        material_sample_interval_s=stride * float(model.opt.timestep),
        peak_floor_pose_qpos=peak_floor_pose.tolist(),
        peak_floor_contact=peak_floor_contact,
        peak_sampled_material_m=peak_material,
        material_pair_peaks_m=pairs_peak,
        peak_speed_m_s=peak_speed,
        peak_angular_speed_rad_s=peak_angular,
        final_speed_m_s=final_speed,
        final_0_4s_peak_speed_m_s=settling_peak,
        final_placement_errors_m=placement_errors(data, task, scene.metadata(), origins(scene.metadata())),
        warnings=data.warning.number.tolist(),
        finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        samples=samples,
        scope="Free drop after one explicit initialization; no support, guide, weld or pose correction. All bodies remain dynamic. Screw mesh excluded from regular-part overlap audit; no claim of threading.",
    )
    report["physics_checks_passed"] = bool(
        report["finite"]
        and not any(report["warnings"])
        and peak_material < PENETRATION_LIMIT_M
        and peak_depth < PENETRATION_LIMIT_M
        and peak_speed < 2
        and final_speed < 0.01
        and settling_peak < 0.01
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / (case + ".npz"), times=times, qpos=poses, qvel=velocities, ctrl=data.ctrl.copy()
    )
    (output / (case + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("samples", "environment")}, indent=2),
        flush=True,
    )


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def view(renderer, data, cam, task):
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer.update_scene(data, camera=cam, scene_option=option)
    # Display-only highlight; the model, collision geometry and trace stay intact.
    active = data.model.body(task).id
    for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
            if data.model.geom_bodyid[geom.objid] == active:
                geom.rgba[:] = [0.15, 0.75, 0.9, 1.0]
    return Image.fromarray(renderer.render())


def render(case, output):
    report = json.loads((output / (case + ".json")).read_text())
    model = scene.model(contact_monitor=True)
    model.opt.timestep = report["timestep_s"]
    # Rendering replays poses; it never re-simulates them. Preserve the recorded
    # source identity, while requiring the exact compiled model for decoding.
    if identity(model)["native_model_sha256"] != report["native_model_sha256"]:
        raise ValueError("Saved recording does not match the compiled model")
    data = mujoco.MjData(model)
    trace = np.load(output / (case + ".npz"))
    peak_material = report["peak_sampled_material_m"]
    checks_passed = report["physics_checks_passed"]
    dense_path = output / (case + "-dense.json")
    if dense_path.exists():
        dense = json.loads(dense_path.read_text())
        if dense["trace_sha256"] != hashlib.sha256((output / (case + ".npz")).read_bytes()).hexdigest():
            raise ValueError("Held-out geometry audit belongs to another recording")
        if (
            "recording_sha256" in dense
            and dense["recording_sha256"]
            != hashlib.sha256((output / (case + ".json")).read_bytes()).hexdigest()
        ):
            raise ValueError("Held-out geometry audit belongs to different recording metadata")
        peak_material = max(peak_material, dense["max_held_out_material_m"])
        checks_passed = checks_passed and dense["passed"]
    data.qpos[:] = trace["qpos"][0]
    data.ctrl[:] = trace["ctrl"]
    mujoco.mj_forward(model, data)
    anchor = data.body("base").xpos.copy()
    anchor[2] += 0.035
    distance = 0.25
    if report["task"] == "yoke":
        anchor[2] += 0.03
        distance = 0.34
    if report["task"] in ("horn", "screw"):
        # Keep the small part legible, while covering its full free-fall path.
        anchor = data.body("horn").xpos.copy()
        if report["task"] == "horn":
            anchor[2] -= report["lift_world_mm"] / 2000.0
        else:
            anchor[2] += 0.005
        distance = 0.105
    cams = [camera(anchor, distance, 125, -22), camera(anchor, distance, -40, -55)]
    renderer = mujoco.Renderer(model, width=640, height=480)
    path = output / (case + ".mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x580",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    frames = []
    try:
        # One-second initial and final holds; motion itself is 1x simulation time.
        duration = float(trace["times"][-1])
        timeline = np.clip(np.arange(round((duration + 2) * 30)) / 30 - 1, 0, duration)
        for n, t in enumerate(timeline):
            i = int(np.argmin(abs(trace["times"] - t)))
            data.qpos[:] = trace["qpos"][i]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            canvas = Image.new("RGB", (1280, 580), "#101820")
            for x, cam in zip((0, 640), cams):
                canvas.paste(view(renderer, data, cam, report["task"]), (x, 66))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (18, 8),
                f"{case.upper()} | FREE RELEASE | t = {trace['times'][i]:.2f} s",
                font=font(23),
                fill="white",
            )
            status = "CHECKS PASS" if checks_passed else "CHECKS FAIL"
            draw.text((1030, 12), status, font=font(20), fill="#7cddab" if checks_passed else "#ffad91")
            draw.text(
                (18, 38),
                f"XY offset {report['xy_offset_world_mm']} mm | lift {report['lift_world_mm']:g} mm | tilt {report['tilt_world_xyz_deg']} deg",
                font=font(17),
                fill="#b8d0dd",
            )
            draw.text(
                (18, 552),
                f"Peak sampled mesh {peak_material * 1000:.3f} mm | native {report['peak_native_depth_m'] * 1000:.3f} mm | cyan = released part | 1x simulation time",
                font=font(17),
                fill="#b8d0dd",
            )
            encoder.stdin.write(canvas.tobytes())
            if n in (30, round((duration / 2 + 1) * 30), len(timeline) - 1):
                frames.append(canvas.copy())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        renderer.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    strip = Image.new("RGB", (1280, 580 * len(frames)))
    for i, frame in enumerate(frames):
        strip.paste(frame, (0, i * 580))
    strip.save(output / (case + "-frames.jpg"), quality=90)
    print(path)


def targets(output):
    output.mkdir(parents=True, exist_ok=True)
    model = scene.model(contact_monitor=True)
    model_identity = identity(model)
    renderer = mujoco.Renderer(model, width=640, height=420)
    canvas = Image.new("RGB", (1280, 5 * 470 + 90), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "MATING TARGETS | " + environment_identity()["profile"], font=font(28), fill="white")
    draw.text(
        (18, 50),
        "Qualified explicit initializations, not an executed assembly. Cyan highlights the current part.",
        font=font(20),
        fill="#b8d0dd",
    )
    reports = {}
    try:
        for row, task in enumerate(scene.ASSEMBLY_TASKS):
            prepared = task_scene(model, task)
            data = restore(model, prepared)
            np.savez_compressed(
                output / ("target-" + task + ".npz"),
                times=[0.0],
                qpos=[prepared["qpos"]],
                qvel=[prepared["qvel"]],
                ctrl=prepared["ctrl"],
            )
            (output / ("target-" + task + ".json")).write_text(
                json.dumps(
                    dict(
                        case="target-" + task,
                        task=task,
                        **model_identity,
                        timestep_s=float(model.opt.timestep),
                        scope="Initialized native reward target, not an executed insertion.",
                        preparation=prepared["preparation"],
                    ),
                    indent=2,
                )
                + "\n"
            )
            anchor = data.body("motor").xpos.copy()
            anchor[2] += 0.012
            distance = 0.17 if task in ("motor", "retainer", "yoke") else 0.10
            if task in ("horn", "screw"):
                anchor = data.body("horn").xpos.copy()
            for x, az, el in ((0, 125, -25), (640, -40, -60)):
                canvas.paste(
                    view(renderer, data, camera(anchor, distance, az, el), task), (x, 90 + row * 470 + 45)
                )
            draw.text(
                (18, 100 + row * 470),
                f"{row + 1}. {task.upper()} | native reward qualified",
                font=font(24),
                fill="white",
            )
            reports[task] = prepared["preparation"]
    finally:
        renderer.close()
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / "matings.jpg", quality=95)
    (output / "matings.json").write_text(json.dumps(dict(**model_identity, targets=reports), indent=2) + "\n")
    print(output / "matings.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("targets", "record", "render"))
    parser.add_argument("--case", choices=(*CASES, "all"), default="motor-right")
    parser.add_argument("--out", type=Path, default=default_output())
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--timestep", type=float)
    args = parser.parse_args()
    if args.seconds <= 0 or (args.timestep is not None and args.timestep <= 0):
        parser.error("Durations must be positive")
    if args.action == "targets":
        targets(args.out)
    else:
        for case in CASES if args.case == "all" else [args.case]:
            if args.action == "record":
                record(case, args.out, args.seconds, args.timestep)
            else:
                render(case, args.out)


if __name__ == "__main__":
    main()
