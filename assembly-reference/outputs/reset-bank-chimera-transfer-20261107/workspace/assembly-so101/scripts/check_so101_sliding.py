"""Force-driven nominal assembly paths, distinct from initialized reward targets.

Force/torque feedback supports the receiver and guides the active part. These are
contact-mechanics fixtures, not robot grasps or learned assembly. No part pose
is assigned after initialization, and the active part is released at the end.
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from assembly_policy.so101 import scene
from assembly_policy.so101.fastener import origins, target_rotations
from assembly_policy.so101.goals import RECEIVERS, placement_errors
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.task_scenes import audit_reward, seed_scene, task_scene
from assembly_policy.so101.workcell import configuration, environment_identity


def material_overlap(data, active, others):
    """Bidirectional original mesh samples, independent of generated contacts."""
    from assembly_policy.so101.contact_audit import sampled_pair_overlap

    return max(
        (sampled_pair_overlap(data, active, other, scene.ASSETS) for other in others if other != "screw"),
        default=0.0,
    )


def assembly_material(data, task):
    from assembly_policy.so101.contact_audit import sampled_assembly_overlaps, sampled_pair_overlap

    prefix = ["base", *scene.ASSEMBLY_TASKS[: scene.ASSEMBLY_TASKS.index(task) + 1]]
    pairs = sampled_assembly_overlaps(data, prefix, scene.ASSETS)
    for other in scene.PARTS:
        if other not in prefix and other != "screw":
            pairs[task + "/" + other] = sampled_pair_overlap(data, task, other, scene.ASSETS)
    return pairs


def limited(vector, cap):
    return vector * min(1.0, cap / max(np.linalg.norm(vector), 1e-12))


def free_body_state(model, data, body):
    """Current world pose/twist, without mj_step's stale derived-state cache."""
    joint = int(model.body_jntadr[body])
    if (
        model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        or model.body_parentid[body] != 0
        or np.any(model.body_ipos[body])
    ):
        raise ValueError("Fixture requires a world-root free body with centred inertia")
    q = int(model.jnt_qposadr[joint])
    v = int(model.jnt_dofadr[joint])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[q + 3 : q + 7])
    rotation = rotation.reshape(3, 3)
    return data.qpos[q : q + 3], rotation, np.r_[rotation @ data.qvel[v + 3 : v + 6], data.qvel[v : v + 3]]


def gains(
    model, body, position_cap=10000.0, rotation_cap=2.0, rotation_stiffness=None, position_stiffness=None
):
    # Motor-scale damping destabilizes a tiny horn when applied explicitly.
    # Bound the natural frequency to500rad/s and use critical damping for
    # the lightest inertia axis. The same gains are used at both timesteps.
    mass = float(model.body_mass[body])
    inertia = float(model.body_inertia[body].min())
    kp = min(position_cap, mass * 500**2)
    kr = min(rotation_cap, inertia * 500**2)
    if rotation_stiffness is not None:
        kr = max(kr, rotation_stiffness)
    if position_stiffness is not None:
        kp = position_stiffness
    return dict(kp=kp, kd=2 * np.sqrt(kp * mass), kr=kr, dr=2 * np.sqrt(kr * inertia))


def free_space_tracking(model, task, rotation_stiffness=None, position_stiffness=None, force_cap=20.0):
    from assembly_policy.so101.robot import home

    d = mujoco.MjData(model)
    home(model, d)
    q = int(model.joint(task + "_free").qposadr[0])
    a = model.body(task).id
    d.qpos[q : q + 7] = [0.2, 0.2, 1.0, 1, 0, 0, 0]
    mujoco.mj_forward(model, d)
    target = d.body(task).xpos + [0.002, 0, 0]
    R = Rotation.from_rotvec([0.03, 0, 0]).as_matrix()
    g = gains(model, a, rotation_stiffness=rotation_stiffness, position_stiffness=position_stiffness)
    v = int(model.joint(task + "_free").dofadr[0])
    damping = model.dof_damping[v : v + 6].copy()
    if rotation_stiffness is not None:
        model.dof_damping[v + 3 : v + 6] += g["dr"]
    if position_stiffness is not None:
        model.dof_damping[v : v + 3] += g["kd"]
    peak_speed = 0.0
    peak_angular = 0.0
    start = d.body(task).xpos.copy()
    for _ in range(round(0.3 / model.opt.timestep)):
        u = min(1.0, d.time / 0.05)
        blend = u * u * (3 - 2 * u)
        moving_target = start + blend * (target - start)
        moving_rotation = Rotation.from_rotvec([0.03 * blend, 0, 0]).as_matrix()
        position, rotation, velocity = free_body_state(model, d, a)
        force = (
            g["kp"] * (moving_target - position) - g["kd"] * velocity[3:] + [0, 0, model.body_mass[a] * 9.81]
        )
        dr = Rotation.from_matrix(moving_rotation @ rotation.T).as_rotvec()
        d.xfrc_applied[a, :3] = limited(force, force_cap) + (
            g["kd"] * velocity[3:] if position_stiffness is not None else 0.0
        )
        d.xfrc_applied[a, 3:] = limited(g["kr"] * dr - g["dr"] * velocity[:3], 0.2) + (
            g["dr"] * velocity[:3] if rotation_stiffness is not None else 0.0
        )
        mujoco.mj_step(model, d)
        peak_speed = max(peak_speed, float(np.linalg.norm(velocity[3:])))
        peak_angular = max(peak_angular, float(np.linalg.norm(velocity[:3])))
    position, rotation, _ = free_body_state(model, d, a)
    error = float(np.linalg.norm(target - position))
    angle = float(Rotation.from_matrix(R @ rotation.T).magnitude())
    model.dof_damping[v : v + 6] = damping
    return dict(
        position_error_m=error,
        rotation_error_rad=angle,
        peak_speed_m_s=peak_speed,
        peak_angular_speed_rad_s=peak_angular,
        passed=bool(
            error < 0.00005
            and angle < 0.002
            and peak_speed < 1
            and peak_angular < 30
            and not d.warning.number.any()
        ),
    )


def contacts(model, data, body):
    rows = []
    for i, c in enumerate(data.contact):
        if body not in model.geom_bodyid[c.geom]:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        rows.append(
            dict(
                geoms=[model.geom(int(g)).name for g in c.geom],
                distance_m=float(c.dist),
                position_m=c.pos.tolist(),
                frame=c.frame.tolist(),
                force=force.tolist(),
            )
        )
    return rows


def audit_target(task):
    """Check reward-qualified targets against all regular predecessor meshes."""
    m = scene.model(contact_monitor=True)
    prepared = task_scene(m, task)
    d = mujoco.MjData(m)
    d.qpos[:] = prepared["qpos"]
    d.qvel[:] = prepared["qvel"]
    d.ctrl[:] = prepared["ctrl"]
    mujoco.mj_forward(m, d)
    names = ["base", *scene.ASSEMBLY_TASKS[: scene.ASSEMBLY_TASKS.index(task) + 1]]
    names = [n for n in names if n != "screw"]
    overlaps = {a + "/" + b: material_overlap(d, a, [b]) for a, b in combinations(names, 2)}
    depth = max([0.0, *[-float(c.dist) for c in d.contact]])
    return dict(
        task=task,
        native_reward_qualified=prepared["preparation"]["native_reward"]["passed"],
        placement_errors_m=prepared["preparation"]["placement_error_m"],
        native_depth_m=depth,
        sampled_material_overlap_m=overlaps,
        passed=bool(
            max(overlaps.values(), default=0.0) < PENETRATION_LIMIT_M and depth < PENETRATION_LIMIT_M
        ),
        scope="Initialized target, not insertion; screw supplement excluded from material comparison",
    )


def slide(
    task,
    timestep,
    force_cap=None,
    extraction=None,
    part_friction=None,
    target_correction=None,
    motor_interface="both",
    noslip=None,
    feed_force=None,
    rotation_stiffness=None,
    position_stiffness=None,
    receiver=None,
    support_body=None,
    entry_lift=0.0,
    approach_s=None,
    on_step=None,
):
    approach_s = (2.0 if task == "yoke" else 1.5) if approach_s is None else float(approach_s)
    if not np.isfinite(approach_s) or approach_s <= 0:
        raise ValueError("Invalid approach duration")
    drive_s = 0.1 + approach_s + 1.4
    rotation_stiffness = (
        {"motor": 15.0, "yoke": 50.0}.get(task, 0.3) if rotation_stiffness is None else rotation_stiffness
    )
    position_stiffness = 10000.0 if position_stiffness is None else position_stiffness
    if motor_interface == "both":
        m = scene.model(contact_monitor=True)
    else:
        scene.load_plugin()
        root = ET.fromstring(scene.build_xml(contact_monitor=True))
        remove = "motor_base_seating" if motor_interface == "volume" else "motor_base_volume"
        node = root.find(f"contact/pair[@name='{remove}']")
        root.find("contact").remove(node)
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        scene.apply_model(m)
    if task == "motor":
        prepared = task_scene(m, task)
    else:
        seed = seed_scene(m, task)
        prepared = dict(qpos=seed.qpos.copy(), qvel=seed.qvel.copy(), ctrl=seed.ctrl.copy())
    m.opt.timestep = timestep
    if noslip is not None:
        if noslip < 0:
            raise ValueError("Invalid NoSlip iteration count")
        m.opt.noslip_iterations = noslip
    if part_friction is not None:
        if not np.isfinite(part_friction) or part_friction < 0:
            raise ValueError("Invalid diagnostic friction")
        active = m.body(task).id
        # Change the moving part only: clearing predecessor friction can let
        # the already-seated motor fall out before a horn reaches it.
        g = m.geom(task + "_contact").id
        m.geom_friction[g] = [part_friction, 0, 0]
        m.geom_priority[g] = 1
        for p in range(m.npair):
            if active in (m.geom_bodyid[m.pair_geom1[p]], m.geom_bodyid[m.pair_geom2[p]]):
                m.pair_friction[p] = [part_friction, part_friction, 0, 0, 0]
    if rotation_stiffness is not None and (not np.isfinite(rotation_stiffness) or rotation_stiffness <= 0):
        raise ValueError("Invalid rotational stiffness")
    if position_stiffness is not None and (not np.isfinite(position_stiffness) or position_stiffness <= 0):
        raise ValueError("Invalid positional stiffness")
    force_cap = (
        {"motor": 20.0, "horn": 5.0, "retainer": 5.0, "yoke": 7.0}[task]
        if force_cap is None
        else float(force_cap)
    )
    if not np.isfinite(force_cap) or force_cap <= 0:
        raise ValueError("Invalid force cap")
    tracking = free_space_tracking(m, task, rotation_stiffness, position_stiffness, force_cap)
    if not tracking["passed"]:
        raise ValueError(f"Unqualified force controller for {task}: {tracking}")
    d = mujoco.MjData(m)
    d.qpos[:] = prepared["qpos"]
    d.qvel[:] = prepared["qvel"]
    d.ctrl[:] = prepared["ctrl"]
    mujoco.mj_forward(m, d)
    support_body = support_body or ("motor" if task in ("horn", "yoke") else "base")
    if support_body not in scene.PARTS or support_body == task:
        raise ValueError("Invalid support body")
    b = m.body(support_body).id
    a = m.body(task).id
    q = int(m.joint(task + "_free").qposadr[0])
    v = int(m.joint(task + "_free").dofadr[0])
    base_position = d.body(support_body).xpos.copy()
    base_rotation = d.body(support_body).xmat.reshape(3, 3).copy()
    receiver = receiver or ("motor" if task == "yoke" else RECEIVERS[task])
    if receiver not in scene.PARTS or receiver == task:
        raise ValueError("Invalid receiving body")
    targets = origins(scene.metadata())
    rotations = target_rotations(scene.metadata())
    offset = targets[task] - targets[receiver]
    # A small guide clearance accommodates the lower lip; canonical goals stay fixed.
    if target_correction is None and task == "yoke":
        target_correction = [0.0, 0.0, -0.00005]
    if target_correction is not None:
        correction = np.asarray(target_correction, float)
        if correction.shape != (3,) or not np.isfinite(correction).all():
            raise ValueError("Invalid target correction")
        offset = offset + correction  # Guide bias must not change the reward/validation target.
    extract = np.asarray(
        configuration()["assembly_plan"]["extractions"][task] if extraction is None else extraction, float
    )
    if extract.shape != (3,) or not np.isfinite(extract).all() or np.linalg.norm(extract) == 0:
        raise ValueError("Invalid extraction direction")
    if not np.isfinite(entry_lift) or entry_lift < 0:
        raise ValueError("Invalid entry lift")
    clearance = np.array([0.0, 0.0, entry_lift])
    others = [p for p in scene.PARTS if p != task]
    # Reject overlapping starts; lengthening the same recorded extraction
    # direction is initialization only and does not change the insertion axis.
    initial = None
    for scale in (1.0, 1.5, 2.0, 3.0):
        receiving = d.body(receiver)
        frame = receiving.xmat.reshape(3, 3) @ rotations[receiver].T
        d.qpos[q : q + 3] = receiving.xpos + frame @ (offset + scale * extract + clearance)
        mujoco.mju_mat2Quat(d.qpos[q + 3 : q + 7], (frame @ rotations[task]).ravel())
        d.qvel[v : v + 6] = 0
        mujoco.mj_forward(m, d)
        depth = max([0.0, *[-float(c.dist) for c in d.contact if a in m.geom_bodyid[c.geom]]])
        overlap = material_overlap(d, task, others)
        if depth < 1e-6 and overlap < 1e-6:
            initial = dict(
                extraction_scale=scale, active_native_depth_m=depth, sampled_material_overlap_m=overlap
            )
            break
    if initial is None:
        return dict(
            task=task,
            timestep_s=timestep,
            passed=False,
            rejected_initialization=True,
            active_native_depth_m=depth,
            sampled_material_overlap_m=overlap,
        )
    feed_force = 0.0 if feed_force is None else float(feed_force)
    if not np.isfinite(feed_force) or not 0 <= feed_force <= force_cap:
        raise ValueError("Invalid feed force")
    active_gains = gains(m, a, rotation_stiffness=rotation_stiffness, position_stiffness=position_stiffness)
    base_gains = gains(m, b, 30000.0, 15.0, rotation_stiffness=15.0 if support_body != "base" else None)
    support_dof = int(m.joint(support_body + "_free").dofadr[0])
    support_damping = m.dof_damping[support_dof + 3 : support_dof + 6].copy()
    if support_body != "base":
        m.dof_damping[support_dof + 3 : support_dof + 6] += base_gains["dr"]
    original_damping = m.dof_damping[v : v + 6].copy()
    if rotation_stiffness is not None:
        m.dof_damping[v + 3 : v + 6] += active_gains["dr"]
    if position_stiffness is not None:
        m.dof_damping[v : v + 3] += active_gains["kd"]
    peak_depth = 0.0
    peak_material = 0.0
    peak_pose = d.qpos.copy()
    samples = []
    material_peaks = {}
    peak_speed = 0.0
    peak_angular_speed = 0.0
    peak_force = 0.0
    peak_torque = 0.0
    previous_pose = d.qpos.copy()
    part_dofs = [int(m.joint(name + "_free").dofadr[0]) for name in scene.PARTS]

    def measure_material(data):
        pairs = assembly_material(data, task)
        for pair, value in pairs.items():
            material_peaks[pair] = max(material_peaks.get(pair, 0.0), value)
        return max(pairs.values(), default=0.0)

    def measure_motion():
        nonlocal peak_speed, peak_angular_speed
        peak_speed = max(peak_speed, max(float(np.linalg.norm(d.qvel[j : j + 3])) for j in part_dofs))
        peak_angular_speed = max(
            peak_angular_speed, max(float(np.linalg.norm(d.qvel[j + 3 : j + 6])) for j in part_dofs)
        )

    def base_support():
        if support_body == "motor" and d.time < 0.1:
            d.xfrc_applied[b] = 0
            return
        position, rotation, velocity = free_body_state(m, d, b)
        d.xfrc_applied[b, :3] = limited(
            base_gains["kp"] * (base_position - position)
            - base_gains["kd"] * velocity[3:]
            + [0, 0, m.body_mass[b] * 9.81],
            20.0,
        )
        dr = Rotation.from_matrix(base_rotation @ rotation.T).as_rotvec()
        d.xfrc_applied[b, 3:] = limited(base_gains["kr"] * dr - base_gains["dr"] * velocity[:3], 0.5)
        if support_body != "base":
            d.xfrc_applied[b, 3:] += base_gains["dr"] * velocity[:3]

    if on_step is not None:
        on_step(m, d)
    for step in range(round(drive_s / timestep)):
        if support_body == "motor" and step == round(0.1 / timestep):
            position, rotation, _ = free_body_state(m, d, b)
            base_position = position.copy()
            base_rotation = rotation.copy()
        if step % round(1.0 / timestep) == 0:
            print(
                f"{task} dt={timestep:g}: t={d.time:.2f}s, contacts={d.ncon}, peak material={peak_material * 1000:.3f}mm",
                file=sys.stderr,
                flush=True,
            )
        t = step * timestep
        u = np.clip((t - 0.1) / approach_s, 0, 1)
        blend = u * u * (3 - 2 * u)
        lowering = np.clip((t - 0.1 - approach_s) / 0.6, 0, 1)
        lowering = lowering * lowering * (3 - 2 * lowering)
        receiving_position, receiving_rotation, _ = free_body_state(m, d, m.body(receiver).id)
        R = receiving_rotation @ rotations[receiver].T
        goal = receiving_position + R @ (offset + scale * extract * (1 - blend) + clearance * (1 - lowering))
        position, rotation, velocity = free_body_state(m, d, a)
        push = -R @ extract / np.linalg.norm(extract) * feed_force if u >= 1 else np.zeros(3)
        net_force = limited(
            active_gains["kp"] * (goal - position)
            - active_gains["kd"] * velocity[3:]
            + push
            + [0, 0, m.body_mass[a] * 9.81],
            force_cap,
        )
        d.xfrc_applied[a, :3] = net_force
        if position_stiffness is not None:
            d.xfrc_applied[a, :3] += active_gains["kd"] * velocity[3:]
        dr = Rotation.from_matrix(R @ rotations[task] @ rotation.T).as_rotvec()
        net_torque = limited(active_gains["kr"] * dr - active_gains["dr"] * velocity[:3], 0.2)
        d.xfrc_applied[a, 3:] = net_torque
        if rotation_stiffness is not None:
            d.xfrc_applied[a, 3:] += active_gains["dr"] * velocity[:3]
        peak_force = max(peak_force, float(np.linalg.norm(net_force)))
        peak_torque = max(peak_torque, float(np.linalg.norm(net_torque)))
        base_support()
        mujoco.mj_step(m, d)
        if on_step is not None:
            on_step(m, d)
        measure_motion()
        depth = max(0.0, -float(min(d.contact.dist, default=0.0)))
        if depth > peak_depth:
            peak_depth = depth
            peak_pose = previous_pose
        previous_pose = d.qpos.copy()
        if step % round(0.01 / timestep) == 0:
            mujoco.mj_kinematics(m, d)
            material = measure_material(d)
            peak_material = max(peak_material, material)
            samples.append(
                dict(
                    time_s=float(d.time),
                    placement_error_m=max(placement_errors(d, task, scene.metadata(), targets).values()),
                    sampled_material_overlap_m=material,
                )
            )
    mujoco.mj_forward(m, d)
    driven_error = max(placement_errors(d, task, scene.metadata(), targets).values())
    driven_state = dict(
        qpos=d.qpos.tolist(),
        qvel=d.qvel.tolist(),
        contacts=contacts(m, d, a),
        active_error_receiver_m=(
            d.body(receiver).xmat.reshape(3, 3).T @ (d.body(task).xpos - d.body(receiver).xpos) - offset
        ).tolist(),
    )
    d.xfrc_applied[a] = 0
    m.dof_damping[v : v + 6] = original_damping
    # Receiver support remains during the first active-part release interval.
    for step in range(round(0.6 / timestep)):
        base_support()
        mujoco.mj_step(m, d)
        if on_step is not None:
            on_step(m, d)
        measure_motion()
        depth = max(0.0, -float(min(d.contact.dist, default=0.0)))
        if depth > peak_depth:
            peak_depth = depth
            peak_pose = previous_pose
        previous_pose = d.qpos.copy()
        if step % round(0.01 / timestep) == 0:
            mujoco.mj_kinematics(m, d)
            material = measure_material(d)
            peak_material = max(peak_material, material)
            samples.append(
                dict(
                    time_s=float(d.time),
                    phase="released",
                    placement_error_m=max(placement_errors(d, task, scene.metadata(), targets).values()),
                    sampled_material_overlap_m=material,
                )
            )
    mujoco.mj_forward(m, d)
    if on_step is not None:
        on_step(m, d)
    peak_depth = max(peak_depth, max(0.0, -float(min(d.contact.dist, default=0.0))))
    data = mujoco.MjData(m)
    data.qpos[:] = peak_pose
    mujoco.mj_kinematics(m, data)
    peak_material = max(peak_material, measure_material(data), measure_material(d))
    errors = placement_errors(d, task, scene.metadata(), targets)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, a, velocity, 0)
    desired = d.body(receiver).xmat.reshape(3, 3) @ rotations[receiver].T @ rotations[task]
    angle = float(Rotation.from_matrix(desired.T @ d.body(task).xmat.reshape(3, 3)).magnitude())
    finite = bool(
        np.isfinite(d.qpos).all()
        and np.isfinite(d.qvel).all()
        and not d.warning.number.any()
        and abs(d.time - drive_s - 0.6) < 1e-8
    )
    # Replay through the actual policy environment with zero external fixture
    # forces, including release of the base and a fresh stability window.
    released = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy())
    m.dof_damping[support_dof + 3 : support_dof + 6] = support_damping
    reward, _ = audit_reward(m, released, task, ticks=40, capture=True)
    return dict(
        task=task,
        timestep_s=timestep,
        force_cap_N=force_cap,
        terminal_feed_force_N=feed_force,
        extraction_base_m=extract.tolist(),
        torque_cap_Nm=0.2,
        base_force_cap_N=20.0,
        base_torque_cap_Nm=0.5,
        free_space_controller=tracking,
        active_gains=active_gains,
        base_gains=base_gains,
        diagnostic_part_friction=part_friction,
        diagnostic_target_correction_m=target_correction,
        diagnostic_motor_interface=motor_interface,
        fixture_rotation_stiffness=rotation_stiffness,
        fixture_position_stiffness=position_stiffness,
        receiving_body=receiver,
        support_body=support_body,
        entry_lift_m=entry_lift,
        approach_duration_s=approach_s,
        drive_duration_s=drive_s,
        diagnostic_noslip_iterations=noslip,
        diagnostic_friction_scope="Active contact geom priority1; active explicit pairs; predecessors unchanged"
        if part_friction is not None
        else None,
        driven_state=driven_state,
        final_state=dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), contacts=contacts(m, d, a)),
        canonical_released_errors_m=placement_errors(d, task, scene.metadata(), origins(scene.metadata())),
        initial=initial,
        driven_placement_error_m=driven_error,
        released_placement_errors_m=errors,
        released_rotation_error_rad=angle,
        released_speed_m_s=float(np.linalg.norm(velocity[3:])),
        peak_native_depth_m=peak_depth,
        peak_sampled_material_overlap_m=peak_material,
        peak_sampled_material_by_pair_m=material_peaks,
        material_sample_interval_s=0.01,
        peak_part_speed_m_s=peak_speed,
        peak_part_angular_speed_rad_s=peak_angular_speed,
        peak_feedback_force_N=peak_force,
        peak_feedback_torque_Nm=peak_torque,
        fixture_method="Clipped PD feedback with implicit damping compensation; temporary fixture, not robot grasps",
        unassisted_native_reward=reward,
        arena_capacity_bytes=int(m.narena),
        peak_arena_bytes=int(d.maxuse_arena),
        peak_contacts=int(d.maxuse_con),
        peak_constraints=int(d.maxuse_efc),
        warnings=d.warning.number.tolist(),
        numerically_valid=finite,
        samples=samples,
        passed=bool(
            finite
            and max(errors.values()) < 0.001
            and angle < 0.03
            and np.linalg.norm(velocity[3:]) < 0.01
            and peak_depth < PENETRATION_LIMIT_M
            and peak_material < PENETRATION_LIMIT_M
            and peak_speed < 1.0
            and peak_angular_speed < 30.0
            and reward["passed"]
        ),
    )


def wall_blocking(timestep, direction):
    """Press a seated motor against either closed base rail, with finite forces."""
    if direction not in (-1, 1):
        raise ValueError("Expected a rail direction of -1 or +1")
    m = scene.model(contact_monitor=True)
    prepared = task_scene(m, "motor")
    m.opt.timestep = timestep
    d = mujoco.MjData(m)
    d.qpos[:] = prepared["qpos"]
    d.qvel[:] = prepared["qvel"]
    d.ctrl[:] = prepared["ctrl"]
    mujoco.mj_forward(m, d)
    base = m.body("base").id
    motor = m.body("motor").id
    anchor, orientation, _ = free_body_state(m, d, base)
    anchor = anchor.copy()
    orientation = orientation.copy()
    initial = orientation.T @ (d.body("motor").xpos - anchor)
    support = gains(m, base, 30000.0, 15.0)
    peak_depth = 0.0
    peak_material = 0.0
    peak_displacement = 0.0
    peak_speed = 0.0
    samples = []
    for step in range(round(0.6 / timestep)):
        position, rotation, velocity = free_body_state(m, d, base)
        d.xfrc_applied[base, :3] = limited(
            support["kp"] * (anchor - position)
            - support["kd"] * velocity[3:]
            + [0, 0, m.body_mass[base] * 9.81],
            20.0,
        )
        error = Rotation.from_matrix(orientation @ rotation.T).as_rotvec()
        d.xfrc_applied[base, 3:] = limited(support["kr"] * error - support["dr"] * velocity[:3], 0.5)
        strength = 20 * min(1.0, d.time / 0.05) if d.time < 0.3 else 0.0
        d.xfrc_applied[motor, :3] = rotation @ np.array([0, 0, direction * strength])
        mujoco.mj_step(m, d)
        peak_depth = max(peak_depth, max(0.0, -float(min(d.contact.dist, default=0.0))))
        position, rotation, _ = free_body_state(m, d, base)
        motor_position, _, velocity = free_body_state(m, d, motor)
        displacement = abs(float((rotation.T @ (motor_position - position) - initial)[2]))
        peak_displacement = max(peak_displacement, displacement)
        peak_speed = max(peak_speed, float(np.linalg.norm(velocity[3:])))
        if step % round(0.01 / timestep) == 0:
            mujoco.mj_kinematics(m, d)
            overlap = material_overlap(d, "motor", ["base"])
            peak_material = max(peak_material, overlap)
            samples.append(
                dict(
                    time_s=float(d.time), rail_displacement_m=displacement, sampled_material_overlap_m=overlap
                )
            )
    finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any())
    return dict(
        direction=direction,
        timestep_s=timestep,
        force_N=20.0,
        peak_rail_displacement_m=peak_displacement,
        peak_native_depth_m=peak_depth,
        peak_sampled_material_overlap_m=peak_material,
        peak_speed_m_s=peak_speed,
        samples=samples,
        numerically_valid=finite,
        passed=bool(
            finite
            and peak_displacement < 0.001
            and peak_depth < PENETRATION_LIMIT_M
            and peak_material < PENETRATION_LIMIT_M
            and peak_speed < 1.0
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=scene.ROOT / "artifacts_local/sliding-check")
    p.add_argument("--tasks", default="motor+horn+retainer+yoke")
    p.add_argument("--timesteps", default=".0005,.00025")
    p.add_argument("--force-cap", type=float)
    p.add_argument(
        "--extraction",
        type=float,
        nargs=3,
        help="Diagnostic base-frame start offset; source config is unchanged",
    )
    p.add_argument(
        "--feed-force", type=float, help="Terminal axial preload; default zero uses position guidance"
    )
    p.add_argument(
        "--rotation-stiffness",
        type=float,
        help="Minimum fixture rotational stiffness with native implicit damping; removed on release",
    )
    p.add_argument(
        "--position-stiffness",
        type=float,
        help="Fixture positional stiffness with native implicit damping; removed on release",
    )
    p.add_argument(
        "--receiver",
        choices=scene.PARTS,
        help="Override the actual mating reference (normally base or motor)",
    )
    p.add_argument(
        "--support-body",
        choices=scene.PARTS,
        help="Supported body; motor for horn/yoke fitting, base for other tasks",
    )
    p.add_argument(
        "--approach-seconds",
        type=float,
        help="Smooth approach duration; default2s for yoke,1.5s otherwise, followed by1.4s hold",
    )
    p.add_argument(
        "--entry-lift",
        type=float,
        default=0.0,
        help="Raise the approach in the mating frame, then lower after horizontal insertion",
    )
    p.add_argument(
        "--part-friction",
        type=float,
        help="Diagnostic regular-part coefficient; also clears torsional/rolling friction",
    )
    p.add_argument(
        "--target-correction",
        type=float,
        nargs=3,
        help="Guide offset in the reference CAD frame; yoke defaults50micrometres down; reward targets stay fixed",
    )
    p.add_argument(
        "--motor-interface",
        choices=("both", "volume", "surface"),
        default="both",
        help="Diagnostic contact ablation; canonical source unchanged",
    )
    p.add_argument("--noslip", type=int, help="Diagnostic NoSlip iteration count")
    p.add_argument(
        "--targets-only", action="store_true", help="Audit all five completed targets instead of sliding"
    )
    p.add_argument(
        "--wall", action="store_true", help="Also press the motor against both closed rails at each timestep"
    )
    p.add_argument("--thread", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split("+")
    steps = [float(x) for x in a.timesteps.split(",")]
    if any(t not in ("motor", "horn", "retainer", "yoke") for t in tasks) or any(
        not 0 < t <= 0.0005 for t in steps
    ):
        raise ValueError("Invalid task or timestep")
    report = dict(
        environment=environment_identity(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=__doc__,
        cases=[],
    )
    if a.targets_only:
        report["target_cases"] = []
        for task in scene.ASSEMBLY_TASKS:
            row = audit_target(task)
            report["target_cases"].append(row)
            (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["summary"] = dict(
            cases=len(report["target_cases"]), passed=sum(x["passed"] for x in report["target_cases"])
        )
        (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(0 if all(x["passed"] for x in report["target_cases"]) else 1)
    for task in tasks:
        for dt in steps:
            row = slide(
                task,
                dt,
                a.force_cap,
                a.extraction,
                a.part_friction,
                a.target_correction,
                a.motor_interface,
                a.noslip,
                a.feed_force,
                a.rotation_stiffness,
                a.position_stiffness,
                a.receiver,
                a.support_body,
                a.entry_lift,
                a.approach_seconds,
            )
            report["cases"].append(row)
            (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("samples", "driven_state", "final_state", "unassisted_native_reward")
                    }
                ),
                flush=True,
            )
    if a.wall:
        report["wall_cases"] = [wall_blocking(dt, direction) for dt in steps for direction in (-1, 1)]
    if a.thread:
        from scripts.so101_thread_check import bench

        report["thread"] = {
            str(dt): {
                name: {k: v for k, v in bench(push, dt).items() if k != "trace"}
                for name, push in (("rotation", False), ("axial", True))
            }
            for dt in steps
        }
    checks = [x["passed"] for x in report["cases"] + report.get("wall_cases", [])]
    if a.thread:
        for pair in report["thread"].values():
            rotation = pair["rotation"]
            axial = pair["axial"]
            checks.extend(
                [
                    abs(rotation["turns"]) > 1
                    and abs(rotation["advance_m"] - rotation["expected_advance_m"]) < 0.00005,
                    abs(axial["advance_m"]) < 0.0001,
                ]
            )
    report["summary"] = dict(cases=len(checks), passed=sum(checks))
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
