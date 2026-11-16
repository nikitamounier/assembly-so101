"""Generate passive-articulation starts and independently test unchanged holds.

The settling state is retained, including its real velocity and contact warmstart.
No hinge target, external holding force, velocity clearing or helper weld is used.
Equilibrium is a separate requirement from task/reward admission.
"""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np

from .wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS, PARTS, physical_split_step


def rotation_degrees(a, b):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def pose_snapshot(model, data):
    return {
        name: (
            data.xpos[model.body(name).id].copy(),
            data.xmat[model.body(name).id].reshape(3, 3).copy(),
        )
        for name in (
            *PARTS,
            "target_servo_2_pivot",
            "target_servo_3_pivot",
            "target_servo_4_pivot",
        )
    }


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, result, spec)
    return result


def gravity_preload_controls(model, data):
    """Balance gravity through existing, hardware-limited position commands."""
    mujoco.mj_forward(model, data)
    old = data.ctrl.copy()
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Gravity preload requires scalar joint position servos")
        kp = float(model.actuator_gainprm[a, 0])
        gear = float(model.actuator_gear[a, 0])
        if kp <= 0 or gear != 1:
            raise ValueError("Unsupported actuator transmission for gravity preload")
        data.ctrl[a] += data.qfrc_bias[int(model.jnt_dofadr[joint])] / kp
        if model.actuator_ctrllimited[a]:
            data.ctrl[a] = np.clip(data.ctrl[a], *model.actuator_ctrlrange[a])
    return (data.ctrl - old).tolist()


def settle_and_hold(
    validator,
    stage,
    source,
    *,
    settle_seconds=2.0,
    hold_seconds=0.5,
    close_jaws=True,
    gravity_preload=False,
    friction_multipliers=None,
    pickup_requires_hold=True,
    minimum_only=False,
):
    """Close actual jaws, settle freely, then replay precisely the saved controls.

    Completed hardware attachments retain their original physical hinge DOFs.
    Every object is measured during the hold, including loose future components.
    A part falling out of a grasp and coming to rest cannot pass the grasp gate.
    """
    if settle_seconds <= 0 or hold_seconds <= 0:
        raise ValueError("Both settling and hold windows must be positive")
    m = validator.model
    d = mujoco.MjData(m)
    d.qpos[:] = source.qpos
    d.qvel[:] = source.qvel
    d.ctrl[:] = source.ctrl
    number = int(stage.split("_", 1)[0])
    for name, _, _, _, first in ASSEMBLY_ATTACHMENTS:
        d.eq_active[m.equality(name).id] = number >= first
    mujoco.mj_forward(m, d)
    control_preload = (
        gravity_preload_controls(m, d) if gravity_preload else [0.0] * m.nu
    )
    authored = pose_snapshot(m, d)
    pad_loads = (
        validator._loaded_pads_source_extent
        if minimum_only and hasattr(validator, "_loaded_pads_source_extent")
        else validator._loaded_pads
    )
    pairs = validator._grip_pairs(stage)
    if pickup_requires_hold is False:
        active = validator.definitions[stage].get("active_arm")
        pairs = [pair for pair in pairs if pair[0] != active]
    jaws = {}
    for side, _ in pairs:
        a = int(
            np.flatnonzero(m.actuator_trnid[:, 0] == m.joint(side + "_gripper").id)[0]
        )
        jaws[side] = (a, float(d.ctrl[a]))
    friction = {"finger_part": 1.0, "part_table": 1.0, "part_part": 1.0}
    if friction_multipliers is not None:
        if set(friction_multipliers) != set(friction) or not all(
            np.isfinite(value) and value > 0 for value in friction_multipliers.values()
        ):
            raise ValueError("Expected all three finite positive material multipliers")
        friction.update(friction_multipliers)
    trace = []
    for step in range(round(settle_seconds / m.opt.timestep)):
        if close_jaws:
            loads = pad_loads(d, pairs)
            for side, (a, initial) in jaws.items():
                if min(loads[side].values()) < 0.10:
                    d.ctrl[a] = max(
                        float(m.actuator_ctrlrange[a, 0]),
                        initial - 0.12,
                        d.ctrl[a] - 0.4 * m.opt.timestep,
                    )
        physical_split_step(m, d, friction)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError(f"{stage}: nonfinite settling")
        if step % max(1, round(0.1 / m.opt.timestep)) == 0:
            trace.append(
                {
                    "time": float(d.time),
                    "max_abs_velocity": float(np.max(np.abs(d.qvel))),
                }
            )
    # This is the candidate reset. Do not clear velocities or change the controls.
    saved = integration_state(m, d)
    ctrl = d.ctrl.copy()
    hold_start = pose_snapshot(m, d)
    initial_boundary = validator._native_audit(d)
    grip_reference = {}
    for side, target in pairs:
        if target == "upper_arm_module":
            case_load = pad_loads(d, [(side, "target_servo_2_pivot")])[side]
            if min(case_load.values()) >= 0.01:
                target = "target_servo_2_pivot"
        site = m.site(side + "_gripperframe").id
        body = m.body(target).id
        frame = d.site_xmat[site].reshape(3, 3)
        grip_reference[side] = (
            target,
            frame.T @ (d.xpos[body] - d.site_xpos[site]),
            frame.T @ d.xmat[body].reshape(3, 3),
        )
    hinge_start = np.array(
        [d.joint(f"target_servo_{i}_shaft").qpos[0] for i in range(1, 5)]
    )
    displacement = {name: 0.0 for name in hold_start}
    rotation = {name: 0.0 for name in hold_start}
    loads_min = {
        side: {"fixed": float("inf"), "moving": float("inf")} for side, _ in pairs
    }
    max_velocity = 0.0
    hinge_motion = np.zeros(4)
    bilateral_samples = {side: 0 for side, _ in pairs}
    grasp_motion = {side: 0.0 for side, _ in pairs}
    grasp_rotation = {side: 0.0 for side, _ in pairs}
    boundary_failures = (
        [float(d.time)] if initial_boundary["candidate_rejected"] else []
    )
    hold_steps = max(1, round(hold_seconds / m.opt.timestep))
    for step in range(hold_steps):
        physical_split_step(m, d, friction)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError(f"{stage}: nonfinite hold")
        for name, (pos, rot) in pose_snapshot(m, d).items():
            displacement[name] = max(
                displacement[name], float(np.linalg.norm(pos - hold_start[name][0]))
            )
            rotation[name] = max(
                rotation[name], rotation_degrees(rot, hold_start[name][1])
            )
        angles = np.array(
            [d.joint(f"target_servo_{i}_shaft").qpos[0] for i in range(1, 5)]
        )
        hinge_motion = np.maximum(hinge_motion, np.abs(angles - hinge_start))
        for side, loads in pad_loads(d, pairs).items():
            bilateral_samples[side] += min(loads.values()) >= 0.01
            for pad, value in loads.items():
                loads_min[side][pad] = min(loads_min[side][pad], value)
        for side, (target, local_position, local_rotation) in grip_reference.items():
            site = m.site(side + "_gripperframe").id
            body = m.body(target).id
            frame = d.site_xmat[site].reshape(3, 3)
            grasp_motion[side] = max(
                grasp_motion[side],
                float(
                    np.linalg.norm(
                        frame.T @ (d.xpos[body] - d.site_xpos[site]) - local_position
                    )
                ),
            )
            grasp_rotation[side] = max(
                grasp_rotation[side],
                rotation_degrees(frame.T @ d.xmat[body].reshape(3, 3), local_rotation),
            )
        if (
            minimum_only
            and (step + 1) % max(1, round(0.05 / m.opt.timestep)) == 0
            and validator._native_audit(d)["candidate_rejected"]
        ):
            boundary_failures.append(float(d.time))
        max_velocity = max(max_velocity, float(np.max(np.abs(d.qvel))))
    controls_unchanged = np.array_equal(ctrl, d.ctrl)
    boundary = validator._native_audit(d)
    definition = validator.definitions[stage]
    task = definition["task"]
    anchor = {
        "upper_arm_module": "target_servo_2_pivot",
        "forearm_module": "target_servo_3_pivot",
    }.get(task, task)
    anchor_motion = (
        float(np.linalg.norm(hold_start[anchor][0] - authored[anchor][0]))
        if anchor in authored
        else 0.0
    )
    anchor_rotation = (
        rotation_degrees(hold_start[anchor][1], authored[anchor][1])
        if anchor in authored
        else 0.0
    )
    minimum_reasons = []
    if any(value > 0.005 for value in grasp_motion.values()) or any(
        value > 10.0 for value in grasp_rotation.values()
    ):
        minimum_reasons.append("held_part_escaped_grip")
    if any(count / hold_steps < 0.8 for count in bilateral_samples.values()):
        minimum_reasons.append("held_part_not_captured_by_both_pads")
    if boundary_failures or boundary["candidate_rejected"]:
        minimum_reasons.append("unintended_material_overlap")
    if not controls_unchanged:
        minimum_reasons.append("hold_controls_changed")
    reasons = [] if not minimum_only else minimum_reasons.copy()
    if not minimum_only and max(displacement.values()) > 0.00025:
        reasons.append("part_moves_during_hold")
    if not minimum_only and (
        max(rotation.values()) > 0.25 or np.max(np.rad2deg(hinge_motion)) > 0.25
    ):
        reasons.append("bracket_or_part_rotates_during_hold")
    if not minimum_only and max_velocity > 0.05:
        reasons.append("nonzero_hold_velocity")
    if not minimum_only and any(
        min(loads.values()) < 0.01 for loads in loads_min.values()
    ):
        reasons.append("unloaded_finger_pad")
    # A support handoff has no active incoming grasp. Future loose modules
    # must settle freely; their authored airborne shape is not a stage target.
    stage_anchor_required = definition.get("active_arm") is not None
    if (
        not minimum_only
        and stage_anchor_required
        and (anchor_motion > 0.01 or anchor_rotation > 5.0)
    ):
        reasons.append("authored_stage_not_preserved")
    if not minimum_only and boundary["candidate_rejected"]:
        reasons.append("native_boundaries")
    if not controls_unchanged:
        reasons.append("hold_controls_changed")
    report = {
        "stage": stage,
        "equilibrium_pass": (not reasons) if not minimum_only else None,
        "minimum_validity_pass": not minimum_reasons,
        "minimum_validity_rejection_reasons": minimum_reasons,
        "valid_for_initial_reset_bank": not minimum_reasons,
        "stationarity_required": not minimum_only,
        "finger_contact_region": "original_source_mesh_extent"
        if minimum_only
        else "legacy_pad_window",
        "grasp_translation_drift_m": grasp_motion,
        "grasp_rotation_drift_degrees": grasp_rotation,
        "grip_anchor_by_side": {side: row[0] for side, row in grip_reference.items()},
        "bilateral_contact_fraction": {
            side: count / hold_steps for side, count in bilateral_samples.items()
        },
        "boundary_failure_times": boundary_failures,
        "rejection_reasons": reasons,
        "settle_seconds": settle_seconds,
        "hold_seconds": hold_seconds,
        "part_displacement_m": displacement,
        "part_rotation_degrees": rotation,
        "hinge_motion_degrees": np.rad2deg(hinge_motion).tolist(),
        "settled_hinge_degrees": np.rad2deg(hinge_start).tolist(),
        "maximum_abs_hold_velocity": max_velocity,
        "minimum_pad_normal_force_N": loads_min,
        "active_feature_settling_translation_m": anchor_motion,
        "active_feature_settling_rotation_degrees": anchor_rotation,
        "active_feature_stage_anchor_required": stage_anchor_required,
        "controls_unchanged_during_hold": controls_unchanged,
        "gravity_preload_control_offset_rad": control_preload,
        "friction_multipliers": friction,
        "velocity_cleared_after_settling": False,
        "prescribed_bracket_angle": False,
        "external_forces_used": False,
        "boundary": boundary,
        "settling_trace": trace,
        "state_sha256": hashlib.sha256(saved.tobytes()).hexdigest(),
        "control_sha256": hashlib.sha256(ctrl.tobytes()).hexdigest(),
        "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "training_eligible": bool(minimum_only and not minimum_reasons),
        "qualification": "Minimum initial-reset validity; movement is allowed"
        if minimum_only
        else "Equilibrium check only; task/reward admission remains separate",
    }
    return saved, ctrl, report, integration_state(m, d)
