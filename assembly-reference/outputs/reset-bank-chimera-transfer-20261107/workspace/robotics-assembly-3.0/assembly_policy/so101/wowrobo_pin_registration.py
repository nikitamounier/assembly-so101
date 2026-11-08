"""Measure a custom pin's real printed head stop and conservative blind-bore end.

Uses exact native visual triangles/rays; never trusts inside tests on the
non-watertight servo case mesh. Registration is not insertion qualification.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from assembly_policy.so101.wowrobo_boundaries import world_mesh


def measure_pin_registration(model, data, dimensions, *, head_gap_m=0.0001):
    mujoco.mj_forward(model, data)
    joint = model.joint("target_servo_1_shaft").id
    axis = data.xaxis[joint].copy()
    axis /= np.linalg.norm(axis)
    pivot = data.xanchor[joint].copy()
    pin = model.body("assembly_pin").id
    original = data.xmat[pin].reshape(3, 3)
    rotation = (
        Rotation.align_vectors([-axis], [original[:, 2]])[0].as_matrix() @ original
    )
    theta = np.arange(128) * 2 * np.pi / 128
    radii = np.linspace(
        dimensions["shaft_diameter"] * 0.0005, dimensions["grip_diameter"] * 0.0005, 12
    )
    annulus = np.array(
        [[r * np.cos(t), r * np.sin(t), 0] for r in radii for t in theta]
    )
    origins = annulus @ rotation.T + pivot + axis * 0.08
    receiver_mesh = world_mesh(
        model, data, model.geom("shoulder_connector_printed_0").id
    )
    hits, _, _ = receiver_mesh.ray.intersects_location(
        origins, np.tile(-axis, (len(origins), 1)), multiple_hits=False
    )
    if not len(hits):
        raise ValueError("No actual printed stop under pin head")
    projected = (hits - pivot) @ axis
    highest = int(np.argmax(projected))
    stop = float(projected[highest])
    case_mesh = world_mesh(model, data, model.geom("servo_1_case").id)
    axial_hits, _, _ = case_mesh.ray.intersects_location(
        [pivot + axis * 0.05], [-axis], multiple_hits=True
    )
    depths = (axial_hits - pivot) @ axis
    inside = depths[depths < -0.0001]
    if not len(inside):
        raise ValueError("No bounded servo bore end detected")
    blind_end = float(inside.max())
    grip = dimensions["grip_length"] * 0.001
    shaft = dimensions["shaft_length"] * 0.001
    root = pivot + axis * (stop + head_gap_m + grip)
    head = root - axis * grip
    tip = head - axis * shaft
    radial_directions = np.array(
        [np.cos(t) * rotation[:, 0] + np.sin(t) * rotation[:, 1] for t in theta]
    )
    radial_samples = []
    for depth in np.linspace(stop - shaft, min(stop, 0.0), 16):
        centre = pivot + axis * depth
        radial_hits, _, _ = case_mesh.ray.intersects_location(
            np.tile(centre, (len(theta), 1)), radial_directions, multiple_hits=False
        )
        if len(radial_hits):
            radial_samples.extend(np.linalg.norm(radial_hits - centre, axis=1).tolist())
    minimum_radius = min(radial_samples) if radial_samples else None
    receiver = model.body("shoulder_connector").id
    rr = data.xmat[receiver].reshape(3, 3)
    result = {
        "profile": "wowrobo-pin-registration-20261107-v2",
        "receiver_body": "shoulder_connector",
        "root_world_m": root.tolist(),
        "rotation_world": rotation.ravel().tolist(),
        "receiver_local_position_m": ((root - data.xpos[receiver]) @ rr).tolist(),
        "receiver_local_rotation": (rr.T @ rotation).ravel().tolist(),
        "bore_axis_world": axis.tolist(),
        "bore_reference_world_m": pivot.tolist(),
        "head_stop_surface_world_m": hits[highest].tolist(),
        "head_stop_axis_projection_m": stop,
        "blind_end_axis_projection_m": blind_end,
        "stop_to_blind_end_m": stop - blind_end,
        "head_gap_m": head_gap_m,
        "shaft_length_m": shaft,
        "stand_off_by_stage_m": {
            "13_pin_aligned": 0.032,
            "14_pin_half": shaft * 0.5,
            "15_pin_near": 0.002,
        },
        "shaft_tip_world_m": tip.tolist(),
        "seated_axial_clearance_m": stop - blind_end - shaft,
        "initial_axial_clearance_m": stop + head_gap_m - blind_end - shaft,
        "shaft_tip_below_bore_reference_m": shaft - stop - head_gap_m,
        "annulus_rays": len(origins),
        "annulus_hits": len(hits),
        "case_mesh_watertight": bool(case_mesh.is_watertight),
        "sampled_min_bore_radius_m": minimum_radius,
        "sampled_min_radial_clearance_m": minimum_radius
        - dimensions["shaft_diameter"] * 0.0005
        if minimum_radius is not None
        else None,
        "training_eligible": False,
        "qualification": "Exact source-triangle support/bore registration; dynamic insertion and hardware tolerances unqualified",
    }
    if result["seated_axial_clearance_m"] <= 0:
        raise ValueError("Pin shaft bottoms before its head reaches the printed stop")
    return result


def apply_pin_registration(
    model, data, registration, stage_id, *, extra_standoff_m=0.0
):
    """Apply a measured receiver-relative pin anchor; caller must re-solve held arm IK."""
    number = int(stage_id.split("_", 1)[0])
    if number < 13:
        return False
    receiver = model.body(registration["receiver_body"]).id
    mujoco.mj_forward(model, data)
    rr = data.xmat[receiver].reshape(3, 3)
    r = rr @ np.array(registration["receiver_local_rotation"]).reshape(3, 3)
    p = data.xpos[receiver] + rr @ np.array(registration["receiver_local_position_m"])
    stand = {13: 0.032, 14: registration["shaft_length_m"] * 0.5, 15: 0.002}.get(
        number, 0.0
    ) + extra_standoff_m
    p -= r[:, 2] * stand
    address = model.joint("assembly_pin_free").qposadr[0]
    data.qpos[address : address + 3] = p
    mujoco.mju_mat2Quat(data.qpos[address + 3 : address + 7], r.ravel())
    mujoco.mj_forward(model, data)
    return True
