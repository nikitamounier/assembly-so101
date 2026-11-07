"""Explicit coherent prerequisite frames for isolated insertion-fix candidates.

Independent step alignments were authored before their predecessors were fitted.
This adapter carries later steps with the fitted receivers. It never selects CAD,
changes a production bank, or claims an engineering target is a settled state.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_cover_fit import registered_pose
from assembly_policy.so101.wowrobo_pin_registration import apply_pin_registration

ROOT = Path(__file__).resolve().parents[2]
ALIGNMENTS = ROOT / "outputs/linear-insertion-aligned-20261107"


def pose(data, name):
    matrix = np.eye(4)
    matrix[:3, :3] = data.body(name).xmat.reshape(3, 3)
    matrix[:3, 3] = data.body(name).xpos
    return matrix


def set_root(model, data, body, matrix):
    if int(model.body_parentid[model.body(body).id]) != 0:
        raise ValueError("Expected a world-parented incoming free root")
    address = int(model.joint(body + "_free").qposadr[0])
    data.qpos[address : address + 3] = matrix[:3, 3]
    mujoco.mju_mat2Quat(data.qpos[address + 3 : address + 7], matrix[:3, :3].ravel())
    mujoco.mj_forward(model, data)


def completed_frames(
    model, qpos, step, *, pin_registration=None, upper_reference_qpos=None
):
    """Return a full qpos with prerequisites fitted and the incoming path carried.

    For final insertion, use the successful upper step's FULL51 physical endpoint
    as upper_reference_qpos. Without it, the frozen aligned engineering target
    is used and explicitly reported as provisional. Its shaft angle replaces the
    old photo angle: retaining that angle while fitting the case caused >2mm
    printed-part collisions. Helper-arm coordinates are preserved bit for bit.
    """
    if step not in {"cover", "connector", "pin", "upper", "final"}:
        raise ValueError(step)
    values = np.asarray(qpos, dtype=float)
    if values.shape != (model.nq,) or model.nq != 51:
        raise ValueError("This contract requires the complete 51-coordinate model")
    data = mujoco.MjData(model)
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    old_upper = pose(data, "upper_arm_module")
    old_forearm = pose(data, "forearm_module")
    reference = mujoco.MjData(model)
    connector_alignment = json.loads(
        (ALIGNMENTS / "connector/alignment.json").read_text()
    )
    reference.qpos[:] = connector_alignment["reference_qpos"]
    mujoco.mj_forward(model, reference)
    correction_local = reference.body("base_servo").xmat.reshape(3, 3).T @ np.asarray(
        connector_alignment["correction_world_m"]
    )
    report = {
        "step": step,
        "helper_coordinates_unchanged": True,
        "physical_endpoint_used_for_installed_upper": upper_reference_qpos is not None,
    }
    if step != "cover":
        receiver = data.body("base_servo")
        position, quaternion, _ = registered_pose(receiver.xpos, receiver.xquat)
        address = int(model.joint("base_side_cover_free").qposadr[0])
        data.qpos[address : address + 3] = position
        data.qpos[address + 3 : address + 7] = quaternion
        mujoco.mj_forward(model, data)
    if step in {"pin", "upper", "final"}:
        correction = data.body("base_servo").xmat.reshape(3, 3) @ correction_local
        connector = pose(data, "shoulder_connector")
        connector[:3, 3] += correction
        set_root(model, data, "shoulder_connector", connector)
        report["connector_centering_world_m"] = correction.tolist()
        if step == "upper":
            upper = old_upper.copy()
            upper[:3, 3] += correction
            set_root(model, data, "upper_arm_module", upper)
        if step == "final":
            if upper_reference_qpos is None:
                upper_reference_qpos = json.loads(
                    (ALIGNMENTS / "upper/alignment.json").read_text()
                )["reference_qpos"]
                report["installed_upper_scope"] = (
                    "Engineering fit target; awaiting physical endpoint"
                )
            else:
                report["installed_upper_scope"] = "Caller-supplied physical endpoint"
            reference.qpos[:] = upper_reference_qpos
            mujoco.mj_forward(model, reference)
            relative = np.linalg.inv(pose(reference, "shoulder_connector")) @ pose(
                reference, "upper_arm_module"
            )
            target = pose(data, "shoulder_connector") @ relative
            hinge = int(model.joint("target_servo_2_shaft").qposadr[0])
            data.qpos[hinge] = reference.qpos[hinge]
            set_root(model, data, "upper_arm_module", target)
            set_root(
                model,
                data,
                "forearm_module",
                target @ np.linalg.inv(old_upper) @ old_forearm,
            )
            report["installed_upper_hinge_rad"] = float(data.qpos[hinge])
    if step in {"upper", "final"}:
        if pin_registration is None:
            raise ValueError(
                "Supply the pin registration measured with the centered connector"
            )
        apply_pin_registration(model, data, pin_registration, "17_installed")
        receiver_r = data.body(pin_registration["receiver_body"]).xmat.reshape(3, 3)
        pin_r = receiver_r @ np.asarray(
            pin_registration["receiver_local_rotation"]
        ).reshape(3, 3)
        address = int(model.joint("assembly_pin_free").qposadr[0])
        data.qpos[address : address + 3] += pin_r[:, 2] * pin_registration["head_gap_m"]
        mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos[:13], values[:13]):
        raise AssertionError("Prerequisite fitting changed helper arms or shaft1")
    return data.qpos.copy(), report


def coherent_step_alignment(
    model, alignment, step, *, pin_registration=None, upper_reference_qpos=None
):
    """Adapt both endpoints together; the axial trajectory remains force driven."""
    result = copy.deepcopy(alignment)
    reports = {}
    for key in ("start_qpos", "reference_qpos"):
        qpos, report = completed_frames(
            model,
            alignment[key],
            step,
            pin_registration=pin_registration,
            upper_reference_qpos=upper_reference_qpos,
        )
        if step == "pin":
            if pin_registration is None:
                raise ValueError("Supply the centered-connector pin registration")
            data = mujoco.MjData(model)
            data.qpos[:] = qpos
            apply_pin_registration(
                model,
                data,
                pin_registration,
                "13_pin_aligned" if key == "start_qpos" else "17_installed",
            )
            if key == "reference_qpos":
                receiver_r = data.body(pin_registration["receiver_body"]).xmat.reshape(
                    3, 3
                )
                pin_r = receiver_r @ np.asarray(
                    pin_registration["receiver_local_rotation"]
                ).reshape(3, 3)
                address = int(model.joint("assembly_pin_free").qposadr[0])
                data.qpos[address : address + 3] += (
                    pin_r[:, 2] * pin_registration["head_gap_m"]
                )
            qpos = data.qpos.copy()
        result[key] = qpos.tolist()
        reports[key] = report
    if step == "pin":
        result["authored_axial_travel_m"] = 0.032 + pin_registration["head_gap_m"]
        if "reference_axial_travel_m" in result:
            result["reference_axial_travel_m"] = result["authored_axial_travel_m"]
    if step == "final":
        before = mujoco.MjData(model)
        after = mujoco.MjData(model)
        before.qpos[:] = alignment["start_qpos"]
        after.qpos[:] = result["start_qpos"]
        mujoco.mj_forward(model, before)
        mujoco.mj_forward(model, after)
        rotation = (
            pose(after, "upper_arm_module")[:3, :3]
            @ pose(before, "upper_arm_module")[:3, :3].T
        )
        result["axis_world"] = (rotation @ np.asarray(alignment["axis_world"])).tolist()
    piece = {
        "cover": "base_side_cover",
        "connector": "shoulder_connector",
        "pin": "assembly_pin",
        "upper": "upper_arm_module",
        "final": "forearm_module",
    }[step]
    address = int(model.joint(piece + "_free").qposadr[0])
    result["aligned_root_qpos"] = result["start_qpos"][address : address + 7]
    result["final_root_qpos"] = result["reference_qpos"][address : address + 7]
    if "feature_frame" in result:
        result["feature_frame_scope"] = (
            "Original feature extraction before prerequisite fitting; world poses "
            "are superseded by coherent_feature_poses. Receiver-local geometry is unchanged."
        )
    result["coherent_feature_poses"] = {}
    for key in ("start_qpos", "reference_qpos"):
        data = mujoco.MjData(model)
        data.qpos[:] = result[key]
        mujoco.mj_forward(model, data)
        result["coherent_feature_poses"][key] = {
            body: pose(data, body).tolist()
            for body in (
                piece,
                result.get("mating_body", piece),
                result.get(
                    "receiver_body",
                    {
                        "cover": "base_servo",
                        "connector": "base_servo",
                        "pin": "shoulder_connector",
                        "upper": "shoulder_connector",
                        "final": "upper_arm_module",
                    }[step],
                ),
            )
        }
    result["coherent_prerequisites"] = reports
    result["scope"] = (
        "Isolated coherent fit candidate; source selection and physical qualification are separate"
    )
    return result
