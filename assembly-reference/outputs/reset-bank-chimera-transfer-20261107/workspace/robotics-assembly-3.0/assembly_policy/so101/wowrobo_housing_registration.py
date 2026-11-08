"""Center servo housings in their existing printed cages without editing meshes.

Measured source planes along the output axis (millimeters): printed inside faces
[-34.1475, -0.7475], housing caps [-33.5975, -0.4975]. The housing is 0.4 mm
off center: a 0.25 mm front interference and 0.55 mm rear gap become two 0.15 mm
clearances. Rotor/printed geometry, the hinge axis line and contact laws remain.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

OFFSET_M = 0.0004
MARKER = "wowrobo_housing_axial_centering_m"
ROOTS = {2: "upper_arm_module", 3: "forearm_module"}


def apply_housing_centering(root, offset_m=OFFSET_M):
    """Apply once to a scene Element; return an explicit registration receipt."""
    if not np.isfinite(offset_m) or offset_m <= 0:
        raise ValueError("Housing centering needs a finite positive axial offset")
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    marker = custom.find(f"numeric[@name='{MARKER}']")
    if marker is not None:
        if float(marker.get("data")) != offset_m:
            raise ValueError("Scene already has a different housing registration")
        return {
            "profile": "centered_housing_v1",
            "offset_m": offset_m,
            "already_applied": True,
        }
    edits = []
    for servo in ROOTS:
        body = root.find(f".//body[@name='target_servo_{servo}_pivot']")
        if body is None:
            raise ValueError(f"Missing servo {servo} housing body")
        joint = body.find(f"joint[@name='target_servo_{servo}_shaft']")
        if joint is None or joint.get("type", "hinge") != "hinge":
            raise ValueError("Housing centering requires the existing output hinge")
        if any(key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Use an explicit body quaternion for housing registration")
        axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
        axis /= np.linalg.norm(axis)
        rotation = np.empty(9)
        quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        mujoco.mju_quat2Mat(rotation, quat)
        old = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        new = old - offset_m * (rotation.reshape(3, 3) @ axis)
        body.set("pos", " ".join(format(float(x), ".17g") for x in new))
        edits.append(
            {
                "servo": servo,
                "body": body.get("name"),
                "old_pos": old.tolist(),
                "new_pos": new.tolist(),
            }
        )
    ET.SubElement(custom, "numeric", name=MARKER, data=format(offset_m, ".17g"))
    return {
        "profile": "centered_housing_v1",
        "offset_m": offset_m,
        "edits": edits,
        "printed_inside_planes_m": [-0.0341475, -0.0007475],
        "source_housing_cap_planes_m": [-0.0335975, -0.0004975],
        "centered_face_clearances_m": [0.00015, 0.00015],
        "joint_axis_and_joint_pos_unchanged": True,
        "source_meshes_and_contact_laws_unchanged": True,
    }


def _marker(model):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, MARKER)
    return None if index < 0 else float(model.numeric_data[model.numeric_adr[index]])


def migrate_qpos(
    before_model, after_model, qpos, *, propagate_final_with_receiver=False
):
    """Move printed roots so the actual motor cases retain their world poses.

    For final insertion stages, also translate the whole incoming final module
    by the upper receiver's shift. The caller must move/re-solve its LEFT hand
    by that reported receiver shift, and RIGHT upper support by the same shift.
    This is initial-pose registration, not an angle or holding constraint.
    """
    old_offset, offset = _marker(before_model), _marker(after_model)
    if offset is None:
        raise ValueError("Destination model lacks the housing registration marker")
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape != (before_model.nq,) or before_model.nq != after_model.nq:
        raise ValueError("Registration migration must preserve the native state layout")
    if old_offset is not None:
        if old_offset != offset:
            raise ValueError("Cannot migrate between different centering offsets")
        return qpos.copy(), {
            "already_registered": True,
            "receiver_shift_world_m": [0.0, 0.0, 0.0],
        }
    before, after = mujoco.MjData(before_model), mujoco.MjData(after_model)
    before.qpos[:] = qpos
    mujoco.mj_kinematics(before_model, before)
    migrated = qpos.copy()
    shifts = {}
    for servo, name in ROOTS.items():
        shifts[servo] = (
            offset * before.xaxis[before_model.joint(f"target_servo_{servo}_shaft").id]
        )
        address = int(after_model.joint(name + "_free").qposadr[0])
        migrated[address : address + 3] += shifts[servo]
    receiver_shift = shifts[2] if propagate_final_with_receiver else np.zeros(3)
    if propagate_final_with_receiver:
        address = int(after_model.joint("forearm_module_free").qposadr[0])
        migrated[address : address + 3] += receiver_shift
    after.qpos[:] = migrated
    mujoco.mj_kinematics(after_model, after)
    errors = {}
    for servo in ROOTS:
        name = f"target_servo_{servo}_pivot"
        old_id, new_id = before_model.body(name).id, after_model.body(name).id
        expected_shift = receiver_shift if servo == 3 else np.zeros(3)
        position = float(
            np.linalg.norm(after.xpos[new_id] - before.xpos[old_id] - expected_shift)
        )
        rotation = float(np.max(np.abs(after.xmat[new_id] - before.xmat[old_id])))
        old_joint, new_joint = (
            before_model.joint(f"target_servo_{servo}_shaft").id,
            after_model.joint(f"target_servo_{servo}_shaft").id,
        )
        anchor = float(
            np.linalg.norm(
                after.xanchor[new_joint] - before.xanchor[old_joint] - expected_shift
            )
        )
        axis = float(np.max(np.abs(after.xaxis[new_joint] - before.xaxis[old_joint])))
        if max(position, rotation, anchor, axis) > 1e-10:
            raise ValueError(
                "Housing registration did not preserve the intended case/axis frames"
            )
        errors[str(servo)] = {
            "case_position_error_m": position,
            "case_rotation_error": rotation,
            "anchor_error_m": anchor,
            "axis_error": axis,
        }
    return migrated, {
        "offset_m": offset,
        "receiver_shift_world_m": receiver_shift.tolist(),
        "printed_root_shifts_world_m": {str(k): v.tolist() for k, v in shifts.items()},
        "frame_errors": errors,
    }
