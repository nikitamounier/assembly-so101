"""Both SO101 grippers holding four distinct faces of one cube."""

from __future__ import annotations

import math

CONFIG = {
    "name": "dual_grasp",
    "label": "Dual grasp",
    "description": (
        "Near the parabolic apex, the left gripper pinches opposite local side faces and the right gripper pinches the local top and bottom: four distinct loaded faces."
    ),
    "cube_center": [0.0, 0.0, 0.065],
    "cube_jitter": [0.004, 0.004, 0.004],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.025,
    "left_mode": "grasp",
    "right_mode": "grasp",
    "left_offset": [0.0, 0.0, 0.0],
    "right_offset": [0.0, 0.0, 0.0],
    "settle_steps": 0,
}


def sample(rng):
    """Materialize bounded transfer noise once, with the supplied seeded RNG."""
    result = dict(CONFIG)
    result["cube_center"] = [
        float(center + rng.uniform(-width, width)) if width else float(center)
        for center, width in zip(CONFIG["cube_center"], CONFIG["cube_jitter"])
    ]
    result["cube_yaw"] = float(CONFIG["cube_yaw"] + rng.uniform(-CONFIG["yaw_jitter"], CONFIG["yaw_jitter"]))
    if not all(math.isfinite(v) for v in result["cube_center"] + [result["cube_yaw"]]):
        raise ValueError("The reset RNG returned a non-finite value")
    result["cube_jitter"] = [0.0, 0.0, 0.0]
    result["yaw_jitter"] = 0.0
    result["left_offset"] = list(CONFIG["left_offset"])
    result["right_offset"] = list(CONFIG["right_offset"])
    return result


def validate_pad_contacts(
    model,
    data,
    cube_geom,
    pad_geoms,
    *,
    expected_faces=None,
    normal_alignment=0.85,
    min_normal_force=0.0,
):
    """Assign loaded native contacts to signed faces in the cube local frame.

    ``pad_geoms`` maps arm names to sequences of physical jaws. Each jaw is
    one geom ID or a sequence of geom IDs. A flat sequence of jaws is also
    supported when arm-specific expectations are unnecessary. ``expected_faces``
    maps arm names to allowed signed ``(axis, sign)`` pairs, e.g.
    ``{"left": {(1, -1), (1, 1)}, "right": {(2, -1), (2, 1)}}``.
    Local faces remain identifiable through arbitrary cube rotations. Each jaw
    needs a distinct allowed face with normal force above ``min_normal_force``.
    No collision geometry, contact pairs, or response is changed.
    """
    from collections.abc import Mapping

    import mujoco
    import numpy as np

    if not 0 < normal_alignment <= 1 or min_normal_force < 0:
        raise ValueError("Invalid normal alignment or minimum normal force")
    entries = (
        [(side, jaw) for side, jaws in pad_geoms.items() for jaw in jaws]
        if isinstance(pad_geoms, Mapping)
        else [(None, jaw) for jaw in pad_geoms]
    )
    pads = tuple(int(jaw) if np.isscalar(jaw) else tuple(int(g) for g in jaw) for _, jaw in entries)
    groups = {pad: (pad,) if isinstance(pad, int) else pad for pad in pads}
    all_geoms = [g for group in groups.values() for g in group]
    if (
        not pads
        or len(groups) != len(pads)
        or any(not group for group in groups.values())
        or len(set(all_geoms)) != len(all_geoms)
        or int(cube_geom) in all_geoms
    ):
        raise ValueError("Expected nonempty, disjoint physical jaw geom groups")
    allowed = {}
    for (side, _), pad in zip(entries, pads):
        if expected_faces is not None:
            if side not in expected_faces:
                raise ValueError(f"Missing expected cube faces for arm {side!r}")
            allowed[pad] = {tuple(face) for face in expected_faces[side]}
            if any(axis not in (0, 1, 2) or sign not in (-1, 1) for axis, sign in allowed[pad]):
                raise ValueError("Faces require axis 0/1/2 and sign -1/+1")
    geom_to_pad = {g: pad for pad, group in groups.items() for g in group}
    candidates = {pad: set() for pad in pads}
    force = np.zeros(6)
    cube_rotation = np.asarray(data.geom_xmat[cube_geom]).reshape(3, 3)
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        other = b if a == cube_geom else a if b == cube_geom else None
        pad = geom_to_pad.get(other)
        if pad not in candidates or contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] <= min_normal_force:
            continue
        # Native normal is directed geom1 -> geom2; obtain cube outward normal.
        normal = cube_rotation.T @ np.asarray(contact.frame[:3])
        if b == cube_geom:
            normal = -normal
        axis = int(np.argmax(np.abs(normal)))
        face = (axis, 1 if normal[axis] >= 0 else -1)
        if abs(normal[axis]) >= normal_alignment and (pad not in allowed or face in allowed[pad]):
            candidates[pad].add(face)

    def assign(index, used, result):
        if index == len(pads):
            return result
        pad = pads[index]
        for face in sorted(candidates[pad] - used):
            found = assign(index + 1, used | {face}, {**result, pad: face})
            if found is not None:
                return found
        return None

    result = assign(0, set(), {})
    if result is None:
        raise ValueError(f"Distinct expected loaded native cube faces missing: {candidates}")
    return result


def validate_four_pad_contacts(model, data, cube_geom, pad_geoms, **kwargs):
    """Validate four physical jaws, supporting all cube axes and arm expectations."""
    from collections.abc import Mapping

    if isinstance(pad_geoms, Mapping):
        pad_geoms = {side: tuple(jaws) for side, jaws in pad_geoms.items()}
        count = sum(map(len, pad_geoms.values()))
    else:
        pad_geoms = tuple(pad_geoms)
        count = len(pad_geoms)
    if count != 4:
        raise ValueError("Expected exactly four physical jaws")
    return validate_pad_contacts(model, data, cube_geom, pad_geoms, **kwargs)
