"""Nearby open-arm proposals and conservative, read-only geometry validation.

This checks complete geometry bounds, including visual meshes whose contact masks
are disabled. Bounds can reject a physically clear pose; they cannot certify a
clearance that the enclosed geometry does not have. Nonadjacent self-link
pairs use convex native geometry distance after the bounds broad phase.
Dynamic retention still
requires the curriculum's production/fine-step replay.
"""

import mujoco
import numpy as np

from .robot import arm_indices, ik
from .scene import JOINTS

READY_OFFSETS = ((0.10, 0.04, 0.13), (0.13, -0.02, 0.15), (0.15, 0.06, 0.16))
OPEN_GRIPPER = 0.65
CLEARANCE_M = 0.001
CORRIDOR_RADIUS_M = 0.025


def _bounds(model, data, geom):
    """World AABB enclosing the whole geometry, not sampled surface points."""
    kind = int(model.geom_type[geom])
    center = data.geom_xpos[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    mesh = int(model.geom_dataid[geom])
    if kind in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF)) and mesh >= 0:
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        points = model.mesh_vert[start : start + count] @ rotation.T + center
        return points.min(axis=0), points.max(axis=0)
    size = model.geom_size[geom]
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = np.full(3, size[0])
    elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        extent = np.abs(rotation[:, 2]) * size[1] + size[0]
    else:
        extent = np.abs(rotation) @ size
    return center - extent, center + extent


def _segment_intersects_box(start, end, lower, upper):
    direction = end - start
    lo, hi = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
        else:
            a, b = sorted(
                ((lower[axis] - start[axis]) / direction[axis], (upper[axis] - start[axis]) / direction[axis])
            )
            lo, hi = max(lo, a), min(hi, b)
            if lo > hi:
                return False
    return True


def validate_ready_arms(
    model,
    data,
    sides,
    *,
    corridor_start=None,
    corridor_end=None,
    clearance_m=CLEARANCE_M,
    corridor_radius_m=CORRIDOR_RADIUS_M,
):
    """Inspect saved state without changing it; no qpos repair or control changes.

    The caller should supply the actual active-part insertion corridor. The
    default reserves the vertical 15 cm corridor above the receiving base.
    Every moving link/finger is tested, including collision-disabled visuals.
    """
    sides = tuple(sides)
    if not sides or any(side not in ("left", "right") for side in sides):
        raise ValueError("Expected one or both SO101 sides")
    if not np.isfinite([clearance_m, corridor_radius_m]).all() or clearance_m < 0 or corridor_radius_m < 0:
        raise ValueError("Clearance and corridor radius must be nonnegative")
    if not np.isfinite(data.qpos).all():
        raise ValueError("Nonfinite robot state")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_forward(model, scratch)
    anchor = scratch.body("base").xpos.copy()
    start = np.asarray(anchor if corridor_start is None else corridor_start, dtype=float)
    end = np.asarray(anchor + [0, 0, 0.15] if corridor_end is None else corridor_end, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
        raise ValueError("Corridor endpoints must be finite xyz vectors")
    names = [model.body(int(b)).name or "" for b in model.geom_bodyid]
    boxes = {
        g: _bounds(model, scratch, g)
        for g in range(model.ngeom)
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    }
    errors, checks = [], 0
    for side in sides:
        qi, _ = arm_indices(model, side)
        sign = np.sign(scratch.body(side + "_base").xpos[0] - anchor[0])
        position = scratch.site(side + "_gripperframe").xpos
        centers = anchor + np.asarray(READY_OFFSETS) * [sign, 1, 1]
        if not np.any(np.all(np.abs(centers - position) <= 0.015, axis=1)):
            errors.append(f"{side}: gripper is outside nearby ready region")
        if abs(float(scratch.qpos[qi[5]]) - OPEN_GRIPPER) > 0.08:
            errors.append(f"{side}: gripper is not open")
        moving = [g for g, name in enumerate(names) if name.startswith(side + "_") and name != side + "_base"]
        obstacles = [g for g, name in enumerate(names) if not name.startswith(side + "_") and g in boxes]
        obstacle_lower = np.asarray([boxes[g][0] for g in obstacles])
        obstacle_upper = np.asarray([boxes[g][1] for g in obstacles])
        for g in moving:
            lower, upper = boxes[g]
            label = model.geom(g).name or f"geom{g}"
            checks += 1
            if lower[2] < clearance_m:
                errors.append(f"{side}: {label} lacks table clearance")
            checks += 1
            radius = corridor_radius_m + clearance_m
            if _segment_intersects_box(start, end, lower - radius, upper + radius):
                errors.append(f"{side}: {label} intersects insertion corridor bounds")
            checks += len(obstacles)
            hits = np.flatnonzero(
                np.all(upper + clearance_m >= obstacle_lower, axis=1)
                & np.all(obstacle_upper + clearance_m >= lower, axis=1)
            )
            for hit in hits:
                other = obstacles[int(hit)]
                errors.append(
                    f"{side}: {label} lacks conservative clearance from "
                    f"{model.geom(other).name or names[other]}"
                )
        # Same-body solids and directly connected bearing/jaw interfaces are
        # designed overlaps. Other links must remain mutually clear, including
        # the stationary base that native parent filtering can omit.
        own = [g for g, name in enumerate(names) if name.startswith(side + "_") and g in boxes]
        for index, g in enumerate(own):
            body = int(model.geom_bodyid[g])
            for other in own[index + 1 :]:
                other_body = int(model.geom_bodyid[other])
                if (
                    body == other_body
                    or int(model.body_parentid[body]) == other_body
                    or int(model.body_parentid[other_body]) == body
                ):
                    continue
                checks += 1
                lower, upper = boxes[g]
                a, b = boxes[other]
                if np.all(upper + clearance_m >= a) and np.all(b + clearance_m >= lower):
                    distance = float(mujoco.mj_geomDistance(model, scratch, g, other, clearance_m, None))
                    if distance >= clearance_m:
                        continue
                    errors.append(
                        f"{side}: nonadjacent self-link clearance failure: "
                        f"{model.body(body).name}/{model.body(other_body).name}"
                    )
    return dict(
        ok=not errors,
        sides=list(sides),
        errors=errors,
        geometry_checks=checks,
        method="complete-geometry-aabb-convex-self-v1",
        clearance_m=float(clearance_m),
        corridor_start_m=start.tolist(),
        corridor_end_m=end.tolist(),
        corridor_radius_m=float(corridor_radius_m),
        dynamic_retention_verified=False,
    )


def prepare_ready_arms(
    model,
    data,
    rng,
    active_side=None,
    *,
    corridor_start=None,
    corridor_end=None,
    variation_m=0.01,
    attempts=24,
):
    """Prepare idle arm(s); commit a proposal only after conservative checks pass.

    Sets ordinary position actuator targets, without gravity compensation. The
    caller applies its existing compensation exactly once after this function.
    Both arms retain their ordinary policy actuators; no weld or support grasp.
    """
    if active_side not in (None, "left", "right"):
        raise ValueError("Unknown active arm")
    if not np.isfinite(variation_m) or not 0 <= variation_m <= 0.01:
        raise ValueError("Ready variation must be within the planned +/-10 mm")
    sides = ("left", "right") if active_side is None else (("right",) if active_side == "left" else ("left",))
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = data.qvel
    scratch.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, scratch)
    anchor = scratch.body("base").xpos.copy()
    targets = {}
    for side in sides:
        qi, vi = arm_indices(model, side)
        sign = np.sign(scratch.body(side + "_base").xpos[0] - anchor[0])
        last = None
        first_offset = int(rng.integers(len(READY_OFFSETS)))
        for attempt in range(attempts):
            offset = np.array(READY_OFFSETS[(first_offset + attempt) % len(READY_OFFSETS)])
            offset[0] *= sign
            target = anchor + offset + rng.uniform(-variation_m, variation_m, 3)
            try:
                q = ik(model, scratch.qpos, side, target)
            except ValueError:
                continue
            scratch.qpos[qi[:5]] = q
            scratch.qpos[qi[5]] = OPEN_GRIPPER
            scratch.qvel[vi] = 0
            mujoco.mj_forward(model, scratch)
            last = validate_ready_arms(
                model, scratch, (side,), corridor_start=corridor_start, corridor_end=corridor_end
            )
            if last["ok"]:
                targets[side] = target.tolist()
                break
        else:
            raise ValueError(f"No clear nearby {side} pose: {last}")
    receipt = validate_ready_arms(
        model, scratch, sides, corridor_start=corridor_start, corridor_end=corridor_end
    )
    if not receipt["ok"]:
        raise ValueError(f"Combined ready-arm geometry invalid: {receipt}")
    for side in sides:
        qi, vi = arm_indices(model, side)
        data.qpos[qi] = scratch.qpos[qi]
        data.qvel[vi] = 0
        for index, joint in enumerate(JOINTS):
            data.ctrl[model.actuator(side + "_" + joint).id] = scratch.qpos[qi[index]]
    mujoco.mj_forward(model, data)
    receipt.update(targets_m=targets, variation_m=float(variation_m), open_gripper_rad=OPEN_GRIPPER)
    return receipt
