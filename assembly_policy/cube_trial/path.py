"""Shared, coherent parabolic cube trajectory for every reset family."""

from __future__ import annotations

import math

CUBE_HALF_EXTENT = 0.018
PROGRESS_RANGES = {
    "loose": (0.0, 0.0),
    "left_pickup": (0.0, 0.0),
    "converging_transport": (0.15, 0.42),
    "pre_handoff": (0.46, 0.49),
    "dual_grasp": (0.49, 0.51),
    "post_handoff": (0.51, 0.54),
    "right_transport": (0.60, 0.86),
    "near_target": (0.975, 0.995),
    "placed_held": (1.0, 1.0),
}


def sample_path(family, rng):
    """Sample geometric reset fields with a seeded ``uniform(low, high)`` RNG.

    Coordinates are world meters, quaternion order is wxyz, and rotation_angle
    is radians about world X. The noisy path endpoint includes a small placement
    offset inside the target; interpolation shares it with intermediate poses.
    No independent cube XYZ or yaw perturbation is added. Arm modes and offsets
    remain the responsibility of the family configuration and scene generator.
    """
    low, high = PROGRESS_RANGES[family]
    progress = float(rng.uniform(low, high)) if high > low else low
    pickup = [-0.10 + float(rng.uniform(-0.004, 0.004)), float(rng.uniform(-0.003, 0.003)), CUBE_HALF_EXTENT]
    target = [0.10 + float(rng.uniform(-0.004, 0.004)), float(rng.uniform(-0.003, 0.003)), 0.0]
    endpoint = [
        target[0] + float(rng.uniform(-0.003, 0.003)),
        target[1] + float(rng.uniform(-0.003, 0.003)),
        CUBE_HALF_EXTENT,
    ]
    arc_height = float(rng.uniform(0.322, 0.327))
    center = [
        (1.0 - progress) * pickup[0] + progress * endpoint[0],
        (1.0 - progress) * pickup[1] + progress * endpoint[1],
        CUBE_HALF_EXTENT + 4.0 * arc_height * progress * (1.0 - progress),
    ]
    # Complete the retained top/bottom-face grip's quarter turn while the cube
    # is high, keeping the receiving arm within its real joint limits. The
    # right-transport family starts partway through this turn; near target is
    # already flat. Smoothstep gives zero angular slope at both endpoints.
    rotation_end = 0.68
    rotation_progress = min(1.0, max(0.0, (progress - 0.54) / (rotation_end - 0.54)))
    rotation_progress = rotation_progress**2 * (3.0 - 2.0 * rotation_progress)
    angle = (math.pi / 2.0) * rotation_progress
    quaternion = [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]
    support_extent = CUBE_HALF_EXTENT * (abs(math.cos(angle)) + abs(math.sin(angle)))
    values = pickup + target + endpoint + center + quaternion + [progress, arc_height, angle]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The reset RNG returned a non-finite value")
    if center[2] + 1e-12 < support_extent:
        raise ValueError("The sampled cube trajectory intersects the table")
    return {
        "path_progress": progress,
        "progress": progress,
        "cube_center": center,
        "cube_quaternion": quaternion,
        "cube_rotation": list(quaternion),
        "cube_yaw": 0.0,
        "cube_jitter": [0.0, 0.0, 0.0],
        "yaw_jitter": 0.0,
        "target_center": target,
        "target_jitter": [0.0, 0.0, 0.0],
        "pickup_center": pickup,
        "endpoint_center": endpoint,
        "arc_height": arc_height,
        "rotation_angle": angle,
        "cube_support_extent": support_extent,
    }
