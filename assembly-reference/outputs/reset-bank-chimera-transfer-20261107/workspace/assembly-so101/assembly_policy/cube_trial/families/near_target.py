"""A flat cube held by the right arm just above the circular target."""

from __future__ import annotations

import math

CONFIG = {
    "name": "near_target",
    "label": "Near target",
    "description": (
        "The cube is flat after a quarter turn and near the circle. The right gripper holds the original top and bottom faces, now oriented as side faces; the left arm is clear."
    ),
    "cube_center": [0.13, 0.0, 0.040],
    "cube_jitter": [0.0, 0.0, 0.006],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.0,
    "left_mode": "home",
    "right_mode": "grasp",
    "left_offset": [0.0, 0.0, 0.0],
    "right_offset": [0.0, 0.0, 0.0],
    "gripper_open": 0.8,
    "settle_steps": 0,
    "target_center": [0.13, 0.0, 0.0],
    "target_jitter": [0.006, 0.006, 0.0],
    "target_radius": 0.036,
}


def sample(rng):
    """Materialize bounded target and height noise without rotating the cube."""
    result = dict(CONFIG)
    for field in ("cube", "target"):
        result[f"{field}_center"] = [
            float(center + rng.uniform(-width, width)) if width else float(center)
            for center, width in zip(CONFIG[f"{field}_center"], CONFIG[f"{field}_jitter"])
        ]
        result[f"{field}_jitter"] = [0.0, 0.0, 0.0]
    result["cube_center"][:2] = result["target_center"][:2]
    if not all(math.isfinite(value) for value in result["cube_center"] + result["target_center"]):
        raise ValueError("The reset RNG returned a non-finite value")
    result["left_offset"] = list(CONFIG["left_offset"])
    result["right_offset"] = list(CONFIG["right_offset"])
    return result
