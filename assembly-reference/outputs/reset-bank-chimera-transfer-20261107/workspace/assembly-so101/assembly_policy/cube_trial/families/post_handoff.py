"""Cube received by the right gripper while the open left gripper retreats."""

from __future__ import annotations

import math

CONFIG = {
    "name": "post_handoff",
    "label": "Post handoff",
    "description": "Just after the handoff apex, the right gripper holds the cube by its top and bottom while the left gripper clears the cube.",
    "cube_center": [0.0, 0.0, 0.065],
    "cube_jitter": [0.004, 0.004, 0.004],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.025,
    "left_mode": "retreat",
    "right_mode": "grasp",
    "left_offset": [-0.035, 0.0, 0.015],
    "right_offset": [0.0, 0.0, 0.0],
    "settle_steps": 0,
}


def sample(rng):
    """Materialize bounded shared cube/arm noise without perturbing the grasp."""
    result = dict(CONFIG)
    result["cube_center"] = [
        float(center + rng.uniform(-width, width))
        for center, width in zip(CONFIG["cube_center"], CONFIG["cube_jitter"])
    ]
    result["cube_yaw"] = float(CONFIG["cube_yaw"] + rng.uniform(-CONFIG["yaw_jitter"], CONFIG["yaw_jitter"]))
    if not all(math.isfinite(value) for value in result["cube_center"] + [result["cube_yaw"]]):
        raise ValueError("The reset RNG returned a non-finite value")
    result["cube_jitter"] = [0.0, 0.0, 0.0]
    result["yaw_jitter"] = 0.0
    result["left_offset"] = list(CONFIG["left_offset"])
    result["right_offset"] = list(CONFIG["right_offset"])
    return result
