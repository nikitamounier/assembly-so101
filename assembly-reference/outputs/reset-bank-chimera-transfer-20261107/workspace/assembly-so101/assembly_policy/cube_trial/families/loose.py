"""Loose-cube trials with both SO101 arms at their fixed home poses."""

from __future__ import annotations

import math

CONFIG = {
    "name": "loose",
    "label": "Loose cube",
    "description": (
        "Both arms start at home. The cube rests at the noisy left endpoint of the parabolic path, away from the left base; the circle marks the right endpoint."
    ),
    "cube_center": [-0.125, 0.0, 0.018],
    "cube_jitter": [0.015, 0.025, 0.0],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.35,
    "left_mode": "home",
    "right_mode": "home",
    "left_offset": [0.0, 0.0, 0.0],
    "right_offset": [0.0, 0.0, 0.0],
    "gripper_open": 0.8,
    "settle_steps": 0,
}


def sample(rng):
    """Return a sampled core configuration using a seeded RNG's uniform API.

    The core may consume CONFIG directly and sample its jitter fields itself.
    This helper instead materializes one reset, clearing the jitter fields so
    that a caller passing the result to the core does not apply noise twice.
    Arm poses and cube height stay fixed; all sampled values are finite.
    """
    result = dict(CONFIG)
    result["cube_center"] = [
        float(center + rng.uniform(-width, width)) if width else float(center)
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
