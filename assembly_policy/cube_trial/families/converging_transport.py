"""A left-held cube and the receiving right arm converge toward handoff."""

from __future__ import annotations

import math

CONFIG = {
    "name": "converging_transport",
    "label": "Converging transport",
    "description": (
        "The left arm carries the cube up the parabolic path toward the handoff apex while the right arm approaches to grasp the top and bottom faces."
    ),
    "cube_center": [-0.0425, 0.0, 0.055],
    "cube_jitter": [0.0, 0.004, 0.0],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.10,
    "left_mode": "grasp",
    "right_mode": "approach",
    "left_offset": [0.0, 0.0, 0.0],
    "right_offset": [0.04, 0.01, 0.015],
    "progress": 0.5,
    "progress_range": [0.0, 1.0],
    "settle_steps": 0,
}


def sample(rng):
    """Materialize one correlated transport state using a seeded uniform RNG.

    The scene's validated transport corridor spans x=-.07 to -.015 m and
    z=.045 to .065 m. Offsets are world-frame meters. Clearing the jitter
    fields prevents the scene from perturbing the sampled state a second time.
    """
    result = dict(CONFIG)
    progress = float(rng.uniform(*CONFIG["progress_range"]))
    result["progress"] = progress
    result["progress_range"] = list(CONFIG["progress_range"])
    result["cube_center"] = [
        -0.07 + 0.055 * progress,
        float(rng.uniform(-0.004, 0.004)),
        0.045 + 0.020 * progress,
    ]
    result["cube_yaw"] = float(rng.uniform(-0.10, 0.10))
    result["left_offset"] = [0.0, 0.0, 0.0]
    # Keep a positive receiving-jaw gap throughout this prehandoff family.
    remaining = 1.0 - progress
    result["right_offset"] = [0.025 + 0.035 * remaining, 0.005, 0.010 + 0.010 * remaining]
    result["cube_jitter"] = [0.0, 0.0, 0.0]
    result["yaw_jitter"] = 0.0
    values = result["cube_center"] + result["right_offset"] + [progress, result["cube_yaw"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The reset RNG returned a non-finite value")
    return result
