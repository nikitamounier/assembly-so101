"""A flat cube already inside the target, still held by the right gripper.

The planar noise budget leaves every footprint corner inside the circle even
at the largest displacement and yaw. Success still requires the right hand to
release: geometric placement alone must never count as initial success.
"""

from __future__ import annotations

CONFIG = {
    "name": "placed_held",
    "label": "Placed, right hand holding",
    "description": (
        "The quarter-turned cube rests entirely inside the circle, held only by the right gripper on faces now oriented as sides. The left arm is clear; success requires release."
    ),
    "cube_center": [0.13, 0.0, 0.018],
    "cube_jitter": [0.003, 0.003, 0.0],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.025,
    "left_mode": "home",
    "right_mode": "grasp",
    "left_offset": [0.0, 0.0, 0.0],
    "right_offset": [0.0, 0.0, 0.0],
    "settle_steps": 0,
    "target_center": [0.13, 0.0, 0.0],
    "target_jitter": [0.0, 0.0, 0.0],
    "target_radius": 0.036,
}
