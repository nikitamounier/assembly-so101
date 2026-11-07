"""Right-arm transport after transfer, with the left arm retreating."""

CONFIG = {
    "name": "right_transport",
    "label": "Right transport",
    "description": (
        "The right arm follows the descending parabolic path and rotates the cube toward a quarter turn, retaining contact with the same original top and bottom faces."
    ),
    "cube_center": [0.055, 0.005, 0.0575],
    "cube_jitter": [0.015, 0.004, 0.0075],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.15,
    "left_mode": "retreat",
    "right_mode": "grasp",
    "left_offset": [-0.05, 0.0, 0.02],
    "right_offset": [0.0, 0.0, 0.0],
    "left_offset_jitter": [0.012, 0.012, 0.008],
    "right_offset_jitter": [0.0, 0.0, 0.002],
    "target_center": [0.13, 0.0, 0.0],
    "target_jitter": [0.004, 0.004, 0.0],
    "gripper_open": 0.8,
    "settle_steps": 0,
}
