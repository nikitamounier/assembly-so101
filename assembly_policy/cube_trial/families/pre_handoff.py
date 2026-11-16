"""Left-held cube with the perpendicular receiving gripper just short of contact."""

CONFIG = {
    "name": "pre_handoff",
    "label": "Pre-handoff",
    "description": (
        "Near the parabolic apex, the left gripper holds the side faces while the open right gripper aligns with the cube top and bottom, just before contact."
    ),
    "cube_center": [0.0, 0.0, 0.065],
    "cube_jitter": [0.004, 0.004, 0.004],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.025,
    "left_mode": "grasp",
    "right_mode": "approach",
    "left_offset": [0.0, 0.0, 0.0],
    # Withdraw toward the right base: opening the jaw alone leaves the fixed
    # finger touching the bottom face. Native clearance is checked separately.
    "right_offset": [0.030, 0.0, 0.0],
    "right_offset_jitter": [0.002, 0.001, 0.001],
    "settle_steps": 0,
}
