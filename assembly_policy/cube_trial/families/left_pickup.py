"""A left-hand grasp at the cube's initial tabletop position."""

CONFIG = {
    "name": "left_pickup",
    "label": "Left pickup",
    "description": (
        "The left gripper securely pinches opposite cube side faces at the left endpoint. The right arm stays at home."
    ),
    "cube_center": [-0.125, 0.0, 0.018],
    # Keep the 36 mm cube supported on the table while varying its grasp location.
    "cube_jitter": [0.015, 0.012, 0.0],
    "cube_yaw": 0.0,
    "yaw_jitter": 0.08,
    "left_mode": "grasp",
    "left_pitch": 1.5707963267948966,
    "right_mode": "home",
    "settle_steps": 120,
}
