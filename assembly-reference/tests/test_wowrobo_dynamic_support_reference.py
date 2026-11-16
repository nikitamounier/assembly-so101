"""Support sampling stays continuous around measured moving snapshots."""

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_repaired_resets import RepairedResetSequence


def test_hinge_fit_preserves_measured_gap_and_registered_motor_case():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base_side_cover"><geom size=".01"/></body>
      <body name="upper_arm_module" pos="0 0 .1">
        <freejoint name="upper_arm_module_free"/><geom size=".01"/>
        <body name="target_servo_2_pivot" pos="0 0 .02">
          <joint name="target_servo_2_shaft" axis="0 1 0"/><geom size=".01"/>
        </body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fitter = object.__new__(RepairedResetSequence)
    fitter.model = model
    evidence = {
        "receiver_body": "base_side_cover",
        "far_point_piece_local_m": [0.1, 0, 0],
        "reference_signed_support_offset_m": 0.1,
        "contact_receiver_local_m": [0, 0, 0],
        "support_normal_receiver_local": [0, 0, 1],
    }
    before = data.body("target_servo_2_pivot").xpos.copy()
    fit = fitter._fit_support_hinge(data, "upper_arm_module", evidence, 0.0)
    assert abs(fit["fitted_hinge_rad"]) < 1e-6
    np.testing.assert_allclose(
        data.body("target_servo_2_pivot").xpos, before, atol=1e-12
    )
    # Move the motor1mm along entry. The bracket can articulate on its actual
    # shaft while preserving the measured100mm offset, rather than snapping to0.
    data.joint("upper_arm_module_free").qpos[2] += 0.001
    mujoco.mj_forward(model, data)
    before = data.body("target_servo_2_pivot").xpos.copy()
    fit = fitter._fit_support_hinge(data, "upper_arm_module", evidence, 0.0)
    assert 0.001 < abs(fit["fitted_hinge_rad"]) < 0.12
    np.testing.assert_allclose(
        data.body("target_servo_2_pivot").xpos, before, atol=1e-12
    )
    root = data.body("upper_arm_module")
    far = root.xpos + root.xmat.reshape(3, 3) @ [0.1, 0, 0]
    assert abs(far[2] - 0.1) < 1e-7
    assert fit["witness_distance_m"] < 1e-7
