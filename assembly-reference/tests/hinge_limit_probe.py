import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from assembly_policy.so101.wowrobo_hinge_limits import install

xml = """<mujoco><option timestep=".0005" integrator="implicitfast" gravity="0 0 0"/><size nuserdata="4"/><worldbody><body><joint name="target_servo_2_shaft" type="hinge" limited="true" range="-.1 .1" damping=".02"/><geom type="box" size=".01 .02 .03" mass="1" contype="0"/><body pos="0 0 .1"><joint name="target_servo_3_shaft" type="hinge" limited="true" range="-.1 .1" damping=".02"/><geom type="box" size=".01 .02 .03" mass="1" contype="0"/></body></body></worldbody></mujoco>"""
# XML joint ranges are degrees by default; use explicit radians.
xml = xml.replace("<option", '<compiler angle="radian"/><option')
m = mujoco.MjModel.from_xml_string(xml)
lib = install(m, Path(sys.argv[1]), offset=0)
original = m.jnt_range.copy()


def check(sign):
    reference = mujoco.MjModel.from_xml_string(xml)
    a = mujoco.MjData(m)
    b = mujoco.MjData(reference)
    delta = np.array([sign * 0.0005, -sign * 0.0003, -sign * 0.0004, sign * 0.0006])
    a.userdata[:] = delta
    reference.jnt_range[:] += delta.reshape(2, 2)
    for d in [a, b]:
        d.qpos[:] = [0.0999, -0.0999]
        d.qvel[:] = [1, -1]
    for _ in range(300):
        for model, data in [(m, a), (reference, b)]:
            data.qfrc_applied[:] = [0.002, -0.003]
            mujoco.mj_step(model, data)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)
    return True


try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(check, [-1, 1]))
    from mujoco import rollout

    initial = np.zeros((2, mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
    initial[:, 1:3] = [0.0999, -0.0999]
    initial[:, 3:5] = [1.0, -1.0]
    deltas = np.array(
        [[0.0005, -0.0003, -0.0004, 0.0006], [-0.0005, 0.0003, 0.0004, -0.0006]]
    )
    control = np.repeat(
        np.concatenate([np.tile([0.002, -0.003], (2, 1)), deltas], axis=1)[:, None, :],
        300,
        axis=1,
    )
    references = [mujoco.MjModel.from_xml_string(xml) for _ in range(2)]
    for ref, delta in zip(references, deltas):
        ref.jnt_range[:] += delta.reshape(2, 2)
    with rollout.Rollout(nthread=2) as engine:
        actual, _ = engine.rollout(
            m,
            [mujoco.MjData(m) for _ in range(2)],
            initial,
            control,
            control_spec=mujoco.mjtState.mjSTATE_QFRC_APPLIED
            | mujoco.mjtState.mjSTATE_USERDATA,
        )
        expected, _ = engine.rollout(
            references,
            [mujoco.MjData(references[0]) for _ in range(2)],
            initial,
            control,
            control_spec=mujoco.mjtState.mjSTATE_QFRC_APPLIED
            | mujoco.mjtState.mjSTATE_USERDATA,
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(m.jnt_range, original)
    print(
        "PASS:600steps,two simultaneous noisy worlds match separate native range models exactly; shared range unchanged."
    )
finally:
    lib.wowrobo_remove_hinge_limits()
