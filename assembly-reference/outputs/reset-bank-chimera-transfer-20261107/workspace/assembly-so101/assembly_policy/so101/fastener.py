"""Nominal M3 fastener supplement to Hiwonder's incomplete assembly CAD.

Hiwonder's kit manual specifies an M3 x 5 x 7 washer-head screw. Interpreted
as 3 mm diameter, 5 mm under-head length, 7 mm flange. Coarse 0.5 mm pitch
and 60 um radial simulation clearance are engineering assumptions, not measurements.
Uses AssembleNIST's existing metric helical SDF implementation.
"""

import mujoco
import numpy as np

PITCH = 0.0005
LENGTH = 0.005
CLEARANCE = 0.00006
AXIS = np.array([0.00000172, -0.007227, 0.06409296])


def target_adjustments():
    from .workcell import configuration

    return configuration()["mating_targets"]


def target_rotations(meta):
    result = {name: np.eye(3) for name in meta}
    for name, adjustment in target_adjustments().items():
        angle = np.deg2rad(adjustment.get("shaft_yaw_deg", 0.0))
        c, s = np.cos(angle), np.sin(angle)
        result[name] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return result


def origins(meta):
    pos = {n: np.asarray(p["origin"]).copy() for n, p in meta.items()}
    rotations = target_rotations(meta)
    for name, adjustment in target_adjustments().items():
        pos[name] = AXIS + rotations[name] @ (pos[name] - AXIS)
        pos[name][2] += adjustment["lift_m"]
    top = pos["horn"][2] + meta["horn"]["bounds"][1][2]
    pos["screw"] = np.array([AXIS[0], AXIS[1], top])  # under-head bearing plane
    return pos


def bearing_pose(model, data, meta, preload_turns=0.25):
    """Initialize a coaxial screw at the actual horn plane with matching helix phase.

    A quarter turn initializes elastic thread/head preload, replacing the old
    absolute270-degree guess. This never controls a live screw.
    Both pinned analytic plugins store pitch first and phase ninth.
    """

    def parameters(name):
        instance = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PLUGIN, name)
        if instance < 0:
            raise ValueError("Missing analytic thread instance")
        start = int(model.plugin_attradr[instance])
        values = bytes(model.plugin_attr[start:]).split(b"\0")[:9]
        return float(values[0]), float(values[8])

    pitch, phase = parameters("female")
    bolt_pitch, bolt_phase = parameters("male")
    if not np.isclose(pitch, bolt_pitch, rtol=0, atol=1e-12):
        raise ValueError("Thread pitches differ")
    motor, horn = data.body("motor"), data.body("horn")
    rotation = motor.xmat.reshape(3, 3)
    axis = rotation[:, 2]
    normal = horn.xmat.reshape(3, 3)[:, 2]
    if normal @ axis < 0.9:
        raise ValueError("Horn plane is not aligned with the motor shaft")
    point = motor.xpos + rotation @ (AXIS - np.asarray(meta["motor"]["origin"]))
    bearing = horn.xpos + normal * meta["horn"]["bounds"][1][2]
    height = float(normal @ (bearing - point) / (normal @ axis))
    yaw = 2 * np.pi * ((height / pitch + bolt_phase - phase) % 1 - preload_turns)
    quaternion = np.empty(4)
    mujoco.mju_mulQuat(quaternion, motor.xquat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    return point + height * axis, quaternion


def attributes():
    rp = 0.0015 - 0.3248 * PITCH
    bolt = dict(
        pitch=PITCH,
        rp=rp,
        root=0.00119,
        crest=0.0015,
        thread_z0=0,
        thread_z1=LENGTH,
        head_flat=0.0035,
        head_top=0,
        phase=0,
    )
    nut = dict(
        pitch=PITCH,
        rp=rp,
        clearance=CLEARANCE,
        minor=0.0012311,
        major=0.00159,
        flat=0.0020,
        z0=0,
        z1=0.006,
        phase=0,
    )
    return bolt, nut
