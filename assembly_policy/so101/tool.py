"""Photo-reconstructed HOTO SNAPBLOQ S-A01 with native passive controls.

Published envelope/mass are from HOTO Japan; feature proportions are estimated
from official photographs/manual, not manufacturer CAD. Local +Z points at the
bit, +Y at the controls. Legacy observation/joint names remain compatible:
'driver_sleeve' is now the real control bezel, and 'trigger_saddle' is the round
gear-selector button. The motor retains the existing low-torque setting.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .tool_geometry import (
    capsule_frame,
    capsule_plate,
    lathe,
    ring,
    rocker_half,
    shell,
    stroke_mesh,
    tapered_wing,
)

CASE_HALF_WIDTH = 0.00985
CASE_HALF_LENGTH = 0.08675
NOSE_LENGTH = 0.0082
GRASP_Z = 0.0358
TIP_Z = CASE_HALF_LENGTH + NOSE_LENGTH + 0.020
GRASP_CENTER = np.array([0.0, 0.00025, GRASP_Z])
DEFAULT_POSITION = (0.12, -0.12, 0.01185)
DEFAULT_QUATERNION = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
PUBLISHED_DIMENSIONS_M = (0.1817, 0.0197, 0.0225)
PUBLISHED_MASS_KG = 0.105  # Driver without the installed C4 bit.


def _numbers(values):
    return " ".join(format(float(v), ".10g") for v in np.asarray(values).ravel())


def _add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def add_driver(
    world,
    extension,
    contact,
    *,
    position=DEFAULT_POSITION,
    quaternion=DEFAULT_QUATERNION,
    asset=None,
    with_sleeve=False,
):
    """Add the actual bare-tool reconstruction; the old sleeve proxy is opt-in."""
    if with_sleeve:
        raise ValueError("Only the bare HOTO tool is supported")
    if asset is None:
        raise ValueError("Detailed HOTO model requires the shared MJCF asset element")

    def mesh(name, geometry):
        v, f = geometry
        _add(asset, "mesh", name=name, vertex=_numbers(v), face=_numbers(f))
        return name

    def visual(body, name, geometry, material, **attrs):
        return _add(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=mesh(name + "_mesh", geometry),
            material=material,
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
            **attrs,
        )

    def collision(body, name, kind, **attrs):
        return _add(body, "geom", name=name, type=kind, group=3, mass=0, **attrs)

    for name, rgba, spec, shine in (
        ("hoto_aluminium", ".57 .60 .62 1", 0.85, 0.65),
        ("hoto_ribs", ".43 .46 .48 1", 0.65, 0.5),
        ("hoto_rubber", ".045 .052 .058 1", 0.15, 0.08),
        ("hoto_button", ".075 .083 .09 1", 0.20, 0.18),
        ("hoto_icon", ".52 .55 .56 1", 0.2, 0.1),
        ("hoto_steel", ".52 .56 .59 1", 0.9, 0.8),
        ("hoto_diffuser", ".90 .94 .91 1", 0.2, 0.1),
        ("hoto_amber", "1 .56 .08 1", 0.1, 0.1),
    ):
        _add(asset, "material", name=name, rgba=rgba, specular=spec, shininess=shine)
    tool = _add(world, "body", name="driver", pos=_numbers(position), quat=_numbers(quaternion))
    _add(tool, "freejoint", name="driver_free")
    # The measured total mass is known; its distribution/inertia remain estimates.
    mass = 0.101
    radius = CASE_HALF_WIDTH
    length = 2 * CASE_HALF_LENGTH
    _add(
        tool,
        "inertial",
        pos="0 0 0",
        mass=mass,
        diaginertia=_numbers(
            [
                mass * (0.011**2 / 4 + length**2 / 12),
                mass * (radius**2 / 4 + length**2 / 12),
                mass * (radius**2 + 0.011**2) / 4,
            ]
        ),
    )
    profile = [(-CASE_HALF_LENGTH, 0.975, None), (-CASE_HALF_LENGTH + 0.0003, 1, None)]
    cap_z = [*np.linspace(-0.0295, -0.0242, 7), 0.0299, 0.030, *np.linspace(0.0412, 0.0465, 7)]
    profile.extend((float(z), 1, 0.0094) for z in cap_z)
    profile.extend(((CASE_HALF_LENGTH - 0.0003, 1, None), (CASE_HALF_LENGTH, 0.975, None)))
    visual(tool, "driver_shell", shell(profile, fluted=True, capsule_recess=True), "hoto_aluminium")
    for name, z0, z1, clip in (
        ("tail", -CASE_HALF_LENGTH, -0.0305, None),
        ("control", -0.0305, 0.0475, 0.0094),
        ("front", 0.0475, CASE_HALF_LENGTH, None),
    ):
        collision(
            tool,
            "driver_case_" + name,
            "mesh",
            mesh=mesh("driver_case_" + name + "_mesh", shell([(z0, 1, clip), (z1, 1, clip)], segments=96)),
        )
    ellipse = np.diag([1, 0.011 / radius, 1])
    for name, geometry, material in (
        (
            "driver_tail_seam",
            ring(-CASE_HALF_LENGTH + 0.0013, radius - 0.00005, radius + 0.00001, 0.00016),
            "hoto_rubber",
        ),
        ("driver_led_ring", ring(CASE_HALF_LENGTH + 0.00012, 0.0051, 0.00955, 0.00024), "hoto_diffuser"),
    ):
        v, f = geometry
        visual(tool, name, (v @ ellipse.T, f), material)
    nose = [
        (CASE_HALF_LENGTH, 0.0049),
        (CASE_HALF_LENGTH + 0.0006, 0.005),
        (CASE_HALF_LENGTH + 0.0068, 0.0044),
        (CASE_HALF_LENGTH + NOSE_LENGTH, 0.0042),
    ]
    visual(tool, "driver_nose", lathe(nose), "hoto_rubber")
    collision(
        tool,
        "driver_nose_contact",
        "cylinder",
        size=_numbers([0.005, NOSE_LENGTH / 2]),
        pos=_numbers([0, 0, CASE_HALF_LENGTH + NOSE_LENGTH / 2]),
    )
    visual(
        tool,
        "driver_collet_rim",
        ring(CASE_HALF_LENGTH + NOSE_LENGTH - 0.00035, 0.00245, 0.00425, 0.00045),
        "hoto_steel",
    )
    # USB-C recess, rim and central tongue; these fine surfaces carry no contacts.
    port_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    for name, width, length, thickness, z, material in (
        ("driver_usb_rim", 0.0034, 0.009, 0.00012, -CASE_HALF_LENGTH - 0.00001, "hoto_ribs"),
        ("driver_usb_opening", 0.0027, 0.0082, 0.00014, -CASE_HALF_LENGTH - 0.00007, "hoto_rubber"),
        ("driver_usb_tongue", 0.0006, 0.0055, 0.00008, -CASE_HALF_LENGTH - 0.00015, "hoto_steel"),
    ):
        v, f = capsule_plate(width, length, thickness)
        visual(tool, name, (v @ port_rotation.T + [0, 0, z], f), material)
    # This fixed bezel replaces the prototype's bulky external robot sleeve.
    bezel = _add(tool, "body", name="driver_sleeve")
    _add(bezel, "inertial", pos="0 .009 .0085", mass=".0012", diaginertia="6.0e-7 6.0e-7 2.0e-8")
    visual(
        bezel,
        "driver_control_recess",
        capsule_plate(0.0116, 0.077, 0.0002),
        "hoto_rubber",
        pos="0 .00945 .0085",
    )
    visual(
        bezel,
        "driver_control_bezel",
        capsule_frame(0.0116, 0.077, 0.0012, 0.0014),
        "hoto_rubber",
        pos="0 .0105 .0085",
    )
    # Convex collision pieces leave genuine openings for both ends and selector.
    for x in (-0.0052, 0.0052):
        collision(
            bezel,
            "control_rail_" + str(x),
            "box",
            size=".0006 .0007 .0327",
            pos=_numbers([x, 0.0105, 0.0085]),
        )
    _add(
        bezel,
        "geom",
        name="driver_led_bridge",
        type="box",
        size=".0043 .0007 .0105",
        pos="0 .0103 -.0035",
        material="hoto_rubber",
        group=2,
        mass=0,
    )
    # Two native momentary contacts under the two ends of the real rocker.
    for name, z, sign in (("forward", GRASP_Z, 1), ("reverse", 0.017, -1)):
        key = mesh("driver_" + name + "_key_mesh", rocker_half(0.0082, 0.0186, 0.0015, reverse=sign < 0))
        body = _add(tool, "body", name="driver_" + name, pos=_numbers([0, 0.01075, z]))
        _add(
            body,
            "joint",
            name="button_" + name,
            type="slide",
            axis="0 -1 0",
            range="0 .001",
            stiffness="1400",
            damping="1",
            armature=".001",
            solreflimit=".001 1",
            solimplimit=".99 .999 .00001",
        )
        _add(body, "inertial", pos="0 0 0", mass=".0007", diaginertia="2e-8 2.2e-8 5e-9")
        _add(
            body,
            "geom",
            name="button_" + name,
            type="mesh",
            mesh=key,
            material="hoto_button",
            group=2,
            mass=0,
        )
        lines = [
            ((-0.0014, 0.00077, -sign * 0.0008), (0, 0.00077, sign * 0.0011)),
            ((0, 0.00077, sign * 0.0011), (0.0014, 0.00077, -sign * 0.0008)),
        ]
        visual(body, name + "_arrow", stroke_mesh(lines), "hoto_icon")
        # Housing/guide clearance is implicit in the control joint; no fake stop
        # may prevent a real external gripper from depressing the key.
        _add(contact, "exclude", body1="driver_sleeve", body2="driver_" + name)
    # Preserve the previous third passive DOF/name for checkpoint diagnostics;
    # it now represents the genuine round torque selector, not a trigger saddle.
    gear = _add(tool, "body", name="driver_trigger_saddle", pos="0 .01075 -.021")
    _add(
        gear,
        "joint",
        name="trigger_saddle",
        type="slide",
        axis="0 -1 0",
        range="0 .001",
        stiffness="1400",
        damping="1",
        armature=".001",
        solreflimit=".001 1",
        solimplimit=".99 .999 .00001",
    )
    _add(gear, "inertial", pos="0 0 0", mass=".0014", diaginertia="7e-9 1.2e-8 7e-9")
    torque_button = mesh("driver_torque_button_mesh", capsule_plate(0.0082, 0.0082, 0.0015))
    _add(
        gear,
        "geom",
        name="driver_torque_button",
        type="mesh",
        mesh=torque_button,
        material="hoto_button",
        group=2,
        mass=0,
    )
    arcs = []
    for start in (0.25, np.pi + 0.25):
        a = np.linspace(start, start + 2.2, 10)
        points = np.column_stack((0.00165 * np.cos(a), np.full(len(a), 0.00078), 0.00165 * np.sin(a)))
        arcs.extend(zip(points[:-1], points[1:]))
        p = points[-1]
        t = np.array([-np.sin(a[-1]), 0, np.cos(a[-1])])
        radial = np.array([np.cos(a[-1]), 0, np.sin(a[-1])])
        arcs.extend(((p, p - 0.0006 * t + 0.0003 * radial), (p, p - 0.0006 * t - 0.0003 * radial)))
    visual(gear, "driver_torque_icon", stroke_mesh(arcs, radius=0.00007), "hoto_icon")
    _add(contact, "exclude", body1="driver_sleeve", body2="driver_trigger_saddle")
    for i, z in enumerate((-0.010, -0.0065, -0.003)):
        _add(
            bezel,
            "geom",
            name="driver_torque_led_" + str(i + 1),
            type="cylinder",
            size=".00043 .000045",
            pos=_numbers([0, 0.01105, z]),
            quat=".7071067812 .7071067812 0 0",
            material="hoto_amber" if i == 0 else "hoto_icon",
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )
    # A C4x28 bit: 8 mm seated in the collet, 20 mm exposed. Insertion length
    # and PH1 tip proportions are image-derived estimates, not hardware metrology.
    bit = _add(tool, "body", name="driver_bit", pos=_numbers([0, 0, CASE_HALF_LENGTH + NOSE_LENGTH]))
    _add(bit, "joint", name="spindle", type="hinge", axis="0 0 1", damping=".00002", armature=".000002")
    _add(bit, "inertial", pos="0 0 .004", mass=".0011", diaginertia="7.3e-8 7.3e-8 2.5e-9")
    visual(bit, "bit_hex_shank", lathe([(-0.008, 0.0023094), (0.012, 0.0023094)], segments=6), "hoto_steel")
    collision(bit, "bit_shaft", "mesh", mesh="bit_hex_shank_mesh")
    visual(
        bit,
        "bit_taper",
        lathe([(0.012, 0.002), (0.016, 0.0012), (0.018, 0.00065)], segments=48),
        "hoto_steel",
    )
    collision(bit, "bit_taper_contact", "mesh", mesh="bit_taper_mesh")
    for axis, hx, hy in (("x", 0.0013, 0.0003), ("y", 0.0003, 0.0013)):
        wing = mesh("bit_" + axis + "_mesh", tapered_wing(hx, hy, 0.0174, 0.020))
        _add(bit, "geom", name="bit_" + axis, type="mesh", mesh=wing, material="hoto_steel", group=2, mass=0)
    _add(tool, "site", name="driver_tip", pos=_numbers([0, 0, TIP_Z]), size=".001", rgba="0 1 0 .7", group=3)
    _add(
        tool, "site", name="driver_grasp", pos=_numbers(GRASP_CENTER), size=".002", rgba="1 .5 0 .7", group=3
    )
    # The shank seats inside its own collet and controls slide in housing guides.
    # Match free-parent joint filtering even when a test bolts the tool to world.
    # These exclusions involve internal device parts only, never fingers/parts.
    for child in ("driver_forward", "driver_reverse", "driver_trigger_saddle", "driver_bit"):
        _add(contact, "exclude", body1="driver", body2=child)
    _add(contact, "exclude", body1="driver_sleeve", body2="driver_bit")
    plugin = _add(extension, "plugin", plugin="so101.contact_driver")
    instance = _add(plugin, "instance", name="button_motor")
    for key, value in (("forward", "button_forward"), ("reverse", "button_reverse"), ("spindle", "spindle")):
        _add(instance, "config", key=key, value=value)
    return tool
