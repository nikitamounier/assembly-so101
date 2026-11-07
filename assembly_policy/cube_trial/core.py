"""Stock-MuJoCo SO101 cube handoff workcell and reproducible reset sampling.

Only the original SO101 visual/collision meshes and native MuJoCo primitives
are used. Reset states are kinematic initial conditions, never weld constraints.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from assembly_policy.so101.robot import arm_indices
from assembly_policy.so101.scene import ASSETS, JOINTS, add, numbers

CUBE_SIZE = 0.036
BASE_REAR_OFFSET = 0.030982700115234574
BASE_X = 0.375 - BASE_REAR_OFFSET
GRASP_REFERENCE = 0.27
GRASP_QPOS = 0.2695
GRASP_CTRL = 0.26
TARGET_CENTER = np.array([0.10, 0.0, 0.0])
TARGET_RADIUS = CUBE_SIZE
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
FAMILIES = (
    "loose",
    "left_pickup",
    "converging_transport",
    "pre_handoff",
    "dual_grasp",
    "post_handoff",
    "right_transport",
    "near_target",
    "placed_held",
    "partial_right_contact",
    "light_right_grasp",
)


@lru_cache(maxsize=1)
def build_xml():
    root = ET.Element("mujoco", model="so101_native_cube_handoff")
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep=".002", integrator="implicitfast", cone="elliptic", iterations="80")
    default = add(root, "default")
    add(default, "geom", friction="1 .005 .0001", condim="4", solref=".004 1", solimp=".95 .99 .001")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1440", offheight="1080")
    add(visual, "headlight", ambient=".55 .55 .55", diffuse=".7 .7 .7")
    asset = add(root, "asset")
    world = add(root, "worldbody")
    add(world, "light", pos="0 0 1.5", dir="0 0 -1")
    add(world, "geom", name="table", type="plane", size=".48 .32 .02", rgba=".19 .23 .28 1")
    target = add(world, "body", name="target", mocap="true", pos=numbers(TARGET_CENTER))
    add(target, "site", name="target_center", pos="0 0 .001", size=".001", rgba=".1 .8 .5 1")
    for i in range(96):
        angles = np.array([i, i + 1]) * (2 * np.pi / 96)
        p = np.column_stack(
            (TARGET_RADIUS * np.cos(angles), TARGET_RADIUS * np.sin(angles), np.full(2, 0.0007))
        )
        add(
            target,
            "geom",
            name=f"target_ring_{i}",
            type="capsule",
            fromto=numbers(p),
            size=".0007",
            contype="0",
            conaffinity="0",
            rgba=".1 .85 .55 1",
            group="1",
        )
    # Nominal reference parabola, drawn with decorative native primitives.
    path_points = np.array(
        [[-0.10 + 0.20 * t, 0, 0.018 + 4 * 0.327 * t * (1 - t)] for t in np.linspace(0, 1, 49)]
    )
    for i, (start, end) in enumerate(zip(path_points[:-1], path_points[1:])):
        add(
            world,
            "geom",
            name=f"nominal_path_{i}",
            type="capsule",
            fromto=numbers([start, end]),
            size=".0007",
            rgba=".4 .7 .95 .3",
            group="1",
            contype="0",
            conaffinity="0",
        )
    acts = add(root, "actuator")
    for side, x, yaw in [("left", -BASE_X, 0), ("right", BASE_X, np.pi)]:
        src = ET.parse(ASSETS / "so101/so101_new_calib.xml").getroot()
        for mesh in src.findall("asset/mesh"):
            from pathlib import Path

            mesh.set("name", Path(mesh.get("file")).stem)
            mesh.set("file", str(ASSETS / "so101/assets" / mesh.get("file")))
        # Native boxes describe the open fingers; the upstream single convex
        # hulls fill the finger opening. Visual CAD and body inertias stay intact.
        gripper = src.find(".//body[@name='gripper']")
        moving = src.find(".//body[@name='moving_jaw_so101_v1']")
        for geom in src.iter("geom"):
            if geom.get("class") == "collision" and geom.get("mesh") in (
                "wrist_roll_follower_so101_v1",
                "moving_jaw_so101_v1",
            ):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        site = gripper.find("site[@name='gripperframe']")
        sp = np.fromstring(site.get("pos"), sep=" ")
        sq = np.fromstring(site.get("quat"), sep=" ")
        sr = Rotation.from_quat(np.r_[sq[1:], sq[0]]).as_matrix()
        bp = np.fromstring(moving.get("pos"), sep=" ")
        bq = np.fromstring(moving.get("quat"), sep=" ")
        br = (
            Rotation.from_quat(np.r_[bq[1:], bq[0]]).as_matrix()
            @ Rotation.from_euler("z", GRASP_REFERENCE).as_matrix()
        )
        mq = Rotation.from_matrix(br.T @ sr).as_quat()
        add(
            gripper,
            "geom",
            name="fixed_pad",
            type="box",
            pos=numbers(sp + sr @ np.array([0, 0, -0.002])),
            quat=numbers(sq),
            size=".008 .003 .002",
            rgba=".06 .07 .09 1",
            group="2",
            margin=".0002",
            gap=".0002",
            density="0",
        )
        add(
            moving,
            "geom",
            name="moving_pad",
            type="box",
            pos=numbers(br.T @ (sp + sr @ np.array([0, 0, 0.038]) - bp)),
            quat=numbers(np.r_[mq[3], mq[:3]]),
            size=".008 .003 .002",
            rgba=".06 .07 .09 1",
            group="2",
            margin=".0002",
            gap=".0002",
            density="0",
        )
        prefix = side + "_"
        for element in src.iter():
            for key in ("name", "class", "childclass", "mesh", "material", "joint", "body", "site"):
                if element.get(key):
                    element.set(key, prefix + element.get(key))
        for defaults in src.findall("default"):
            for item in defaults:
                default.append(copy.deepcopy(item))
        for item in src.find("asset"):
            asset.append(copy.deepcopy(item))
        body = copy.deepcopy(src.find("worldbody/body"))
        body.set("pos", numbers([x, 0, 0.0024]))
        body.set("quat", numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        world.append(body)
        for actuator in src.find("actuator"):
            if actuator.get("name") == prefix + "gripper":
                actuator.set("forcerange", "-.5 .5")
            acts.append(copy.deepcopy(actuator))
    cube = add(world, "body", name="cube", pos=numbers([-0.10, 0, CUBE_SIZE / 2]))
    add(cube, "freejoint", name="cube_free")
    add(
        cube,
        "geom",
        name="cube_geom",
        type="box",
        size=numbers([CUBE_SIZE / 2] * 3),
        mass=".025",
        rgba=".95 .3 .12 1",
        friction="1.2 .01 .001",
    )
    # Face colors stay in the cube frame: +Z blue, -Z purple identify the
    # receiver's retained top/bottom grip throughout its quarter turn.
    for axis, colors in enumerate(
        [
            [(0.95, 0.2, 0.12, 1), (0.2, 0.85, 0.9, 1)],
            [(0.95, 0.8, 0.12, 1), (0.25, 0.8, 0.3, 1)],
            [(0.22, 0.4, 0.95, 1), (0.65, 0.25, 0.9, 1)],
        ]
    ):
        for sign, color in zip((1, -1), colors):
            pos = np.zeros(3)
            pos[axis] = sign * (CUBE_SIZE / 2 + 0.00005)
            size = np.full(3, 0.014)
            size[axis] = 0.00005
            add(
                cube,
                "geom",
                name=f"cube_face_{axis}_{'pos' if sign > 0 else 'neg'}",
                type="box",
                pos=numbers(pos),
                size=numbers(size),
                rgba=numbers(color),
                group="1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
    add(cube, "site", name="cube_center", size=".001", rgba="1 1 1 0")
    return ET.tostring(root, encoding="unicode")


@lru_cache(maxsize=1)
def model():
    return mujoco.MjModel.from_xml_string(build_xml())


def signature():
    return hashlib.sha256((mujoco.__version__ + "\n" + build_xml()).encode()).hexdigest()


def _home(m, d):
    for side in ("left", "right"):
        qi, _ = arm_indices(m, side)
        d.qpos[qi] = [0, -0.7, -0.5, 0.3, 0, 0.8]
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    mujoco.mj_forward(m, d)


def _solve(m, d, side, center, yaw, offset=(0, 0, 0), pitch=None):
    qi, _ = arm_indices(m, side)
    if side == "left" and center[2] > 0.25:
        # Select the real raised-elbow branch rather than the lower branch
        # reached from home. Joint bounds and native contact checks are unchanged.
        d.qpos[qi[:5]] = [0.03167, 0.21894, -1.28851, 1.09993, 1.62042]
    a = np.pi / 4 if pitch is None else float(pitch)
    if side == "left":
        x = np.array([np.cos(a), 0, -np.sin(a)])
        z = np.array([0, 1, 0])
    else:
        x = np.array([-1, 0, 0])
        z = np.array([0, 0, 1])
    cube_rotation = Rotation.from_euler("z", yaw).as_matrix() if np.isscalar(yaw) else np.asarray(yaw)
    orientation = cube_rotation @ np.column_stack((x, np.cross(z, x), z))
    patch_offset = np.array([-0.004, 0, -0.006]) if side == "left" else np.array([0.004, 0, 0])
    position = np.asarray(center) + offset + cube_rotation @ patch_offset - CUBE_SIZE / 2 * orientation[:, 2]
    site = m.site(side + "_gripperframe").id
    ranges = m.jnt_range[[m.joint(side + "_" + name).id for name in JOINTS[:5]]]

    def residual(q):
        d.qpos[qi[:5]] = q
        mujoco.mj_kinematics(m, d)
        return np.r_[
            d.site_xpos[site] - position,
            0.06 * (d.site_xmat[site].reshape(3, 3)[:, 2] - orientation[:, 2]),
            0.00005 * (d.site_xmat[site].reshape(3, 3)[:, 0] - orientation[:, 0]),
        ]

    sol = least_squares(
        residual,
        np.clip(d.qpos[qi[:5]], ranges[:, 0] + 1e-7, ranges[:, 1] - 1e-7),
        bounds=(ranges[:, 0], ranges[:, 1]),
        max_nfev=150,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    error = residual(sol.x)
    actual = d.site_xmat[site].reshape(3, 3)
    jaw_angle = float(np.arccos(np.clip(np.dot(actual[:, 2], orientation[:, 2]), -1, 1)))
    if np.linalg.norm(error[:3]) > 0.001 or jaw_angle > 0.05:
        raise ValueError(
            f"{side} unreachable: position={np.linalg.norm(error[:3]):.4g},jaw_axis_angle={jaw_angle:.4g}"
        )
    return [float(np.linalg.norm(error[:3])), jaw_angle]


def _sample_config(family, rng):
    module = importlib.import_module(f"assembly_policy.cube_trial.families.{family}")
    cfg = copy.deepcopy(module.sample(rng) if hasattr(module, "sample") else module.CONFIG)
    for name in ("cube", "target"):
        default = TARGET_CENTER if name == "target" else [-0.125, 0, 0.018]
        cfg[name + "_center"] = np.asarray(cfg.get(name + "_center", default), float) + rng.uniform(
            -1, 1, 3
        ) * np.asarray(cfg.get(name + "_jitter", [0, 0, 0]))
    cfg["cube_yaw"] = float(cfg.get("cube_yaw", 0) + rng.uniform(-1, 1) * cfg.get("yaw_jitter", 0))
    for side in ("left", "right"):
        cfg[side + "_offset"] = np.asarray(cfg.get(side + "_offset", [0, 0, 0]), float) + rng.uniform(
            -1, 1, 3
        ) * np.asarray(cfg.get(side + "_offset_jitter", [0, 0, 0]))
    from .path import sample_path

    cfg.update(sample_path(family, rng))
    cfg["cube_center"] = np.asarray(cfg["cube_center"], float)
    cfg["target_center"] = np.asarray(cfg["target_center"], float)
    return cfg


def contact_summary(m, d):
    cube = m.geom("cube_geom").id
    pads = {
        side: [m.geom(side + "_" + part + "_pad").id for part in ("fixed", "moving")]
        for side in ("left", "right")
    }
    counts = {side: [0, 0] for side in pads}
    forces = {side: [0.0, 0.0] for side in pads}
    faces = {side: [set(), set()] for side in pads}
    cube_rotation = d.geom_xmat[cube].reshape(3, 3)
    maximum = max([0.0, *[-float(c.dist) for c in d.contact]])
    pairs = []
    for contact_index, c in enumerate(d.contact):
        if cube not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == cube else c.geom1
        for side, geoms in pads.items():
            if other in geoms:
                index = geoms.index(other)
                counts[side][index] += 1
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, contact_index, wrench)
                forces[side][index] += float(wrench[0])
                if wrench[0] > 0:
                    normal = cube_rotation.T @ np.asarray(c.frame[:3])
                    if c.geom2 == cube:
                        normal = -normal
                    axis = int(np.argmax(np.abs(normal)))
                    if abs(normal[axis]) > 0.9:
                        faces[side][index].add(("+" if normal[axis] > 0 else "-") + "XYZ"[axis])
        pairs.append(
            dict(geom=m.geom(other).name or m.body(m.geom_bodyid[other]).name, distance=float(c.dist))
        )
    return dict(
        pad_contact_counts=counts,
        pad_normal_forces_n=forces,
        pad_cube_faces={side: [sorted(v) for v in values] for side, values in faces.items()},
        cube_contacts=pairs,
        max_penetration_m=maximum,
        bilateral_contacts={s: all(v) for s, v in counts.items()},
    )


def right_cube_clearance(m, d):
    """Native signed clearance, including the visible right-hand CAD meshes.

    Querying all right-arm geoms also prevents a non-pad surface touching the
    cube. Mesh distances use MuJoCo's conservative convex collision shapes.
    """
    cube = m.geom("cube_geom").id
    return min(
        mujoco.mj_geomDistance(m, d, cube, geom, 0.1, None)
        for geom in range(m.ngeom)
        if m.body(m.geom_bodyid[geom]).name.startswith("right_")
    )


def _secure_contacts(contacts, holders):
    for side in holders:
        expected = ["-Y", "+Y"] if side == "left" else ["-Z", "+Z"]
        if not contacts["bilateral_contacts"][side] or min(contacts["pad_normal_forces_n"][side]) <= 0:
            return False
        if any(face not in contacts["pad_cube_faces"][side][i] for i, face in enumerate(expected)):
            return False
    return True


def _relative_cube_pose(data, side):
    site = data.site(side + "_gripperframe")
    hand_rotation = site.xmat.reshape(3, 3)
    cube = data.body("cube")
    return hand_rotation.T @ (cube.xpos - site.xpos), hand_rotation.T @ cube.xmat.reshape(3, 3)


def arm_posture(data, side):
    shoulder = data.body(side + "_upper_arm").xpos
    elbow = data.body(side + "_lower_arm").xpos
    wrist = data.body(side + "_wrist").xpos
    hand = data.site(side + "_gripperframe").xpos
    upper = elbow - shoulder
    fore = wrist - elbow
    return dict(
        upper_arm_up_deg=float(np.degrees(np.arctan2(upper[2], np.linalg.norm(upper[:2])))),
        forearm_up_deg=float(np.degrees(np.arctan2(fore[2], np.linalg.norm(fore[:2])))),
        hand_above_elbow_m=float(hand[2] - elbow[2]),
        elbow_height_m=float(elbow[2]),
        wrist_height_m=float(wrist[2]),
        hand_height_m=float(hand[2]),
    )


def generate(family, seed=0, model=None):
    """Return one sampled physical reset and JSON-safe qualification metadata.

    Rejection is bounded and reproducible. Both fingertip boxes must touch for
    every requested holder. All arm joints retain their position actuators.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown cube reset family {family!r}")
    if family in ("partial_right_contact", "light_right_grasp"):
        from pathlib import Path
        import json
        directory = Path(__file__).resolve().parents[2] / "data/cube_trial/banks_handoff_bridge_v8"
        bank = np.load(directory / "cube_handoff_train.npz")
        rows = np.flatnonzero(bank["family"] == FAMILIES.index(family))
        row = int(rows[np.random.default_rng(seed).integers(len(rows))])
        m = model if model is not None else globals()["model"]()
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, bank["states"][row], STATE_SPEC)
        mujoco.mj_forward(m, d)
        records = json.loads((directory / "bridge-provenance.json").read_text())
        receipt = next(r for r in records if r["split"] == "train" and r["row"] == row)
        return d, receipt

    m = model if model is not None else globals()["model"]()
    rng = np.random.default_rng(int(seed))
    last = "not attempted"
    for attempt in range(64):
        cfg = _sample_config(family, rng)
        d = mujoco.MjData(m)
        _home(m, d)
        cube = cfg["cube_center"]
        cube_quat = np.asarray(cfg["cube_quaternion"], float)
        cube_rotation = Rotation.from_quat(np.r_[cube_quat[1:], cube_quat[0]]).as_matrix()
        q = int(m.joint("cube_free").qposadr[0])
        d.qpos[q : q + 3] = cube
        d.qpos[q + 3 : q + 7] = cube_quat
        d.mocap_pos[m.body("target").mocapid[0]] = cfg["target_center"]
        errors = {}
        holders = []
        try:
            for side in ("left", "right"):
                mode = cfg.get(side + "_mode", "home")
                qi, _ = arm_indices(m, side)
                if mode != "home":
                    errors[side] = _solve(
                        m,
                        d,
                        side,
                        cube,
                        cube_rotation if mode == "grasp" or side == "right" else np.eye(3),
                        cfg[side + "_offset"],
                        cfg.get(side + "_pitch"),
                    )
                    d.qpos[qi[5]] = GRASP_QPOS if mode == "grasp" else float(cfg.get("gripper_open", 0.8))
                    if mode == "grasp":
                        holders.append(side)
            for i in range(m.nu):
                d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
                if m.actuator(i).name in [s + "_gripper" for s in holders]:
                    d.ctrl[i] = GRASP_CTRL
            mujoco.mj_forward(m, d)
            contacts = contact_summary(m, d)
            approach_clearance = right_cube_clearance(m, d) if family == "pre_handoff" else None
            if approach_clearance is not None and approach_clearance < 0.005:
                raise ValueError("pre-handoff requires at least 5mm right-hand cube clearance")
            postures = {side: arm_posture(d, side) for side in ("left", "right")}
            if family in ("pre_handoff", "dual_grasp") and any(
                p["upper_arm_up_deg"] < 20 or p["forearm_up_deg"] < 35 or p["hand_above_elbow_m"] < 0.04
                for p in postures.values()
            ):
                raise ValueError("handoff requires both arms pointing upward with hands above elbows")
            if not _secure_contacts(contacts, holders):
                raise ValueError(
                    "requested holder lacks loaded native contacts on the specified cube face pair"
                )
            if contacts["max_penetration_m"] > 0.001:
                worst = min(d.contact, key=lambda c: c.dist)
                raise ValueError(
                    f"native penetration {contacts['max_penetration_m']:.4g}m: {m.geom(worst.geom1).name or m.body(m.geom_bodyid[worst.geom1]).name}/{m.geom(worst.geom2).name or m.body(m.geom_bodyid[worst.geom2]).name}"
                )
            if not np.isfinite(d.qpos).all():
                raise ValueError("nonfinite state")
            # Qualify the requested hold using ordinary forward dynamics, then
            # restore the exact sampled reset for the viewer/bank. This prevents
            # a merely touching but ejecting pose from entering the bank.
            hold_probe = None
            if holders:
                initial_state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
                mujoco.mj_getState(m, d, initial_state, STATE_SPEC)
                initial_cube = d.body("cube").xpos.copy()
                relative = {side: _relative_cube_pose(d, side) for side in holders}
                max_slip = {side: 0.0 for side in holders}
                max_rotation = {side: 0.0 for side in holders}
                max_cube_drift = 0.0
                minimum_pad_force = {side: [float("inf"), float("inf")] for side in holders}
                minimum_arm_posture = {
                    side: dict(
                        upper_arm_up_deg=postures[side]["upper_arm_up_deg"],
                        forearm_up_deg=postures[side]["forearm_up_deg"],
                        hand_above_elbow_m=postures[side]["hand_above_elbow_m"],
                    )
                    for side in ("left", "right")
                }
                cube_id = m.body("cube").id
                for step in range(250):
                    d.xfrc_applied[cube_id] = [0.25, 0, 0, 0.0015, 0, 0] if step < 25 else np.zeros(6)
                    mujoco.mj_step(m, d)
                    mujoco.mj_kinematics(m, d)
                    max_cube_drift = max(
                        max_cube_drift, float(np.linalg.norm(d.body("cube").xpos - initial_cube))
                    )
                    for side in holders:
                        p_now, R_now = _relative_cube_pose(d, side)
                        p_start, R_start = relative[side]
                        max_slip[side] = max(max_slip[side], float(np.linalg.norm(p_now - p_start)))
                        angle = float(np.arccos(np.clip((np.trace(R_start.T @ R_now) - 1) / 2, -1, 1)))
                        max_rotation[side] = max(max_rotation[side], angle)
                    if (step + 1) % 5 == 0:
                        mujoco.mj_forward(m, d)
                        if approach_clearance is not None:
                            approach_clearance = min(approach_clearance, right_cube_clearance(m, d))
                            if approach_clearance < 0.005:
                                raise ValueError("right hand approached cube during pre-handoff hold")
                        held_contacts = contact_summary(m, d)
                        for arm in ("left", "right"):
                            measured = arm_posture(d, arm)
                            for metric in minimum_arm_posture[arm]:
                                minimum_arm_posture[arm][metric] = min(
                                    minimum_arm_posture[arm][metric], measured[metric]
                                )
                        if family in ("pre_handoff", "dual_grasp") and any(
                            p["upper_arm_up_deg"] < 20
                            or p["forearm_up_deg"] < 35
                            or p["hand_above_elbow_m"] < 0.04
                            for p in minimum_arm_posture.values()
                        ):
                            raise ValueError("raised handoff posture lost during500ms hold")
                        if not _secure_contacts(held_contacts, holders):
                            raise ValueError(
                                f"lost specified loaded face pair during500ms hold at {step + 1} steps"
                            )
                        if held_contacts["max_penetration_m"] > 0.001 or d.warning.number.any():
                            raise ValueError("native overlap/warning during500ms hold")
                        for side in holders:
                            minimum_pad_force[side] = [
                                min(a, b)
                                for a, b in zip(
                                    minimum_pad_force[side], held_contacts["pad_normal_forces_n"][side]
                                )
                            ]
                if (
                    max_cube_drift > 0.003
                    or max(max_slip.values()) > 0.002
                    or max(max_rotation.values()) > 0.05
                ):
                    raise ValueError(
                        f"unstable500ms hold: drift={max_cube_drift:.4g},slip={max(max_slip.values()):.4g},rotation={max(max_rotation.values()):.4g}"
                    )
                hold_probe = dict(
                    duration_s=0.5,
                    cube_drift_m=max_cube_drift,
                    max_relative_slip_m=max_slip,
                    max_relative_rotation_rad=max_rotation,
                    minimum_pad_forces_n=minimum_pad_force,
                    minimum_arm_posture=minimum_arm_posture,
                    contact_sample_interval_s=0.01,
                    loaded_face_pairs_maintained=True,
                    perturbation=dict(force_n=[0.25, 0, 0], torque_nm=[0.0015, 0, 0], duration_s=0.05),
                    contacts=held_contacts,
                )
                mujoco.mj_setState(m, d, initial_state, STATE_SPEC)
                mujoco.mj_forward(m, d)
            meta = dict(
                family=family,
                seed=int(seed),
                attempt=attempt,
                cube_size=CUBE_SIZE,
                path_progress=cfg["path_progress"],
                pickup_center=cfg["pickup_center"],
                endpoint_center=cfg["endpoint_center"],
                arc_height=cfg["arc_height"],
                rotation_angle=cfg["rotation_angle"],
                cube_pose=d.qpos[q : q + 7].tolist(),
                target_center=cfg["target_center"].tolist(),
                target_radius=TARGET_RADIUS,
                holder=holders,
                minimum_right_cube_clearance_m=approach_clearance,
                expected_grasp_faces={
                    side: (["-Y", "+Y"] if side == "left" else ["-Z", "+Z"]) for side in holders
                },
                family_description=cfg.get("description", ""),
                grip_geometry=dict(
                    reference_angle=GRASP_REFERENCE,
                    initial_angle=GRASP_QPOS,
                    holding_command=GRASP_CTRL,
                    visible_native_pads=True,
                ),
                ik_errors=errors,
                arm_posture=postures,
                base_rear_span_m=0.75,
                base_center_span_m=2 * BASE_X,
                base_rear_offset_m=BASE_REAR_OFFSET,
                apex_height_m=0.018 + cfg["arc_height"],
                contacts=contacts,
                collision_model="native boxes for open fingertips; original SO101 mesh visuals, arm/servo collisions and inertias",
                constraint_count=int(m.neq),
                initialization="kinematic reset qualified by500ms perturbed native hold dynamics",
                hold_probe=hold_probe,
                settle_steps_requested=int(cfg.get("settle_steps", 0)),
                settle_steps_applied=0,
                success=success(m, d),
            )
            return d, meta
        except ValueError as exc:
            last = str(exc)
    raise ValueError(f"{family} seed {seed} rejected after 64 bounded attempts: {last}")


RELEASE_ANGLE_RAD = 0.275
BOUNDARY_TOLERANCE_M = 0.002
PAD_LOAD_THRESHOLD_N = 0.01
RIGHT_GRASP_LOOKBACK_S = 0.100
MAX_LINEAR_SPEED_M_S = 0.02
MAX_ANGULAR_SPEED_RAD_S = 0.5
REWARD_CONTRACT_VERSION = "placement_v4_table_only"


def placement_contacts(m, d):
    """Read native table contact and summed normal load on every physical pad."""
    cube = m.geom("cube_geom").id
    table = m.geom("table").id
    pads = [
        m.geom(side + "_" + part + "_pad").id for side in ("left", "right") for part in ("fixed", "moving")
    ]
    loads = np.zeros(4)
    touching = np.zeros(4, bool)
    table_contact = False
    table_touching = False
    wrench = np.empty(6)
    for index, contact in enumerate(d.contact):
        if cube not in (contact.geom1, contact.geom2):
            continue
        other = int(contact.geom2 if contact.geom1 == cube else contact.geom1)
        mujoco.mj_contactForce(m, d, index, wrench)
        normal = max(0.0, float(wrench[0]))
        if other == table and contact.dist <= 0:
            table_touching = True
            table_contact |= normal > 1e-6
        if other in pads:
            pad_index = pads.index(other)
            loads[pad_index] += normal
            touching[pad_index] |= contact.dist <= 0
    return dict(
        table_contact=table_contact,
        table_touching=table_touching,
        pad_loads_n=loads,
        pad_touching=touching,
        right_held=bool(np.all(loads[2:] > PAD_LOAD_THRESHOLD_N)),
        pad_support=bool(np.any(loads > PAD_LOAD_THRESHOLD_N)),
    )


def cube_inside_target(m, d):
    """Whole physical cube footprint, with the agreed2mm boundary tolerance."""
    cube = d.body("cube")
    rotation = cube.xmat.reshape(3, 3)
    corners = (
        np.array(np.meshgrid(*[[-CUBE_SIZE / 2, CUBE_SIZE / 2]] * 3, indexing="ij")).reshape(3, -1).T
        @ rotation.T
        + cube.xpos
    )
    target = d.site("target_center").xpos
    return bool(
        np.max(np.linalg.norm(corners[:, :2] - target[:2], axis=1)) <= TARGET_RADIUS + BOUNDARY_TOLERANCE_M
    )


def success(m, d):
    """Placement contact/release checks; episode history is enforced in env."""
    inside = cube_inside_target(m, d)
    contacts = placement_contacts(m, d)
    released = not contacts["pad_support"] and d.qpos[arm_indices(m, "right")[0][5]] >= RELEASE_ANGLE_RAD
    velocity = int(m.joint("cube_free").dofadr[0])
    slow = (
        np.linalg.norm(d.qvel[velocity : velocity + 3]) < MAX_LINEAR_SPEED_M_S
        and np.linalg.norm(d.qvel[velocity + 3 : velocity + 6]) < MAX_ANGULAR_SPEED_RAD_S
    )
    return bool(inside and contacts["table_contact"] and released and slow)
