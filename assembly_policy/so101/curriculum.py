"""Generate noisy, contact-qualified SGS starts without changing shared physics.

Prepared initialization is explicit. Qualification uses the production reset and
bounded twelve-action interface, with active squeeze rather than a grasp weld.
Original loose EVAL files stay unchanged; family holdouts use independent seeds.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import igl
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from assembly_policy.so101.limits import PENETRATION_LIMIT_M

from .env import ARM_ACTIONS, STATE_SPEC, BatchedAssemblyEnv, EnvConfig
from .fastener import PITCH, bearing_pose, origins, target_rotations
from .goals import placement_errors
from .resets import check_bank, directory_signature, validation_identity
from .robot import down_rotation, home, ik
from .scene import ASSEMBLY_TASKS, ASSETS, PARTS, metadata
from .task_scenes import task_scene
from .workcell import configuration, default_bank_directory, environment_identity

FAMILIES = ("loose", "pickup", "inhand", "partial", "near")
DEPTH = PENETRATION_LIMIT_M
FORK_ENTRY_MARGIN = 0.001  # measurable capture beyond the fork opening
NOISE = dict(
    workcell_xy_m=0.003,
    workcell_yaw_deg=2.0,
    grasp_translation_m=0.00015,
    grasp_rotation_deg=0.3,
    approach_lateral_m=0.0005,
    approach_axis_m=0.0003,
    predecessor_preparation_z_m=0.0004,
)


def candidate_digest(item):
    """Bind qualification to exact persisted state and controller targets."""
    digest = hashlib.sha256()
    for name in ("state", "controls"):
        array = np.asarray(item[name], dtype="<f8")
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def get_state(m, d):
    s = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
    mujoco.mj_getState(m, d, s, STATE_SPEC)
    return s


def set_pose(m, d, name, position, rotation):
    q = int(m.joint(name + "_free").qposadr[0])
    d.qpos[q : q + 3] = position
    quat = Rotation.from_matrix(rotation).as_quat()
    d.qpos[q + 3 : q + 7] = quat[[3, 0, 1, 2]]


@lru_cache(maxsize=6)
def surface(name):
    with np.load(ASSETS / "parts" / f"{name}.npz") as z:
        v, f = z["vertices"].astype(float), z["faces"].astype(np.int64)
    winding = igl.FastWindingNumberBVH()
    winding.init(v, f)
    tree = igl.AABB()
    tree.init(v, f)
    return v, f, np.vstack((v, v[f].mean(1))), winding, tree


def material_overlaps(d, names=PARTS):
    """Both meshes' vertices/centroids; analytic screw uses native contacts."""
    names = [n for n in names if n != "screw"]
    result = {}
    boxes = {}
    for name in names:
        v = surface(name)[0]
        body = d.body(name)
        corners = np.array(np.meshgrid(*np.array([v.min(0), v.max(0)]).T, indexing="ij")).reshape(3, -1).T
        world = corners @ body.xmat.reshape(3, 3).T + body.xpos
        boxes[name] = (world.min(0), world.max(0))
        result[name + "/table"] = 0.0
        if world[:, 2].min() < 0:
            floor_depth = -(v @ body.xmat.reshape(3, 3).T + body.xpos)[:, 2].min()
            result[name + "/table"] = max(0.0, float(floor_depth))
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            worst = 0.0
            if np.any(boxes[a][1] < boxes[b][0]) or np.any(boxes[b][1] < boxes[a][0]):
                result[a + "/" + b] = 0.0
                continue
            for source, target in ((a, b), (b, a)):
                obj, other = d.body(source), d.body(target)
                v, f, _, winding, tree = surface(target)
                points = (
                    surface(source)[2] @ obj.xmat.reshape(3, 3).T + obj.xpos - other.xpos
                ) @ other.xmat.reshape(3, 3)
                points = points[np.all((points >= v.min(0)) & (points <= v.max(0)), axis=1)]
                if not len(points):
                    continue
                inside = winding.winding_number(points) > 0.5
                if inside.any():
                    distances = tree.squared_distance(v, f, points[inside])[0]
                    worst = max(worst, float(np.sqrt(distances.max())))
            result[a + "/" + b] = worst
    return result


def contact_state(m, d, task, side, predecessors):
    forces = {side + "_gripper": 0.0, side + "_moving_jaw_so101_v1": 0.0}
    touching = set()
    receiving = 0.0
    table = 0.0
    external = 0.0
    robot = 0.0
    for i, c in enumerate(d.contact):
        names = {m.body(int(b)).name for b in m.geom_bodyid[c.geom]}
        if task not in names:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        for name in names.intersection(forces):
            forces[name] += float(f[0])
        if c.dist <= 0.0001:
            touching.update(names.intersection(forces))
        if names.intersection({"base", *predecessors}):
            receiving += float(f[0])
        if "world" in names:
            table += float(f[0])
        if names.intersection(set(PARTS) - {task} | {"driver", "world"}):
            external += float(f[0])
        if any(n.startswith(("left_", "right_", "driver")) for n in names - {task}):
            robot += float(f[0])
    return dict(
        bilateral=min(forces.values()) > 0.001,
        jaw_forces_N=forces,
        both_jaws_touching=len(touching) == 2,
        table_force_N=table,
        external_support_force_N=external,
        robot_force_N=robot,
        receiver_force_N=receiving,
    )


def thread_engagement(m, d):
    """Geometric interlocking can be unloaded while the fingers carry the screw."""

    def parameters(name):
        instance = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PLUGIN, name)
        start = int(m.plugin_attradr[instance])
        return [float(x) for x in bytes(m.plugin_attr[start:]).split(b"\0")[:9]]

    male, female = parameters("male"), parameters("female")
    fg, mg = m.geom("female_contact").id, m.geom("male_contact").id
    R = d.geom_xmat[fg].reshape(3, 3)
    relative = R.T @ d.geom_xmat[mg].reshape(3, 3)
    position = R.T @ (d.geom_xpos[mg] - d.geom_xpos[fg])
    radial = float(np.linalg.norm(position[:2]))
    angle = float(np.arccos(np.clip(relative[2, 2], -1, 1)))
    overlap = max(
        0.0,
        min(position[2] + male[5] * relative[2, 2], female[7])
        - max(position[2] + male[4] * relative[2, 2], female[6]),
    )
    yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
    phase = position[2] / female[0] - yaw / (2 * np.pi) + female[8] - male[8]
    error = abs((phase + 0.5) % 1 - 0.5) * female[0]
    return dict(
        axial_overlap_m=float(overlap),
        radial_offset_m=radial,
        axis_error_rad=angle,
        phase_error_m=float(error),
        engaged=bool(
            overlap >= female[0] and radial <= female[2] + 0.00002 and angle <= 0.02 and error <= 0.00005
        ),
    )


def yoke_envelope_overlap(d):
    """Insertion depth of the motor into the fork's open end, allowing clearance."""
    axis = d.body("base").xmat.reshape(3, 3) @ np.asarray(
        configuration()["assembly_plan"]["extractions"]["yoke"], float
    )
    axis /= np.linalg.norm(axis)
    ranges = []
    for name in ("motor", "yoke"):
        body = d.body(name)
        values = (surface(name)[0] @ body.xmat.reshape(3, 3).T + body.xpos) @ axis
        ranges.append((float(values.min()), float(values.max())))
    return max(0.0, min(x[1] for x in ranges) - max(x[0] for x in ranges))


def holding_action(env, controls, gentle_grip=None):
    """Bounded oracle used only to qualify starts, never supplied to PPO."""
    q = env.state[:, 1:13]
    scales = np.tile(env.ecfg.action_scale_joints, 2)
    action = np.zeros_like(q)
    action[:, ARM_ACTIONS] = (controls[:, ARM_ACTIONS] - q[:, ARM_ACTIONS]) / scales
    action[:, [5, 11]] = (controls[:, [5, 11]] - q[:, [5, 11]]) / env.ecfg.gripper_scale
    if gentle_grip is not None:
        index, aperture, limit = gentle_grip
        if q[0, index] <= aperture + 0.04:
            action[:, index] = np.clip(action[:, index], -limit, limit)
    return np.clip(action, -1.0, 1.0)


class Builder:
    def __init__(self):
        self.env = BatchedAssemblyEnv(
            list(ASSEMBLY_TASKS), [0], EnvConfig(nthread=1, max_contact_depth=DEPTH)
        )
        self.m = self.env.m
        self.meta = metadata()
        self.targets = origins(self.meta)
        self.target_rotations = target_rotations(self.meta)
        self.identity = environment_identity()
        self.validation_identity = validation_identity()
        self.templates = {}

    def close(self):
        self.env.close()

    def compensate_arm_gravity(self, d, controls):
        controls = np.asarray(controls, float).copy()
        for actuator in ARM_ACTIONS:
            joint = int(self.m.actuator_trnid[actuator, 0])
            dof = int(self.m.jnt_dofadr[joint])
            gain = float(self.m.actuator_gainprm[actuator, 0] * self.m.actuator_gear[actuator, 0])
            controls[actuator] += float(d.qfrc_bias[dof]) / gain
        return np.clip(controls, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])

    def receiver(self, task):
        if task not in self.templates:
            d = mujoco.MjData(self.m)
            home(self.m, d)
            before = ASSEMBLY_TASKS[: ASSEMBLY_TASKS.index(task)]
            preparation = None
            if before:
                loose_q = int(self.m.joint(task + "_free").qposadr[0])
                loose_pose = d.qpos[loose_q : loose_q + 7].copy()
                p = task_scene(self.m, "horn" if task == "horn" else before[-1])
                d.qpos[:] = p["qpos"]
                d.qvel[:] = p["qvel"]
                d.ctrl[:] = p["ctrl"]
                if task == "horn":
                    d.qpos[loose_q : loose_q + 7] = loose_pose
                    d.qvel[:] = 0.0
                    mujoco.mj_forward(self.m, d)
                    mujoco.mj_step(self.m, d, nstep=1000)
                    preparation = dict(assembly_orientation="upright", native_settle_s=0.5)
            else:
                mujoco.mj_step(self.m, d, nstep=1000)
            mujoco.mj_forward(self.m, d)
            overlaps = material_overlaps(d, ("base", *before))
            self.templates[task] = dict(
                state=get_state(self.m, d),
                controls=d.ctrl.copy(),
                predecessors=list(before),
                material=overlaps,
                preparation=preparation,
            )
        return self.templates[task]

    def loose_candidate(self, task, seed):
        """Only the active part is loose; completed predecessors remain dynamic."""
        rng = np.random.default_rng(seed)
        source = self.receiver(task)
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, source["state"], STATE_SPEC)
        d.ctrl[:] = source["controls"]
        mujoco.mj_kinematics(m, d)
        anchor = d.body("base").xpos.copy()
        shift = np.r_[rng.uniform(-0.008, 0.008, 2), 0.0]
        angle = rng.uniform(-8, 8) * np.pi / 180
        Q = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        noise = {}
        for name in (*PARTS, "driver"):
            body = d.body(name)
            position = anchor + Q @ (body.xpos - anchor) + shift
            rotation = Q @ body.xmat.reshape(3, 3)
            if name not in ("base", *source["predecessors"]):
                delta = np.r_[rng.uniform(-0.008, 0.008, 2), 0.0]
                yaw = rng.uniform(-8, 8) * np.pi / 180
                position += delta
                rotation = Rotation.from_rotvec([0, 0, yaw]).as_matrix() @ rotation
                noise[name] = dict(xy_m=delta[:2].tolist(), yaw_rad=yaw)
            set_pose(m, d, name, position, rotation)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        return dict(
            state=get_state(m, d),
            controls=d.ctrl.copy(),
            task=task,
            family="loose",
            seed=int(seed),
            side="left",
            predecessors=source["predecessors"],
            receiver_preparation=source["preparation"],
            prepared=True,
            noise=dict(xy_m=shift[:2].tolist(), yaw_rad=angle, independent_parts=noise),
            reset_distribution="subtask_loose",
        )

    def grasp_ik(self, d, side, point, rotation):
        try:
            return ik(self.m, d.qpos, side, point, rotation)
        except ValueError:
            pass
        # Five-axis arm: try a bounded small wrist tilt, then qualify actual contacts.
        qi = slice(0, 5) if side == "left" else slice(6, 11)
        ids = [
            self.m.joint(side + "_" + n).id
            for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        ranges = self.m.jnt_range[ids]
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = d.qpos
        site = self.m.site(side + "_gripperframe").id

        def residual(q):
            scratch.qpos[qi] = q
            mujoco.mj_kinematics(self.m, scratch)
            dr = Rotation.from_matrix(rotation @ scratch.site_xmat[site].reshape(3, 3).T).as_rotvec()
            return np.r_[scratch.site_xpos[site] - point, 0.008 * dr]

        sol = least_squares(
            residual,
            np.clip(d.qpos[qi], ranges[:, 0] + 1e-7, ranges[:, 1] - 1e-7),
            bounds=(ranges[:, 0], ranges[:, 1]),
            max_nfev=100,
        )
        error = residual(sol.x)
        if np.linalg.norm(error[:3]) > 0.001 or np.linalg.norm(error[3:]) / 0.008 > 0.2:
            raise ValueError("grasp_ik_unreachable")
        return sol.x

    def circular_grasp_ik(self, d, side, center, width):
        """Match a circular head and downward fingers; leave grip yaw free."""
        qi = slice(0, 5) if side == "left" else slice(6, 11)
        ids = [
            self.m.joint(side + "_" + n).id
            for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        bounds = self.m.jnt_range[ids]
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = d.qpos
        site = self.m.site(side + "_gripperframe").id

        def residual(q):
            scratch.qpos[qi] = q
            mujoco.mj_kinematics(self.m, scratch)
            R = scratch.site_xmat[site].reshape(3, 3)
            midpoint = scratch.site_xpos[site] + 0.5 * width * R[:, 2]
            return np.r_[midpoint - center, 0.05 * (R[:, 0] - [0.0, 0.0, -1.0])]

        for roll in (float(d.qpos[qi][4]), 1.5, -1.5):
            seed = d.qpos[qi].copy()
            seed[4] = roll
            solution = least_squares(
                residual,
                np.clip(seed, bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7),
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=180,
            )
            error = residual(solution.x)
            if np.linalg.norm(error[:3]) < 0.00005 and np.linalg.norm(error[3:]) / 0.05 < 0.02:
                return solution.x
        raise ValueError("circular_head_grasp_unreachable")

    def candidate(
        self, task, family, seed, variant=0, *, side_override=None, axis_gap=None, preserve_part_pose=False
    ):
        if side_override not in (None, "left", "right"):
            raise ValueError("Invalid grasp side override")
        if preserve_part_pose and family not in ("near", "partial"):
            raise ValueError("Fixed part pose requires a mating proposal")
        if axis_gap is not None and (
            family not in ("near", "partial") or not np.isfinite(axis_gap) or axis_gap <= 0.0
        ):
            raise ValueError("Axis gap requires a positive finite mating offset")
        rng = np.random.default_rng(seed)
        m = self.m
        receiver = self.receiver(task)
        if max(receiver["material"].values(), default=0.0) > DEPTH:
            raise ValueError("predecessor_material_overlap")
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, receiver["state"], STATE_SPEC)
        d.ctrl[:] = receiver["controls"]
        mujoco.mj_forward(m, d)
        theta = rng.uniform(-NOISE["workcell_yaw_deg"], NOISE["workcell_yaw_deg"]) * np.pi / 180
        Q = Rotation.from_rotvec([0, 0, theta]).as_matrix()
        center = d.body("base").xpos.copy()
        delta = np.r_[rng.uniform(-0.003, 0.003, 2), 0.0]
        independent_noise = {}
        for name in (*PARTS, "driver"):
            body = d.body(name)
            position = center + Q @ (body.xpos - center) + delta
            rotation = Q @ body.xmat.reshape(3, 3)
            if name not in ("base", *receiver["predecessors"]) and (
                name != task or family in ("pickup", "inhand")
            ):
                jitter = np.r_[rng.uniform(-0.003, 0.003, 2), 0.0]
                yaw_jitter = rng.uniform(-2.0, 2.0) * np.pi / 180
                position += jitter
                rotation = Rotation.from_rotvec([0, 0, yaw_jitter]).as_matrix() @ rotation
                independent_noise[name] = dict(xy_m=jitter[:2].tolist(), yaw_rad=yaw_jitter)
            set_pose(m, d, name, position, rotation)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        b = d.body("base")
        Rb = b.xmat.reshape(3, 3).copy()
        goal = b.xpos + Rb @ (self.targets[task] - self.targets["base"])
        local = np.zeros(3)
        if task == "retainer":
            local = np.array([0, (0, 0.028, -0.028)[(variant // 4) % 3], 0.009])
        if task == "yoke":
            local = np.array([0, (0, 0.019, -0.019)[(variant // 4) % 3], (0, 0.028)[(variant // 12) % 2]])
        if task == "screw":
            local = np.array([0, 0, 0.0005])
        rail_width = None
        rail_height = None
        if task == "retainer":
            points = surface(task)[2]
            rail = points[(abs(points[:, 0]) < 0.006) & (points[:, 2] > 0.006)]
            if not len(rail):
                raise ValueError("no_upper_rail_surface")
            low, high = rail[:, 1].min(), rail[:, 1].max()
            local = np.array([0.0, 0.5 * (low + high), 0.010])
            rail_width = float(high - low)
            rail_height = float(rail[:, 2].max() - rail[:, 2].min())
        narrow_wall = task == "yoke" and variant >= 24
        if narrow_wall:
            vertices = surface(task)[0]
            sign = 1.0 if (variant // 2) % 2 else -1.0
            edge = vertices[:, 0].max() if sign > 0 else vertices[:, 0].min()
            band = vertices[
                (abs(vertices[:, 0] - edge) < 0.001) & (vertices[:, 2] > vertices[:, 2].max() - 0.015)
            ]
            if not len(band):
                raise ValueError("no_grasp_wall_surface")
            wanted_y = (0.0, 0.025, -0.025)[(variant // 4) % 3]
            local = band[np.argmin(abs(band[:, 1] - wanted_y))].copy()
            local[0] -= sign * 0.0015
        axis_delta = 0.0
        lateral_delta = np.zeros(3)
        if family in ("near", "partial"):
            axis = Rb @ np.asarray(configuration()["assembly_plan"]["extractions"][task], float)
            axis /= np.linalg.norm(axis)
            gaps = {
                "motor": (0.04, 0.006, 0.008, 0.01),
                "horn": (0.012, 0.001, 0.002, 0.003),
                "screw": (0.007, 0.0005, 0.001, 0.0015),
                "retainer": (0.042, 0.003, 0.005, 0.008),
                "yoke": (0.045, 0.006, 0.01, 0.014),
            }[task]
            gap = (
                float(axis_gap)
                if axis_gap is not None
                else (gaps[0] if family == "near" else gaps[1 + (variant // 4) % 3])
            )
            axial_noise = (
                0.00002
                if task == "screw"
                else (0.00003 if task == "retainer" and family == "partial" else NOISE["approach_axis_m"])
            )
            axis_delta = rng.uniform(-axial_noise, axial_noise)
            gap += axis_delta
            position = goal + gap * axis
            if family == "near":
                lateral_delta = np.r_[rng.uniform(-0.0005, 0.0005, 2), 0.0]
                # Remove the axial component to keep height and lateral noise independent.
                local_axis = np.asarray(configuration()["assembly_plan"]["extractions"][task], float)
                local_axis /= np.linalg.norm(local_axis)
                lateral_delta -= np.dot(lateral_delta, local_axis) * local_axis
                position += Rb @ lateral_delta
            rotation = Rb @ self.target_rotations[task]
            if task == "screw":
                bearing, quaternion = bearing_pose(m, d, self.meta, preload_turns=0.0)
                seated = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
                axis = seated[:, 2]
                position = bearing + gap * axis
                rotation = seated @ Rotation.from_euler("z", gap * 2 * np.pi / PITCH).as_matrix()
            set_pose(m, d, task, position, rotation)
        elif family == "inhand":
            position = d.body(task).xpos.copy()
            position[2] += 0.04
            set_pose(m, d, task, position, d.body(task).xmat.reshape(3, 3).copy())
        mujoco.mj_forward(m, d)
        obj = d.body(task)
        Robj = obj.xmat.reshape(3, 3).copy()
        rotation_delta = np.zeros(3)
        if task != "screw":
            rotation_delta = (
                rng.uniform(-1, 1, 3)
                * np.deg2rad(NOISE["grasp_rotation_deg"])
                * (0.2 if family == "partial" else 1.0)
            )
            Robj = Rotation.from_rotvec(rotation_delta).as_matrix() @ Robj
            set_pose(m, d, task, obj.xpos.copy(), Robj)
            mujoco.mj_forward(m, d)
        heights = (
            (0.004, 0.006, 0.002)
            if family == "pickup"
            else ((0.004, 0.0, 0.002) if family == "inhand" else (0.004, 0.006, 0.008))
        )
        if task == "screw" and family == "partial":
            heights = (0.0055, 0.006, 0.0065)
        center_grasp = obj.xpos + Robj @ local + np.array([0, 0, heights[(variant // 4) % 3]])
        sides = sorted(
            ("left", "right"),
            key=lambda s: np.linalg.norm(m.body_pos[m.body(s + "_base").id, :2] - center_grasp[:2]),
        )
        side = sides[(variant // 2) % 2] if side_override is None else side_override
        yaw = (0.0, np.pi, -0.5 * np.pi, 0.5 * np.pi)[variant % 4] + theta
        if task == "retainer":
            yaw = (-0.5 * np.pi if variant % 2 else 0.5 * np.pi) + theta
        Rs = down_rotation(yaw)
        if (variant // 12) % 2:
            # A side approach keeps large fingers clear of high fixtures.
            forward = np.r_[center_grasp[:2] - m.body_pos[m.body(side + "_base").id, :2], 0.0]
            forward /= np.linalg.norm(forward)
            span = np.array([0.0, 0.0, 1.0]) if task == "retainer" else np.cross(forward, [0.0, 0.0, 1.0])
            if variant % 2:
                span = -span
            Rs = np.column_stack((forward, np.cross(span, forward), span))
        bounds = np.asarray(self.meta[task]["bounds"])
        corners = np.array(np.meshgrid(*bounds.T, indexing="ij")).reshape(3, -1).T @ Robj.T
        width = float(np.ptp(corners @ Rs[:, 2]))
        if task == "screw":
            width = 0.0052
        if narrow_wall:
            width = 0.003
        if rail_width is not None:
            width = rail_height if (variant // 12) % 2 else rail_width
        pad_offset = 0.0 if task in ("horn", "screw") else 0.004
        point = center_grasp - 0.5 * width * Rs[:, 2] + pad_offset * Rs[:, 1]
        grasp_noise = 0.00001 if task == "screw" else NOISE["grasp_translation_m"]
        if family == "partial":
            grasp_noise *= 0.2
        grasp_delta = rng.uniform(-grasp_noise, grasp_noise, 3)
        point += Rs @ grasp_delta
        start = 0 if side == "left" else 6
        if task == "screw" and family in ("near", "partial"):
            d.qpos[start : start + 5] = self.circular_grasp_ik(
                d, side, center_grasp + Rs @ grasp_delta, width
            )
        else:
            d.qpos[start : start + 5] = self.grasp_ik(d, side, point, Rs)
        aperture = float(np.clip(-0.199 + 13 * width, -0.19, 0.7))
        d.qpos[start + 5] = 0.75 if family == "pickup" else aperture
        d.qvel[:] = 0.0
        d.ctrl[:] = d.qpos[:12]
        d.ctrl[start + 5] = -0.23
        mujoco.mj_kinematics(m, d)
        site = d.site(side + "_gripperframe")
        achieved = site.xmat.reshape(3, 3)
        correction = achieved @ Rs.T
        old_position = d.body(task).xpos.copy()
        old_rotation = d.body(task).xmat.reshape(3, 3).copy()
        if family == "pickup":
            shift = np.r_[site.xpos[:2] - point[:2], 0.0]
            set_pose(m, d, task, old_position + shift, old_rotation)
            correction_angle = 0.0
        elif preserve_part_pose or (task == "screw" and family in ("near", "partial")):
            # Keep the proposed part pose; improve the grasp instead.
            shift = np.zeros(3)
            correction_angle = 0.0
        else:
            corrected_position = site.xpos + correction @ (old_position - point)
            shift = corrected_position - old_position
            correction_angle = float(Rotation.from_matrix(correction).magnitude())
            set_pose(m, d, task, corrected_position, correction @ old_rotation)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = self.compensate_arm_gravity(d, d.ctrl)
        mujoco.mj_forward(m, d)
        return dict(
            state=get_state(m, d),
            controls=d.ctrl.copy(),
            side=side,
            task=task,
            family=family,
            seed=int(seed),
            variant=variant,
            grasp_side_override=side_override,
            requested_axis_gap_m=axis_gap,
            preserve_part_pose=preserve_part_pose,
            contact_aperture_rad=aperture,
            predecessors=receiver["predecessors"],
            receiver_preparation=receiver["preparation"],
            noise=dict(
                xy_m=delta[:2].tolist(),
                yaw_rad=theta,
                independent_parts=independent_noise,
                grasp_translation_local_m=grasp_delta.tolist(),
                part_rotation_rad=rotation_delta.tolist(),
                approach_axial_m=axis_delta,
                approach_lateral_base_m=lateral_delta.tolist(),
                kinematic_initialization_shift_m=shift.tolist(),
                kinematic_initialization_rotation_rad=correction_angle,
            ),
            prepared=True,
        )

    def calibrate_yoke_grasp(self, item, forward_m=0.008, pad_target_m=-0.000005):
        """Prepare opposing pad contact without moving any part or receiver."""
        if item["task"] != "yoke" or item["family"] not in ("near", "partial"):
            raise ValueError("Yoke mating grasp calibration requires a mating proposal")
        m = self.m
        side = item["side"]
        start = 0 if side == "left" else 6
        base = mujoco.MjData(m)
        mujoco.mj_setState(m, base, np.asarray(item["state"]), STATE_SPEC)
        mujoco.mj_kinematics(m, base)
        origin = base.site(side + "_gripperframe").xpos.copy()
        rotation = base.site(side + "_gripperframe").xmat.reshape(3, 3).copy()
        work = mujoco.MjData(m)
        work.time = base.time
        jaw_names = (side + "_gripper", side + "_moving_jaw_so101_v1")
        limits = m.jnt_range[m.joint(side + "_gripper").id]
        opened = min(0.65, float(limits[1]) - 1e-6)
        closed = float(limits[0]) + 1e-6
        # Native SDF contacts disappear just outside the surface; a tiny
        # measured preload avoids bisecting a discontinuous no-contact sentinel.
        target = float(pad_target_m)

        def evaluate(span, aperture):
            work.qpos[:] = base.qpos
            work.qvel[:] = base.qvel
            try:
                work.qpos[start : start + 5] = self.grasp_ik(
                    base, side, origin + forward_m * rotation[:, 0] + span * rotation[:, 2], rotation
                )
            except ValueError:
                return None
            work.qpos[start + 5] = aperture
            mujoco.mj_fwdPosition(m, work)
            distances = {name: [] for name in jaw_names}
            for contact in work.contact:
                names = {m.body(int(body)).name for body in m.geom_bodyid[contact.geom]}
                if "yoke" in names:
                    for name in names.intersection(distances):
                        distances[name].append(float(contact.dist))
            return [min(distances[name], default=0.001) for name in jaw_names]

        def bisect(fun, lower, upper):
            a, z = fun(lower), fun(upper)
            if a is None or z is None or (a - target) * (z - target) > 0:
                return None
            for _ in range(20):
                middle = 0.5 * (lower + upper)
                value = fun(middle)
                if value is None:
                    return None
                if (a - target) * (value - target) <= 0:
                    upper = middle
                else:
                    lower = middle
                    a = value
            return 0.5 * (lower + upper)

        def gap(span, aperture, index):
            values = evaluate(span, aperture)
            return None if values is None else values[index]

        samples = []
        for span in np.linspace(-0.006, 0.006, 49):
            value = gap(float(span), opened, 0)
            if value is not None:
                samples.append((float(span), value))
        brackets = [
            (a[0], z[0]) for a, z in zip(samples, samples[1:]) if (a[1] - target) * (z[1] - target) <= 0
        ]
        if not brackets:
            return None, dict(reason="yoke_fixed_pad_first_touch_unreachable")
        span = bisect(lambda x: gap(x, opened, 0), *brackets[0])
        if span is None:
            return None, dict(reason="yoke_fixed_pad_first_touch_unreachable")
        aperture = bisect(lambda q: gap(span, q, 1), closed, opened)
        if aperture is None:
            return None, dict(reason="yoke_moving_pad_first_touch_unreachable")
        distances = evaluate(span, aperture)
        if distances is None or max(abs(value) for value in distances) > 0.0001:
            return None, dict(reason="yoke_first_touch_not_bilateral", pad_distances_m=distances)
        if max(0.0, -min(work.contact.dist, default=0.0)) > DEPTH:
            contact = min(work.contact, key=lambda c: c.dist)
            return None, dict(
                reason="yoke_first_touch_other_collision",
                depth_m=-float(contact.dist),
                geom_pair=[m.geom(int(i)).name for i in contact.geom],
            )
        work.ctrl[:] = work.qpos[:12]
        work.ctrl[start + 5] = -0.23
        mujoco.mj_forward(m, work)
        work.ctrl[:] = self.compensate_arm_gravity(work, work.ctrl)
        prepared = item | dict(
            state=get_state(m, work),
            controls=work.ctrl.copy(),
            contact_aperture_rad=aperture,
            grasp_calibration="native-pad-first-touch-v1",
            grasp_forward_offset_m=forward_m,
            grasp_span_offset_m=span,
            grasp_pad_target_m=target,
            grasp_pad_distances_m=distances,
        )
        return prepared, dict(prepared_only=True, pad_distances_m=distances)

    def inspect(self, d, item):
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return "nonfinite_state", {}
        depth = max(0.0, -min(d.contact.dist, default=0.0))
        if depth > DEPTH:
            contact = min(d.contact, key=lambda c: c.dist)
            return "native_initial_penetration", dict(
                depth_m=depth, geom_pair=[self.m.geom(int(g)).name for g in contact.geom]
            )
        overlaps = material_overlaps(d)
        if max(overlaps.values(), default=0.0) > DEPTH:
            return "sampled_material_overlap", overlaps
        errors = placement_errors(d, item["task"], self.meta, self.targets)
        if any(errors[n] >= 0.001 for n in item["predecessors"]):
            return "predecessor_not_retained", errors
        return None, overlaps

    def lift_pickup(self, item, seed):
        """Prepare an in-hand start by actually closing and lifting a noisy pickup."""
        item = item.copy()
        item["state"] = np.asarray(item["state"], float)
        item["controls"] = np.asarray(item["controls"], float)
        checked, receipt = self.qualify(item, warmup=False)
        if checked is None:
            return None, dict(reason="pickup_replay_failed", detail=receipt)
        env = self.env
        d = env.data[0]
        side = item["side"]
        start = 0 if side == "left" else 6
        rng = np.random.default_rng(seed)
        height = float(rng.uniform(0.03, 0.045))
        target = d.site(side + "_gripperframe").xpos.copy() + [0, 0, height]
        try:
            joints = ik(self.m, d.qpos, side, target)
        except ValueError:
            return None, dict(reason="lift_unreachable")
        begin = env.last_ctrl[0].copy()
        end = begin.copy()
        end[start : start + 5] = joints
        end[start + 5] = -0.23
        gentle = (
            (start + 5, item["contact_aperture_rad"], 0.15 if item["task"] == "screw" else 0.5)
            if item["task"] in ("screw", "horn")
            else None
        )
        initial = d.site(side + "_gripperframe").xmat.reshape(3, 3).T @ (
            d.body(item["task"]).xpos - d.site(side + "_gripperframe").xpos
        )
        preparation_slip = 0.0
        for tick in range(60):
            u = min(1.0, (tick + 1) / 50)
            controls = begin + (3 * u * u - 2 * u * u * u) * (end - begin)
            _, _, info = env.step(holding_action(env, controls[None], gentle))
            if info["safety"][0]:
                return None, dict(reason="lift_safety", tick=tick)
            d = env.data[0]
            relative = d.site(side + "_gripperframe").xmat.reshape(3, 3).T @ (
                d.body(item["task"]).xpos - d.site(side + "_gripperframe").xpos
            )
            preparation_slip = max(preparation_slip, float(np.linalg.norm(relative - initial)))
            if tick % 10 == 9:
                reason, detail = self.inspect(d, item)
                if reason:
                    return None, dict(reason=reason, detail=detail, phase="lift")
        lifted = item | dict(
            state=env.state[0].copy(),
            controls=end,
            seed=int(seed),
            family="inhand",
            source_pickup_seed=item["seed"],
            preparation_motion="Native bounded-action pickup and lift",
            lift_height_m=height,
            preparation_grasp_displacement_m=preparation_slip,
        )
        return self.qualify(lifted, warmup=False)

    def place_grasp(self, item, family, seed, side=None, axis_gap=None):
        """Solve a measured grasp at the existing receiver layout."""
        if family not in ("near", "partial"):
            raise ValueError("Expected mating family")
        m = self.m
        task = item["task"]
        source_side = item["side"]
        side = side or source_side
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, np.asarray(item["state"]), STATE_SPEC)
        d.ctrl[:] = np.asarray(item["controls"])
        mujoco.mj_forward(m, d)
        source = d.site(source_side + "_gripperframe")
        S = source.xmat.reshape(3, 3)
        rel_p = S.T @ (d.body(task).xpos - source.xpos)
        rel_R = S.T @ d.body(task).xmat.reshape(3, 3)
        start = 0 if side == "left" else 6
        source_start = 0 if source_side == "left" else 6
        grasp_angle = float(d.qpos[source_start + 5])
        anchor = d.body("base").xpos.copy()
        base_R = d.body("base").xmat.reshape(3, 3).copy()
        targets = self.targets[task] - self.targets["base"]
        axis = np.array(configuration()["assembly_plan"]["extractions"][task], float)
        axis /= np.linalg.norm(axis)
        rng = np.random.default_rng(seed)
        gap = {
            "motor": (0.04, 0.025),
            "horn": (0.012, 0.002),
            "screw": (0.007, 0.0025),
            "retainer": (0.042, 0.005),
            "yoke": (0.045, 0.022),
        }[task][family == "partial"]
        if axis_gap is not None:
            gap = float(axis_gap)
        gap += rng.uniform(-0.00015, 0.00015) if task != "screw" else rng.uniform(-0.00002, 0.00002)
        lateral = np.zeros(3)
        if family == "near":
            lateral = rng.uniform(-0.0004, 0.0004, 3)
            lateral -= axis * (lateral @ axis)
        screw_pose = None
        if task == "screw":
            p, q = bearing_pose(m, d, self.meta, preload_turns=0.0)
            screw_pose = (base_R.T @ (p - anchor), base_R.T @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        ids = [
            m.joint(side + "_" + n).id
            for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        ranges = m.jnt_range[ids]
        lower = ranges[:, 0]
        upper = ranges[:, 1]
        if task == "screw":
            lower = np.r_[lower, -np.pi]
            upper = np.r_[upper, np.pi]

        def desired(x):
            R = base_R
            position = anchor
            if task == "screw":
                phase = x[5]
                height = gap + (PITCH * phase / (2 * np.pi) if family == "partial" else 0.0)
                p = position + R @ (screw_pose[0] + [0, 0, height])
                Q = R @ screw_pose[1] @ Rotation.from_euler("z", 2 * np.pi * gap / PITCH + phase).as_matrix()
                return p, Q
            return position + R @ (targets + gap * axis + lateral), R @ self.target_rotations[task]

        def residual(x):
            scratch.qpos[start : start + 5] = x[:5]
            mujoco.mj_kinematics(m, scratch)
            hand = scratch.site(side + "_gripperframe")
            R = hand.xmat.reshape(3, 3)
            p = hand.xpos + R @ rel_p
            Q = R @ rel_R
            target, rotation = desired(x)
            return np.r_[p - target, 0.04 * Rotation.from_matrix(rotation @ Q.T).as_rotvec()]

        solution = None
        best_error = None
        # Search arm branches without moving the receiving assembly.
        for pan_delta in (0.0, -0.35, 0.35):
            x = d.qpos[start : start + 5].copy()
            x[0] += pan_delta
            if task == "screw":
                x = np.r_[x, 0.0]
            trial = least_squares(
                residual, np.clip(x, lower + 1e-7, upper - 1e-7), bounds=(lower, upper), max_nfev=250
            )
            error = residual(trial.x)
            metrics = [float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:6]) / 0.04)]
            if best_error is None or sum(metrics) < sum(best_error):
                best_error = metrics
            position_limit, angle_limit = (
                (0.004, 0.25) if family == "near" else (0.0005, 0.14 if task == "yoke" else 0.08)
            )
            if metrics[0] < position_limit and metrics[1] < angle_limit:
                solution = trial.x
                break
        if solution is None:
            return None, dict(
                reason="measured_grasp_mating_unreachable",
                best_position_error_m=best_error[0],
                best_rotation_error_rad=best_error[1],
            )
        if side != source_side:
            # The source hand is reset to its normal parking pose during preparation.
            parked = mujoco.MjData(m)
            home(m, parked)
            d.qpos[source_start : source_start + 6] = parked.qpos[source_start : source_start + 6]
        d.qpos[start : start + 5] = solution[:5]
        d.qpos[start + 5] = grasp_angle
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        hand = d.site(side + "_gripperframe")
        R = hand.xmat.reshape(3, 3)
        set_pose(m, d, task, hand.xpos + R @ rel_p, R @ rel_R)
        d.ctrl[:] = d.qpos[:12]
        d.ctrl[start + 5] = -0.23
        mujoco.mj_forward(m, d)
        d.ctrl[:] = self.compensate_arm_gravity(d, d.ctrl)
        mujoco.mj_forward(m, d)
        candidate = item | dict(
            state=get_state(m, d),
            controls=d.ctrl.copy(),
            side=side,
            family=family,
            seed=int(seed),
            source_inhand_seed=item["seed"],
            preparation_base_yaw_rad=0.0,
            preparation_base_xy_m=[0.0, 0.0],
            requested_axis_gap_m=gap,
            preparation_motion="Prepared measured-grasp placement; insertion path unverified",
        )
        return self.qualify(candidate)

    def alternate_grasp(self, item, angle, seed):
        """Explicitly prepare a rotation around the measured pinch line, then settle."""
        d = mujoco.MjData(self.m)
        mujoco.mj_setState(self.m, d, np.asarray(item["state"]), STATE_SPEC)
        d.ctrl[:] = np.asarray(item["controls"])
        mujoco.mj_forward(self.m, d)
        task, side = item["task"], item["side"]
        hand = d.site(side + "_gripperframe")
        R = hand.xmat.reshape(3, 3)
        points = []
        jaws = {side + "_gripper", side + "_moving_jaw_so101_v1"}
        for c in d.contact:
            names = {self.m.body(int(i)).name for i in self.m.geom_bodyid[c.geom]}
            if task in names and names.intersection(jaws):
                points.append(c.pos.copy())
        if not points:
            return None, dict(reason="missing_measured_pinch_line")
        pivot = np.mean(points, axis=0)
        rotation = Rotation.from_rotvec(R[:, 2] * angle).as_matrix()
        body = d.body(task)
        set_pose(self.m, d, task, pivot + rotation @ (body.xpos - pivot), rotation @ body.xmat.reshape(3, 3))
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, d)
        candidate = item | dict(
            state=get_state(self.m, d),
            controls=d.ctrl.copy(),
            seed=int(seed),
            source_inhand_seed=item["seed"],
            preparation_pinch_rotation_rad=angle,
        )
        return self.qualify(candidate)

    def noisy_grasp(
        self, template, seed, split, validate=True, arm_noise=0.003, workcell_xy=0.003, workcell_yaw=2.0
    ):
        """Resample a fresh scene around fixed measured grasp parameters."""
        task = template["task"]
        side = template["side"]
        m = self.m
        rng = np.random.default_rng(seed)
        reference = mujoco.MjData(m)
        mujoco.mj_setState(m, reference, np.asarray(template["state"]), STATE_SPEC)
        mujoco.mj_kinematics(m, reference)
        hand = reference.site(side + "_gripperframe")
        R = hand.xmat.reshape(3, 3)
        rel_p = R.T @ (reference.body(task).xpos - hand.xpos)
        rel_R = R.T @ reference.body(task).xmat.reshape(3, 3)
        source = self.receiver(task)
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, source["state"], STATE_SPEC)
        mujoco.mj_kinematics(m, d)
        anchor = d.body("base").xpos.copy()
        delta = np.r_[rng.uniform(-workcell_xy, workcell_xy, 2), 0.0]
        yaw = rng.uniform(-workcell_yaw, workcell_yaw) * np.pi / 180
        Q = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
        independent = {}
        for name in (*PARTS, "driver"):
            body = d.body(name)
            p = anchor + Q @ (body.xpos - anchor) + delta
            rotation = Q @ body.xmat.reshape(3, 3)
            if name not in ("base", *source["predecessors"], task):
                shift = np.r_[rng.uniform(-0.003, 0.003, 2), 0.0]
                angle = rng.uniform(-2, 2) * np.pi / 180
                p += shift
                rotation = Rotation.from_rotvec([0, 0, angle]).as_matrix() @ rotation
                independent[name] = dict(xy_m=shift[:2].tolist(), yaw_rad=angle)
            set_pose(m, d, name, p, rotation)
        start = 0 if side == "left" else 6
        perturb = rng.uniform(-arm_noise, arm_noise, 5)
        joints = [
            m.joint(side + "_" + n).id
            for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        bounds = m.jnt_range[joints]
        d.qpos[start : start + 5] = np.clip(
            reference.qpos[start : start + 5] + perturb, bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6
        )
        d.qpos[start + 5] = reference.qpos[start + 5]
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        hand = d.site(side + "_gripperframe")
        R = hand.xmat.reshape(3, 3)
        set_pose(m, d, task, hand.xpos + R @ rel_p, R @ rel_R)
        d.ctrl[:] = d.qpos[:12]
        d.ctrl[start + 5] = float(template["controls"][start + 5])
        mujoco.mj_forward(m, d)
        d.ctrl[:] = self.compensate_arm_gravity(d, d.ctrl)
        mujoco.mj_forward(m, d)
        item = dict(
            state=get_state(m, d),
            controls=d.ctrl.copy(),
            task=task,
            family="inhand",
            seed=int(seed),
            split=split,
            side=side,
            contact_aperture_rad=float(template["contact_aperture_rad"]),
            predecessors=source["predecessors"],
            receiver_preparation=source["preparation"],
            prepared=True,
            grasp_template_seed=int(template["seed"]),
            preparation_motion="Independent noisy scene using fixed measured grasp parameters",
            noise=dict(
                xy_m=delta[:2].tolist(),
                yaw_rad=yaw,
                independent_parts=independent,
                arm_joint_rad=perturb.tolist(),
            ),
        )
        return self.qualify(item) if validate else (item, dict(prepared_only=True))

    def fixture_partial(self, task, seed, variant=0):
        """Prepare a partially engaged part carried by its real mating geometry."""
        rng = np.random.default_rng(seed)
        source = self.receiver(task)
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, source["state"], STATE_SPEC)
        d.ctrl[:] = source["controls"]
        mujoco.mj_forward(m, d)
        theta = rng.uniform(-2, 2) * np.pi / 180
        Q = Rotation.from_rotvec([0, 0, theta]).as_matrix()
        anchor = d.body("base").xpos.copy()
        shift = np.r_[rng.uniform(-0.003, 0.003, 2), 0.0]
        for name in ("base", *source["predecessors"]):
            body = d.body(name)
            set_pose(m, d, name, anchor + shift + Q @ (body.xpos - anchor), Q @ body.xmat.reshape(3, 3))
        mujoco.mj_forward(m, d)
        base = d.body("base")
        R = base.xmat.reshape(3, 3)
        if task == "screw":
            p, q = bearing_pose(m, d, self.meta, preload_turns=0.0)
            R0 = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            gap = (0.0006, 0.001, 0.0015)[variant % 3] + rng.uniform(-0.00002, 0.00002)
            set_pose(
                m,
                d,
                task,
                p + gap * R0[:, 2],
                R0 @ Rotation.from_euler("z", 2 * np.pi * gap / PITCH).as_matrix(),
            )
        else:
            axis = np.array(configuration()["assembly_plan"]["extractions"][task], float)
            axis /= np.linalg.norm(axis)
            gap = (0.018, 0.022, 0.026)[variant % 3] + rng.uniform(-0.0002, 0.0002)
            p = base.xpos + R @ (self.targets[task] - self.targets["base"] + gap * axis)
            set_pose(m, d, task, p, R @ self.target_rotations[task])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        item = dict(
            state=get_state(m, d),
            controls=d.ctrl.copy(),
            task=task,
            family="partial",
            seed=int(seed),
            side="left",
            contact_aperture_rad=0.65,
            predecessors=source["predecessors"],
            support_mode="fixture_supported",
            preparation_ticks=25,
            prepared=True,
            receiver_preparation=source["preparation"],
            noise=dict(xy_m=shift[:2].tolist(), yaw_rad=theta, axis_gap_m=gap),
            preparation_motion="Free native settling; real thread/fixture support, no attachment",
        )
        return self.qualify(item)

    def stage_check(self, d, item, contact, initial=False):
        task, family = item["task"], item["family"]
        fixture = item.get("support_mode") == "fixture_supported"
        if family == "loose":
            if contact["table_force_N"] <= 0.0001:
                return "loose_not_on_table"
            if contact["robot_force_N"] > 0.001 or contact["receiver_force_N"] > 0.001:
                return "loose_already_held_or_engaged"
            return None
        if family == "pickup":
            return "pickup_not_on_table" if initial and contact["table_force_N"] <= 0.0001 else None
        if initial and not fixture and not contact["both_jaws_touching"]:
            return "held_start_missing_jaw_contact"
        if fixture and (family != "partial" or contact["robot_force_N"] > 0.001):
            return "invalid_fixture_support"
        if family == "inhand" and contact["external_support_force_N"] > 0.001:
            return "inhand_not_free"
        if family == "inhand":
            low = float("inf")
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] != self.m.body(task).id or self.m.geom_group[g] != 2:
                    continue
                mesh = int(self.m.geom_dataid[g])
                if mesh < 0:
                    continue
                start, count = int(self.m.mesh_vertadr[mesh]), int(self.m.mesh_vertnum[mesh])
                vertices = self.m.mesh_vert[start : start + count]
                low = min(
                    low, float((vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
                )
            if not np.isfinite(low) or low < 0.001:
                return "inhand_not_airborne"
        if family == "near" and contact["external_support_force_N"] > 0.001:
            return "near_already_engaged"
        if family == "near" and task == "yoke" and yoke_envelope_overlap(d) > 0.0002:
            return "near_already_inside_fork"
        if family == "partial":
            base = d.body("base")
            R = base.xmat.reshape(3, 3)
            local = R.T @ (d.body(task).xpos - base.xpos) - (self.targets[task] - self.targets["base"])
            axis = np.array(configuration()["assembly_plan"]["extractions"][task], float)
            axis /= np.linalg.norm(axis)
            if task == "screw":
                bearing, _ = bearing_pose(self.m, d, self.meta, preload_turns=0.0)
                R = d.body("motor").xmat.reshape(3, 3)
                local = R.T @ (d.body(task).xpos - bearing)
                axis = np.array([0.0, 0.0, 1.0])
            axial = float(local @ axis)
            lateral = float(np.linalg.norm(local - axial * axis))
            limit = 0.0003 if task == "screw" else 0.0015
            if lateral > limit:
                return "partial_lateral_misalignment"
            if axial < 0.0002:
                return "partial_already_seated"
            relative = (R if task == "screw" else R @ self.target_rotations[task]).T @ d.body(
                task
            ).xmat.reshape(3, 3)
            angle = (
                float(np.arccos(np.clip(relative[2, 2], -1, 1)))
                if task == "screw"
                else float(Rotation.from_matrix(relative).magnitude())
            )
            if angle > (0.14 if task == "yoke" else 0.08):
                return "partial_angular_misalignment"
            if task == "screw":
                if not thread_engagement(self.m, d)["engaged"]:
                    return "partial_thread_not_engaged"
            elif task == "yoke":
                if yoke_envelope_overlap(d) < FORK_ENTRY_MARGIN:
                    return "partial_not_inside_fork"
            elif initial and contact["receiver_force_N"] <= 0.001:
                return "partial_not_engaged"
        return None

    def ready_corridor(self, d, item):
        base = d.body("base")
        rotation = base.xmat.reshape(3, 3)
        target = base.xpos + rotation @ (self.targets[item["task"]] - self.targets["base"])
        axis = rotation @ np.asarray(configuration()["assembly_plan"]["extractions"][item["task"]], float)
        axis /= np.linalg.norm(axis)
        return target, target + 0.08 * axis

    @staticmethod
    def ready_sides(item):
        return (
            ["left", "right"]
            if item["family"] == "loose" or item.get("support_mode") == "fixture_supported"
            else ["right" if item["side"] == "left" else "left"]
        )

    def prepare_ready(self, item):
        """Proposal-only placement; saved-state qualification never calls this."""
        from .ready_arm import prepare_ready_arms

        d = mujoco.MjData(self.m)
        mujoco.mj_setState(self.m, d, np.asarray(item["state"]), STATE_SPEC)
        d.ctrl[:] = np.asarray(item["controls"])
        mujoco.mj_forward(self.m, d)
        start, end = self.ready_corridor(d, item)
        sides = self.ready_sides(item)
        receipt = prepare_ready_arms(
            self.m,
            d,
            np.random.default_rng(int(item["seed"]) + 7000000000),
            active_side=None if len(sides) == 2 else item["side"],
            corridor_start=start,
            corridor_end=end,
        )
        if not receipt.get("ok"):
            return None, dict(reason="ready_arm_proposal_failed", detail=receipt)
        compensated = self.compensate_arm_gravity(d, d.ctrl)
        for side in sides:
            start = 0 if side == "left" else 6
            d.ctrl[start : start + 6] = compensated[start : start + 6]
        return item | dict(
            state=get_state(self.m, d),
            controls=d.ctrl.copy(),
            ready_arm=dict(sides=sides, preparation=receipt),
        ), receipt

    def check_ready(self, d, item):
        from .ready_arm import validate_ready_arms

        start, end = self.ready_corridor(d, item)
        return validate_ready_arms(self.m, d, self.ready_sides(item), corridor_start=start, corridor_end=end)

    def qualify(self, item, warmup=True, replay_seconds=None):
        env = self.env
        task = item["task"]
        family = item["family"]
        side = item["side"]
        env.set_states([0], item["state"][None], [ASSEMBLY_TASKS.index(task)])
        reason, material = self.inspect(env.data[0], item)
        if reason:
            return None, dict(reason=reason, values=material)
        ready_checks = []
        if "ready_arm" in item:
            check = self.check_ready(env.data[0], item)
            ready_checks.append(check)
            if not check.get("ok"):
                return None, dict(reason="ready_arm_initial_invalid", detail=check)
        controls = item["controls"][None]
        fixture = item.get("support_mode") == "fixture_supported"
        grip_index = 5 if side == "left" else 11
        gentle = (
            (grip_index, item["contact_aperture_rad"], 0.15 if task == "screw" else 0.5)
            if task in ("screw", "horn") and family != "loose"
            else None
        )
        action = lambda: holding_action(env, controls, gentle)
        if warmup and family not in ("loose", "pickup"):
            for _ in range(item.get("preparation_ticks", 6)):
                _, _, info = env.step(action())
                if info["safety"][0]:
                    return None, dict(reason="warmup_safety")
                if info["success"][0]:
                    return None, dict(reason="already_completed_during_preparation")
            item = item | dict(state=env.state[0].copy())
            env.set_states([0], item["state"][None], [ASSEMBLY_TASKS.index(task)])
            if "ready_arm" in item:
                check = self.check_ready(env.data[0], item)
                ready_checks.append(check)
                if not check.get("ok"):
                    return None, dict(reason="ready_arm_post_warmup_invalid", detail=check)
        d = env.data[0]
        reason, material = self.inspect(d, item)
        if reason:
            return None, dict(reason=reason, values=material)
        contact = contact_state(self.m, d, task, side, item["predecessors"])
        reason = self.stage_check(d, item, contact, initial=True)
        if reason:
            return None, dict(reason=reason)
        if family == "near" and contact["receiver_force_N"] > 0.001:
            return None, dict(reason="near_already_engaged")
        if (
            family == "near"
            and max(
                (v for k, v in material.items() if task in k.split("/") and not k.endswith("/table")),
                default=0.0,
            )
            > 0.00001
        ):
            return None, dict(reason="near_material_engaged")
        if family == "partial" and task not in ("screw", "yoke") and contact["receiver_force_N"] <= 0.001:
            return None, dict(reason="partial_not_engaged")
        if family == "inhand" and contact["receiver_force_N"] > 0.001:
            return None, dict(reason="inhand_supported_by_receiver")
        support = lambda: d.body("base") if fixture else d.site(side + "_gripperframe")
        initial = support().xmat.reshape(3, 3).T @ (d.body(task).xpos - support().xpos)
        initial_rotation = support().xmat.reshape(3, 3).T @ d.body(task).xmat.reshape(3, 3)
        contacts = []
        engagement = []
        peak = 0.0
        peak_slip = 0.0
        max_angle = 0.0
        max_predecessor = 0.0
        ticks = round(
            (replay_seconds if replay_seconds is not None else (1.28 if family == "pickup" else 1.0))
            / env.ecfg.control_dt
        )
        for tick in range(ticks):
            _, _, info = env.step(action())
            d = env.data[0]
            if info["safety"][0]:
                return None, dict(reason="replay_safety", tick=tick)
            if info["success"][0]:
                return None, dict(reason="already_completed_start", tick=tick)
            contact = contact_state(self.m, d, task, side, item["predecessors"])
            contacts.append(contact["bilateral"])
            engagement.append(
                thread_engagement(self.m, d)["engaged"]
                if task == "screw"
                else (
                    yoke_envelope_overlap(d) >= FORK_ENTRY_MARGIN
                    if task == "yoke"
                    else contact["receiver_force_N"] > 0.001
                )
            )
            reason = self.stage_check(d, item, contact)
            if reason:
                return None, dict(reason=reason, tick=tick)
            peak = max(peak, max(0.0, -min(d.contact.dist, default=0.0)))
            relative = support().xmat.reshape(3, 3).T @ (d.body(task).xpos - support().xpos)
            peak_slip = max(peak_slip, float(np.linalg.norm(relative - initial)))
            relative_rotation = support().xmat.reshape(3, 3).T @ d.body(task).xmat.reshape(3, 3)
            max_angle = max(
                max_angle, float(Rotation.from_matrix(relative_rotation @ initial_rotation.T).magnitude())
            )
            errors = placement_errors(d, task, self.meta, self.targets)
            max_predecessor = max(
                max_predecessor, max((errors[n] for n in item["predecessors"]), default=0.0)
            )
            if max_predecessor >= 0.001:
                return None, dict(reason="predecessor_drift", error_m=max_predecessor)
            if "ready_arm" in item:
                check = self.check_ready(d, item)
                ready_checks.append(check)
                if not check.get("ok"):
                    return None, dict(reason="ready_arm_replay_invalid", tick=tick, detail=check)
            if tick in (ticks // 2, ticks - 1):
                reason, values = self.inspect(d, item)
                if reason:
                    return None, dict(reason=reason, values=values, tick=tick)
                material = {k: max(v, material.get(k, 0.0)) for k, v in values.items()}
        if family != "loose" and not fixture and sum(contacts[-5:]) < 4:
            return None, dict(reason="bilateral_grasp_lost")
        if family == "partial" and sum(engagement[-5:]) < 4:
            return None, dict(reason="partial_engagement_lost")
        if family not in ("loose", "pickup") and peak_slip > 0.002:
            return None, dict(reason="grasp_slip", slip_m=peak_slip)
        if family not in ("loose", "pickup") and max_angle > 0.1:
            return None, dict(reason="grasp_rotation_slip", angle_rad=max_angle)
        if fixture and (peak_slip > 0.00025 or max_angle > 0.01):
            return None, dict(
                reason="fixture_start_not_stable", translation_m=peak_slip, rotation_rad=max_angle
            )
        return item, dict(
            passed=True,
            candidate_digest=candidate_digest(item),
            ready_arm_checks=ready_checks,
            replay_seconds=ticks * 0.04,
            active_squeeze=family != "loose" and not fixture,
            support_mode="table"
            if family == "loose"
            else ("fixture_supported" if fixture else "bilateral_grasp"),
            validator_identity=self.validation_identity,
            tail_bilateral_samples=int(sum(contacts[-5:])),
            peak_control_sampled_depth_m=peak,
            max_grasp_slip_m=peak_slip,
            predecessor_error_m=max_predecessor,
            max_grasp_rotation_slip_rad=max_angle,
            diagnostic_timestep_s=float(self.m.opt.timestep),
            thread_geometry=thread_engagement(self.m, d) if task == "screw" and family == "partial" else None,
            fork_overlap_m=yoke_envelope_overlap(d) if task == "yoke" else None,
            penetration_limit_m=DEPTH,
            max_sampled_material_overlap_m=max(material.values(), default=0.0),
            material=material,
            material_sampling="Full vertices and triangle centroids at saved, midpoint and endpoint; analytic screw excluded",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parent", type=Path, default=default_bank_directory())
    ap.add_argument("--tasks", default="+".join(ASSEMBLY_TASKS))
    ap.add_argument("--families", default="pickup+inhand+partial+near")
    ap.add_argument("--train-count", type=int, default=8)
    ap.add_argument("--eval-count", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=48)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument(
        "--grasp-source",
        type=Path,
        action="append",
        default=[],
        help="Fixed measured grasp templates; scene noise is resampled independently",
    )
    args = ap.parse_args()
    if set(args.tasks.split("+")) - set(ASSEMBLY_TASKS):
        raise ValueError("Unknown task")
    if set(args.families.split("+")) - set(FAMILIES[1:]):
        raise ValueError("Unknown held/pickup family")
    if min(args.train_count, args.eval_count) < 0 or args.attempts < 1:
        raise ValueError("Invalid sample counts")
    if args.out.exists():
        raise ValueError("Use a new output directory")
    final_out = args.out
    args.out = final_out.with_name("." + final_out.name + ".building")
    if args.out.exists():
        raise ValueError("Incomplete build exists; choose a new version")
    check_bank(args.parent)
    parent_hash = directory_signature(args.parent)
    shutil.copytree(args.parent, args.out)
    (args.out / "family_eval").mkdir()
    source = Path(__file__).read_bytes()
    (args.out / "generator_source.py").write_bytes(source)
    builder = Builder()
    report = dict(
        environment=builder.identity,
        parent_directory_signature=parent_hash,
        validator_identity=builder.validation_identity,
        generator_sha256=hashlib.sha256(source).hexdigest(),
        requested=dict(train=args.train_count, eval=args.eval_count, attempts=args.attempts),
        noise_bounds=NOISE,
        qualification=dict(
            native_depth_m=DEPTH,
            sampled_material_depth_m=DEPTH,
            predecessor_error_m=0.001,
            grasp_slip_m=0.002,
        ),
        coverage={},
        failures=[],
        scope="Prepared curriculum; physical start validity does not establish an insertion path or learned assembly",
    )
    grasp_templates = {task: [] for task in ASSEMBLY_TASKS}
    for path in args.grasp_source:
        if path.is_file():
            value = json.loads(path.read_text())
            records = value if isinstance(value, list) else [value]
        else:
            records = []
            for file in path.glob("*_provenance.json"):
                records.extend(json.loads(file.read_text()))
        for row in records:
            if (
                row["family"] == "inhand"
                and row.get("split") == "train"
                and row.get("support_mode") != "fixture_supported"
            ):
                grasp_templates[row["task"]].append(row)
    try:
        for task in args.tasks.split("+"):
            report["coverage"][task] = {}
            accepted = []
            for family in args.families.split("+"):
                counts = {}
                rejects = Counter()
                best_variant = None
                failed_streak = 0
                for split, wanted, offset in [
                    ("train", args.train_count, 2000000),
                    ("eval", args.eval_count, 4000000),
                ]:
                    rows = []
                    for attempt in range(args.attempts):
                        if len(rows) >= wanted:
                            break
                        seed = (
                            args.seed_offset
                            + offset
                            + 100000 * ASSEMBLY_TASKS.index(task)
                            + 10000 * FAMILIES.index(family)
                            + attempt
                        )
                        try:
                            variant = attempt if best_variant is None else best_variant
                            templates = grasp_templates[task]
                            if task == "screw" and family == "partial":
                                result, receipt = builder.fixture_partial(task, seed, variant)
                            elif family == "inhand" and templates:
                                result, receipt = builder.noisy_grasp(
                                    templates[attempt % len(templates)], seed, split
                                )
                            elif task in ("screw", "yoke") and family == "inhand" and accepted:
                                picks = [
                                    r for r in accepted if r["family"] == "pickup" and r["split"] == "train"
                                ]
                                if not picks:
                                    raise ValueError("no_qualified_pickup_template")
                                mean, receipt = builder.lift_pickup(
                                    picks[attempt % len(picks)], seed + 100000000
                                )
                                if mean is None:
                                    result = None
                                else:
                                    grasp_templates[task].append(mean)
                                    result, receipt = builder.noisy_grasp(mean, seed, split)
                            elif templates and (
                                (task in ("screw", "yoke") and family in ("near", "partial"))
                                or (task == "retainer" and family == "near")
                            ):
                                mean = templates[attempt % len(templates)]
                                item, _ = builder.noisy_grasp(mean, seed + 100000000, split, validate=False)
                                if task == "yoke" and family == "partial":
                                    item, receipt = builder.alternate_grasp(item, -0.175, seed + 200000000)
                                if item is None:
                                    result = None
                                else:
                                    gap = 0.02475 if task == "yoke" and family == "partial" else None
                                    result, receipt = builder.place_grasp(
                                        item,
                                        family,
                                        seed,
                                        "left" if attempt % 2 == 0 else "right",
                                        axis_gap=gap,
                                    )
                            else:
                                item = builder.candidate(task, family, seed, variant)
                                result, receipt = builder.qualify(item)
                        except (ValueError, FloatingPointError) as e:
                            result = None
                            receipt = dict(reason=str(e))
                        if result is None:
                            failed_streak += 1
                            if failed_streak >= 3:
                                best_variant = None
                                failed_streak = 0
                            rejects[receipt["reason"]] += 1
                            report["failures"].append(
                                dict(task=task, family=family, split=split, seed=seed, **receipt)
                            )
                        else:
                            best_variant = variant
                            failed_streak = 0
                            result["qualification"] = receipt
                            result["split"] = split
                            rows.append(result)
                            accepted.append(result)
                            if family == "inhand" and split == "train" and not grasp_templates[task]:
                                grasp_templates[task].append(result)
                            with (args.out / "accepted.jsonl").open("a") as stream:
                                stream.write(
                                    json.dumps(
                                        {
                                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in result.items()
                                        }
                                    )
                                    + "\n"
                                )
                        print(
                            json.dumps(
                                dict(
                                    task=task,
                                    family=family,
                                    split=split,
                                    attempt=attempt,
                                    accepted=len(rows),
                                    result=receipt.get("reason", "pass"),
                                )
                            ),
                            flush=True,
                        )
                        if receipt.get("reason") == "predecessor_material_overlap":
                            break
                    counts[split] = len(rows)
                report["coverage"][task][family] = dict(**counts, rejections=dict(rejects))
                (args.out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
            for split in ("train", "eval"):
                rows = [r for r in accepted if r["split"] == split]
                target = (
                    args.out / f"{task}_train.npz"
                    if split == "train"
                    else args.out / "family_eval" / f"{task}_eval.npz"
                )
                with np.load(args.parent / f"{task}_{split}.npz") as old:
                    states = [old["states"]]
                    families = [old["family"]]
                    seeds = [old["seed"]]
                if rows:
                    states.append(np.stack([r["state"] for r in rows]))
                    families.append(np.array([FAMILIES.index(r["family"]) for r in rows]))
                    seeds.append(np.array([r["seed"] for r in rows]))
                np.savez_compressed(
                    target,
                    states=np.concatenate(states),
                    family=np.concatenate(families),
                    seed=np.concatenate(seeds),
                )
            serial = [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in accepted
            ]
            (args.out / f"{task}_provenance.json").write_text(json.dumps(serial, indent=2) + "\n")
        manifest = json.loads((args.out / "manifest.json").read_text())
        available = set()
        manifest.update(
            parent_directory_signature=parent_hash,
            reset_distribution="loose_plus_qualified_prepared",
            coverage_note=report["scope"],
            noise_bounds=NOISE,
            qualification_report="qualification.json",
            independent_family_evaluation="family_eval",
            requested_families=list(FAMILIES),
            validator_identity=builder.validation_identity,
        )
        for task in ASSEMBLY_TASKS:
            with np.load(args.out / f"{task}_train.npz") as z:
                ids, counts = np.unique(z["family"], return_counts=True)
                manifest["tasks"][task]["train"]["family_counts"] = {
                    str(int(i)): int(c) for i, c in zip(ids, counts)
                }
                manifest["tasks"][task]["train"]["entries"] = len(z["states"])
                available.update(int(i) for i in ids)
        manifest["families"] = [FAMILIES[i] for i in sorted(available)]
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if directory_signature(args.parent) != parent_hash:
            raise RuntimeError("Source bank changed during generation")
        if environment_identity() != builder.identity or validation_identity() != builder.validation_identity:
            raise RuntimeError("Physical or validator source changed during generation")
    finally:
        builder.close()
    args.out.rename(final_out)
    print(json.dumps(report["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    main()
