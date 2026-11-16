"""Photo-directed, explicitly unqualified kinematic reset candidates.

No constraints hold parts during a rollout. This exports review snapshots, not
contact-qualified reset caches; seating offsets and pin registration are inferred.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from assembly_policy.so101.wowrobo_grasp_refinement import GraspRefiner
from assembly_policy.so101.wowrobo_grasps import native_jaw_grasp, outer_wall_contacts
from assembly_policy.so101.wowrobo_layout import (
    ROOT,
    InitialLayout,
    quaternion,
    rz,
)
from assembly_policy.so101.wowrobo_supported_poses import supported_pose

PLAN = ROOT / "configs/wowrobo_reset_sequence.json"


class PhotoResetSequence:
    def __init__(self, assets, plan=PLAN):
        self.layout = InitialLayout(assets)
        self.plan = json.loads(Path(plan).read_text())
        self.model = self.layout.model
        self.side_grasps = {}
        for name in ["base_side_cover", "shoulder_connector"]:
            center, contacts = outer_wall_contacts(self.layout, name)
            self.layout.grips[name]["center"] = center
            spec = native_jaw_grasp(self.model, contacts[1, 0] - contacts[0, 0], 0.006)
            spec["contacts_piece_local_m"] = contacts.tolist()
            self.side_grasps[name] = spec
        self.side_grasps["assembly_pin"] = native_jaw_grasp(self.model, 0.012, 0.006)
        self.side_grasps["assembly_pin"]["pin_grip_interval_local_z_m"] = [0.008, 0.020]
        # Servo poses stay unchanged; fit the actual asymmetric jaw gap around
        # each casing, rather than centering the nominal tip frame in the case.
        for name, servo in [("upper_arm_module", 2), ("forearm_module", 3)]:
            key = f"servo_{servo}_case"
            bounds = np.array(self.layout.manifest["constituents"][key]["bounds_m"])
            center = bounds.mean(0)
            center[2] = bounds[0, 2] + 0.030
            self.layout.grips[name]["center"] = center
            # IMG1612/1609: pinch the exposed front/back Y faces of the black
            # case. The shaft-end X faces are occupied by the printed yoke.
            vertices = self.layout.constituent_vertices[key]
            width = np.ptp(vertices[:, 1])
            spec = native_jaw_grasp(self.model, width, 0.006, half_length=0.022)
            spec["servo"] = servo
            spec["target"] = key
            self.side_grasps[name] = spec
        # IMG1609 support wraps the installed servo-end exterior fork walls;
        # pinching a middle rail obstructs the incoming final module.
        support = self.plan["authoring_support_seeds"]["hold_upper_arm"]
        center, contacts = outer_wall_contacts(
            self.layout,
            "upper_arm_module",
            np.array(support["guide_piece_cad_m"]),
            ["upper_arm_module_printed_0"],
        )
        self.installed_upper_grip = (
            center,
            native_jaw_grasp(
                self.model, np.ptp(contacts[:, 0]), 0.003, half_length=0.010
            ),
        )
        self.upper_support_span = np.array(support["span_piece_local"])
        self.upper_support_approach = np.array(support["approach_piece_local"])
        center, contacts = outer_wall_contacts(
            self.layout,
            "base_servo",
            np.array([0.0, 0.025, 0.025]),
            ["base_servo_printed_0"],
        )
        self.installed_base_grip = (
            center,
            native_jaw_grasp(
                self.model, np.ptp(contacts[:, 0]), 0.002, half_length=0.010
            ),
        )
        self.loose, self.placed = self.layout.sample(0, nominal=True)
        self.rotation = self.layout.base_rotation
        self.origin = self.layout.base_pos

        def piece_mesh(name, printed_only=True, shaft_angle=None):
            pieces = []
            for key in self.layout.manifest["pieces"][name]["constituents"]:
                if printed_only and "printed" not in key:
                    continue
                mesh = trimesh.load_mesh(
                    self.layout.assets
                    / self.layout.manifest["constituents"][key]["mesh"],
                    process=False,
                )
                if shaft_angle is not None and key == "servo_2_case":
                    axis = self.layout.manifest["servo_axes"]["2"]
                    rotation = Rotation.from_rotvec(
                        np.array(axis["axis"]) * shaft_angle
                    ).as_matrix()
                    pivot = np.array(axis["position_m"])
                    transform = np.eye(4)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = pivot - rotation @ pivot
                    mesh.apply_transform(transform)
                pieces.append(mesh)
            return trimesh.util.concatenate(pieces)

        self.resting = {}
        for name, receiver, servo, far_y, sign, heights in [
            (
                "upper_arm_module",
                "base_side_cover",
                2,
                0.060,
                -1,
                [0.025, 0.012, 0.002, 0],
            ),
            (
                "forearm_module",
                "upper_arm_module",
                3,
                -0.048,
                None,
                [0.025, 0.012, 0.002, 0],
            ),
        ]:
            shaft = self.layout.manifest["servo_axes"][str(servo)]
            self.resting[name] = {
                phase: supported_pose(
                    piece_mesh(name, printed_only=False),
                    piece_mesh(
                        receiver,
                        printed_only=False,
                        shaft_angle=self.resting["upper_arm_module"]["seated"][
                            "shaft_angle"
                        ]
                        if receiver == "upper_arm_module"
                        else None,
                    ),
                    self.layout.grips[name]["center"],
                    shaft["position_m"],
                    shaft["axis"],
                    far_y,
                    h,
                    sign,
                )
                for phase, h in zip(
                    ["far_side_supported", "half_inserted", "near_seated", "seated"],
                    heights,
                )
            }
        self.order = [
            "base_side_cover",
            "shoulder_connector",
            "assembly_pin",
            "upper_arm_module",
            "forearm_module",
        ]
        self.refiner = GraspRefiner(
            self.model, self.layout.assets.parent / "grasp-sdf-cache-v1"
        )
        self.support_cache = {}
        self.lookahead = {
            "hold_base_assembly": self.future_obstacles("upper_arm_module"),
            "hold_upper_arm": self.future_obstacles("forearm_module"),
            "hold_connector": self.future_obstacles("assembly_pin"),
        }
        self.previous = {
            side: self.joint_values(self.loose, side) for side in ["left", "right"]
        }

    def future_obstacles(self, name):
        m = self.model
        ghosts = []
        phases = (
            list(self.resting[name])
            if name in self.resting
            else ["aligned", "half_inserted", "near_seated", "seated"]
        )
        for phase in phases:
            data = mujoco.MjData(m)
            data.qpos[:] = self.loose.qpos
            if name == "assembly_pin":
                r, p = self.final_pose(name)
                p = p + np.array(
                    [
                        0,
                        0,
                        {
                            "aligned": 0.032,
                            "half_inserted": 0.011,
                            "near_seated": 0.002,
                            "seated": 0,
                        }[phase],
                    ]
                )
            else:
                rest = self.resting[name][phase]
                receiver_r, receiver_p = (
                    (self.rotation, self.origin)
                    if name == "upper_arm_module"
                    else self.final_pose("upper_arm_module")
                )
                r = receiver_r @ rest["rotation"]
                p = receiver_p + receiver_r @ rest["position"]
                servo = 2 if name == "upper_arm_module" else 3
                data.qpos[m.joint(f"target_servo_{servo}_shaft").qposadr[0]] = rest[
                    "shaft_angle"
                ]
            self.pose(data, name, r, p)
            mujoco.mj_forward(m, data)
            for key in self.layout.manifest["pieces"][name]["constituents"]:
                gid = m.geom(key).id
                ghosts.append(
                    (
                        gid,
                        data.geom_xpos[gid].copy(),
                        data.geom_xmat[gid].reshape(3, 3).copy(),
                    )
                )
        return ghosts

    def joint_values(self, data, side):
        names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        return np.array(
            [data.qpos[self.model.joint(side + "_" + n).qposadr[0]] for n in names]
        )

    def pose(self, data, name, rotation, position):
        address = self.model.joint(name + "_free").qposadr[0]
        data.qpos[address : address + 3] = position
        data.qpos[address + 3 : address + 7] = quaternion(rotation)

    def sag_held_link(self, data, name, degrees):
        """Lower the free link about its shaft while retaining the held case pose."""
        m = self.model
        servo = 2 if name == "upper_arm_module" else 3
        shaft = m.joint(f"target_servo_{servo}_shaft")
        root = m.body(name).id
        case = m.body(f"target_servo_{servo}_pivot").id
        mujoco.mj_forward(m, data)
        before_position, before_rotation = (
            data.xpos[case].copy(),
            data.xmat[case].copy(),
        )
        root_rotation = data.xmat[root].reshape(3, 3).copy()
        root_position = data.xpos[root].copy()
        far = self.resting[name]["far_side_supported"]["far_point"]
        far_before = root_position + root_rotation @ far
        axis = data.xaxis[shaft.id].copy()
        candidates = []
        for angle in [-np.deg2rad(degrees), np.deg2rad(degrees)]:
            candidate = mujoco.MjData(m)
            candidate.qpos[:] = data.qpos
            rotation = Rotation.from_rotvec(axis * angle).as_matrix() @ root_rotation
            self.pose(candidate, name, rotation, root_position)
            candidate.qpos[shaft.qposadr[0]] -= angle
            mujoco.mj_forward(m, candidate)
            position = root_position + before_position - candidate.xpos[case]
            self.pose(candidate, name, rotation, position)
            mujoco.mj_forward(m, candidate)
            if not (
                np.allclose(candidate.xpos[case], before_position, atol=1e-10, rtol=0)
                and np.allclose(
                    candidate.xmat[case], before_rotation, atol=1e-10, rtol=0
                )
            ):
                raise ValueError("Sag must preserve the held servo case pose")
            far_after = position + rotation @ far
            candidates.append((far_after[2], angle, candidate.qpos.copy(), far_after))
        _, angle, qpos, far_after = min(candidates, key=lambda item: item[0])
        data.qpos[:] = qpos
        mujoco.mj_forward(m, data)
        return {
            "piece": name,
            "held_servo_body": m.body(case).name,
            "angle_degrees": float(np.rad2deg(angle)),
            "far_point_piece_local_m": far.tolist(),
            "far_height_before_m": float(far_before[2]),
            "far_height_after_m": float(far_after[2]),
            "held_case_position_world_m": before_position.tolist(),
            "held_case_rotation_world": before_rotation.tolist(),
            "qualification": "Authored gravity-like sag; not a simulated dynamic equilibrium",
        }

    def final_pose(self, name):
        if name in ["upper_arm_module", "forearm_module"]:
            upper = self.resting["upper_arm_module"]["seated"]
            r = self.rotation @ upper["rotation"]
            p = self.origin + self.rotation @ upper["position"]
            if name == "forearm_module":
                final = self.resting[name]["seated"]
                p = p + r @ final["position"]
                r = r @ final["rotation"]
            return r, p
        if name != "assembly_pin":
            return self.rotation.copy(), self.origin.copy()
        # Inferred pivot-axis registration; actual hole/contact fit is NOT established.
        pivot = np.array(self.layout.manifest["servo_axes"]["1"]["position_m"])
        tip = self.origin + self.rotation @ pivot - np.array([0, 0, 0.018])
        return self.rotation @ np.diag([1.0, -1.0, -1.0]), tip + [0, 0, 0.050]

    def grip(self, data, name):
        mujoco.mj_forward(self.model, data)
        servo = {"upper_arm_module": 2, "forearm_module": 3}.get(name)
        b = self.model.body(f"target_servo_{servo}_pivot" if servo else name).id
        r = data.xmat[b].reshape(3, 3)
        return data.xpos[b] + r @ self.layout.grips[name]["center"], r @ (
            np.array([0.0, 1.0, 0.0]) if servo else self.layout.grips[name]["axis"]
        )

    def solve_arm(
        self,
        data,
        side,
        center,
        span,
        approach,
        opening,
        rng,
        grasp=None,
        joint_reference=None,
        approach_weight=0.10,
        horizontal_approach=False,
    ):
        names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        joints = [self.model.joint(side + "_" + n).id for n in names]
        addresses = self.model.jnt_qposadr[joints]
        limits = self.model.jnt_range[joints].copy()
        if joint_reference is not None:
            limits[:, 0] = np.maximum(limits[:, 0], joint_reference - 0.6)
            limits[:, 1] = np.minimum(limits[:, 1], joint_reference + 0.6)
        site = self.model.site(side + "_gripperframe").id
        span = np.asarray(span, float)
        span /= np.linalg.norm(span)
        g = self.model.joint(side + "_gripper").qposadr[0]
        if grasp is not None:
            opening = grasp["opening_radians"]
        data.qpos[g] = opening
        span_axis = grasp["span_site_axis"] if grasp else 1
        offset = np.array(grasp["center_offset_site"] if grasp else [-0.018, 0, 0])

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_forward(self.model, data)
            frame = data.site_xmat[site].reshape(3, 3)
            # Site is the fingertip frame; grasp center is 18mm behind that tip.
            grasp_center = data.site_xpos[site] + frame @ offset
            return np.r_[
                100 * (grasp_center - center),
                2 * (frame[:, span_axis] - span),
                approach_weight * (frame[:, 0] - approach),
                [2 * frame[2, 0]] if horizontal_approach else np.empty(0),
            ]

        start = np.clip(self.previous[side], limits[:, 0] + 1e-7, limits[:, 1] - 1e-7)
        best = None
        starts = [start, np.array([0.0, -0.4, 1.2, -0.8, 0.0])]
        starts += [
            rng.uniform(limits[:, 0] + 0.01, limits[:, 1] - 0.01) for _ in range(10)
        ]
        for q in starts:
            fit = least_squares(
                residual,
                np.clip(q, limits[:, 0] + 1e-7, limits[:, 1] - 1e-7),
                bounds=(limits[:, 0] + 1e-7, limits[:, 1] - 1e-7),
                max_nfev=180,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            err = residual(fit.x)
            cost = np.linalg.norm(err)
            if best is None or cost < best[0]:
                best = (cost, fit.x.copy(), err.copy())
            if (
                np.linalg.norm(err[:3]) < 0.08
                and np.linalg.norm(err[3:6]) < 0.03
                and np.linalg.norm(err[6:]) < 0.03
            ):
                break
        _, q, err = best
        residual(q)
        self.previous[side] = q.copy()
        return {
            "grasp_position_error_m": float(np.linalg.norm(err[:3]) / 100),
            "span_vector_error": float(np.linalg.norm(err[3:6]) / 2),
            "approach_vector_error": float(np.linalg.norm(err[6:]) / 0.10),
            "opening_radians": opening,
            "grasp_calibration": grasp,
            "center_offset_site": offset.tolist(),
            "span_site_axis": span_axis,
            "requested_grip_center": np.asarray(center).tolist(),
            "requested_span": span.tolist(),
            "joint_positions": q.tolist(),
            "qualification": "IK only; contacts, grip forces and clearances NOT qualified",
        }

    def state(self, definition):
        m = self.model
        data = mujoco.MjData(m)
        data.qpos[:] = self.loose.qpos
        data.ctrl[:] = self.loose.ctrl
        task, phase = definition["task"], definition["phase"]
        # Keep authored stages reproducible when another stage is removed.
        rng = np.random.default_rng(5000 + int(definition["id"].split("_", 1)[0]))
        transforms = {}
        if task == "initial_scene":
            return data, {
                "id": definition["id"],
                "arms": {},
                "qualification": "initial layout only",
            }
        current = self.order.index(task)
        # Support transitions happen AFTER the preceding subtask has been seated.
        for name in self.order[:current]:
            r, p = self.final_pose(name)
            self.pose(data, name, r, p)
            transforms[name] = (r, p)
            if name in self.resting:
                servo = 2 if name == "upper_arm_module" else 3
                data.qpos[m.joint(f"target_servo_{servo}_shaft").qposadr[0]] = (
                    self.resting[name]["seated"]["shaft_angle"]
                )
        if phase == "support_transition":
            # Current task's piece stays loose; both assembly-support grasps coexist.
            pass
        elif phase in ["floor_grasp", "pickup", "transport"]:
            item = self.placed[task]
            r, p = item["rotation"].copy(), item["position"].copy()
            if phase == "pickup":
                p += [0, 0, 0.035]
            if phase == "transport":
                p += [0, 0, 0.090]
                r_final, p_final = self.final_pose(task)
                # Intermediate lift/transport is authored, not reconstructed from a photo.
                center = r @ self.layout.grips[task]["center"] + p
                final_center = r_final @ self.layout.grips[task]["center"] + p_final
                new_center = center.copy()
                new_center[:2] += 0.5 * (final_center[:2] - center[:2])
                side = definition["active_arm"]
                old_radial = center[:2] - self.layout.axes[side][:2]
                new_radial = new_center[:2] - self.layout.axes[side][:2]
                yaw = np.arctan2(new_radial[1], new_radial[0]) - np.arctan2(
                    old_radial[1], old_radial[0]
                )
                r = rz(yaw) @ r
                p = new_center - r @ self.layout.grips[task]["center"]
            self.pose(data, task, r, p)
            transforms[task] = (r, p)
        else:
            r, p = self.final_pose(task)
            if task == "base_side_cover":
                p += [
                    0,
                    0,
                    {
                        "aligned": 0.055,
                        "entry": 0.020,
                        "near_seated": 0.002,
                        "seated": 0,
                    }[phase],
                ]
            elif task == "shoulder_connector":
                p += [
                    {
                        "aligned": 0.065,
                        "entry": 0.025,
                        "near_seated": 0.002,
                        "seated": 0,
                    }[phase],
                    0,
                    0,
                ]
            elif task == "assembly_pin":
                p += [
                    0,
                    0,
                    {
                        "aligned": 0.032,
                        "half_inserted": 0.011,
                        "near_seated": 0.002,
                        "seated": 0,
                    }[phase],
                ]
            else:
                rest = self.resting[task][phase]
                if task == "upper_arm_module":
                    receiver_r, receiver_p = self.rotation, self.origin
                else:
                    upper = self.resting["upper_arm_module"]["seated"]
                    receiver_r = self.rotation @ upper["rotation"]
                    receiver_p = self.origin + self.rotation @ upper["position"]
                r = receiver_r @ rest["rotation"]
                p = receiver_p + receiver_r @ rest["position"]
                servo = 2 if task == "upper_arm_module" else 3
                data.qpos[m.joint(f"target_servo_{servo}_shaft").qposadr[0]] = rest[
                    "shaft_angle"
                ]
            self.pose(data, task, r, p)
            transforms[task] = (r, p)
        mujoco.mj_forward(m, data)
        sag_degrees = self.plan.get("held_body_sag_degrees_by_stage", {}).get(
            definition["id"]
        )
        sag = self.sag_held_link(data, task, sag_degrees) if sag_degrees else None
        arms = {}
        for side in ["left", "right"]:
            role = definition[side + "_role"]
            connector_grip = role == "hold_connector" or (
                role == "grasp_active" and task == "shoulder_connector"
            )
            opening = 0.20
            grasp_spec = None
            approach = np.array([1.0 if side == "left" else -1.0, 0, 0])
            if role == "grasp_active":
                center, span = self.grip(data, task)
                grasp_spec = self.side_grasps.get(task)
                if task == "base_side_cover":
                    approach = np.array([0.0, 0, -1.0])
                elif task in ["upper_arm_module", "forearm_module"]:
                    servo = 2 if task == "upper_arm_module" else 3
                    case_r = data.xmat[
                        m.body(f"target_servo_{servo}_pivot").id
                    ].reshape(3, 3)
                    approach = case_r @ np.array([0.0, 0.0, -1.0])
            elif role == "hold_connector":
                center, span = self.grip(data, "shoulder_connector")
                grasp_spec = self.side_grasps["shoulder_connector"]
            elif role == "hold_base_assembly":
                local_center, grasp_spec = self.installed_base_grip
                center = self.origin + self.rotation @ local_center
                span = self.rotation @ np.array([-1.0, 0, 0.0])
                approach = self.rotation @ np.array([0.0, -1.0, 0.0])
            elif role == "hold_upper_arm":
                body = m.body("upper_arm_module").id
                root_r = data.xmat[body].reshape(3, 3)
                local_center, grasp_spec = self.installed_upper_grip
                center = data.xpos[body] + root_r @ local_center
                span = root_r @ self.upper_support_span
                approach = root_r @ self.upper_support_approach
            else:
                # Open, near the work and clear of its insertion corridor; not parked away.
                center = self.origin + [-0.135 if side == "left" else 0.17, 0.11, 0.105]
                radial = center - self.layout.axes[side]
                radial[2] = 0
                radial /= np.linalg.norm(radial)
                span = np.array([-radial[1], radial[0], 0.0])
                opening = 0.85
            if (
                side == "left"
                and task == "shoulder_connector"
                and role == "nearby_open"
            ):
                # Natural cover release -> open jaws -> retreat, retaining wrist pitch.
                center, span = self.grip(data, "base_side_cover")
                center += [-0.080, 0, 0.035]
                approach = np.array([0.0, 0, -1.0])
                grasp_spec = dict(self.side_grasps["base_side_cover"])
                grasp_spec["opening_radians"] = 0.75
                opening = 0.75
            if connector_grip:
                body = m.body("shoulder_connector").id
                # IMG1617/1616: the finger length follows the horizontal
                # insertion axis, so the arm can push without an angled wrist.
                approach = data.xmat[body].reshape(3, 3) @ np.array([0.0, 1.0, 0.0])
            if approach[2] == 0 and not connector_grip:
                approach = center - self.layout.axes[side]
                approach[2] = 0
                approach -= span * np.dot(approach, span)
                approach /= np.linalg.norm(approach)
            # A jaw span is an unoriented line. For horizontal grasps choose
            # the sign that keeps the actual +siteY camera above the hand,
            # rather than the old arbitrary left/right sign convention.
            if (
                abs(span[2]) < 0.3
                and abs(approach[2]) < 0.3
                and np.cross(span, approach)[2] < 0
                and role != "hold_upper_arm"
            ):
                span = -span
            joint_reference = (
                self.previous[side].copy()
                if role == "grasp_active"
                and task in self.resting
                and phase in ["half_inserted", "near_seated", "seated"]
                else None
            )
            if joint_reference is not None:
                probe = mujoco.MjData(m)
                probe.qpos[:] = data.qpos
                addresses = [
                    m.joint(side + "_" + n).qposadr[0]
                    for n in [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ]
                ]
                probe.qpos[addresses] = joint_reference
                mujoco.mj_forward(m, probe)
                axis = grasp_spec["span_site_axis"]
                prior_span = probe.site_xmat[m.site(side + "_gripperframe").id].reshape(
                    3, 3
                )[:, axis]
                if np.dot(prior_span, span) < 0:
                    span = -span
            arms[side] = self.solve_arm(
                data,
                side,
                center,
                span,
                approach,
                opening,
                rng,
                grasp=grasp_spec,
                joint_reference=joint_reference,
                horizontal_approach=connector_grip,
            )
            if grasp_spec is not None and role != "nearby_open":
                target_names = (
                    [f"servo_{2 if task == 'upper_arm_module' else 3}_case"]
                    if role == "grasp_active"
                    and task in ["upper_arm_module", "forearm_module"]
                    else ["upper_arm_module_printed_0"]
                    if role == "hold_upper_arm"
                    else ["base_servo_printed_0"]
                    if role == "hold_base_assembly"
                    else ["base_side_cover_printed_0"]
                    if task == "base_side_cover"
                    else ["shoulder_connector_printed_1"]
                    if role == "hold_connector" or task == "shoulder_connector"
                    else ["pin"]
                )
                cache_key = (
                    role
                    if role in self.lookahead
                    else (
                        "hold_connector"
                        if task == "shoulder_connector" and phase == "seated"
                        else None
                    )
                )
                ghosts = self.lookahead.get(cache_key, ())
                joints = [
                    m.joint(side + "_" + n).qposadr[0]
                    for n in [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                        "gripper",
                    ]
                ]
                seed = self.plan.get("authoring_support_seeds", {}).get(cache_key)
                if seed and cache_key not in self.support_cache:
                    if seed["layout_profile"] != self.layout.cfg["profile"]:
                        raise ValueError("Support seed layout identity mismatch")
                    data.qpos[joints] = seed["joint_positions"]
                    mujoco.mj_forward(m, data)
                if cache_key in self.support_cache:
                    q, info = self.support_cache[cache_key]
                    data.qpos[joints] = q
                    mujoco.mj_forward(m, data)
                    arms[side]["boundary_refinement"] = dict(info)
                    arms[side]["boundary_refinement"]["reused_support_grasp"] = True
                else:
                    if seed and seed.get("retain_native_reviewed_pose"):
                        receiver = m.body(seed["receiver_body"]).id
                        if not (
                            np.allclose(
                                data.xpos[receiver],
                                seed["receiver_position_world_m"],
                                atol=1e-9,
                                rtol=0,
                            )
                            and np.allclose(
                                data.xmat[receiver],
                                seed["receiver_rotation_world"],
                                atol=1e-9,
                                rtol=0,
                            )
                        ):
                            raise ValueError(
                                "Native support seed receiver pose changed; re-author and audit"
                            )
                        # Native triangle authoring resolves submillimeter visible
                        # gaps that the 1mm voxel field cannot reliably optimize.
                        info = dict(
                            seed["native_review"],
                            target_geom_names=target_names,
                            geometric_refinement_pass=None,
                            qualified=False,
                            method="Authored native triangle-mesh near-contact pose; coarse SDF refinement skipped",
                        )
                    else:
                        info = self.refiner.refine(
                            data,
                            side,
                            center,
                            span,
                            target_names,
                            grasp_spec,
                            ghosts=ghosts,
                            joint_reference=joint_reference,
                            approach_direction=approach if connector_grip else None,
                            horizontal_approach=connector_grip,
                        )
                    arms[side]["boundary_refinement"] = info
                    # Keep each acquired support pose continuous even when its
                    # geometric review rejects it. A rejected pose stays visibly
                    # rejected; repeated optimization must not silently relocate
                    # a supporting hand between snapshots.
                    if cache_key:
                        self.support_cache[cache_key] = (
                            data.qpos[joints].copy(),
                            dict(info),
                        )
                if role == "hold_connector" or task == "shoulder_connector":
                    # Independent reciprocal triangle checks expose a small
                    # moving-finger intrusion missed by the 1mm voxel field.
                    # Keep wrist/object/fixed-finger poses; open the hinge 0.02rad.
                    margin = (
                        self.plan["connector_opening_margins_by_phase_radians"].get(
                            phase, 0.0
                        )
                        if task == "shoulder_connector"
                        else 0.0
                    )
                    data.qpos[m.joint(side + "_gripper").qposadr[0]] += margin
                    mujoco.mj_forward(m, data)
                    arms[side]["boundary_refinement"] = dict(
                        arms[side]["boundary_refinement"],
                        post_refinement_opening_margin_radians=margin,
                        margin_verification="Separate state-matched native boundary receipt required",
                    )
            self.previous[side] = self.joint_values(data, side)
            arms[side]["joint_positions_after_refinement"] = self.previous[
                side
            ].tolist()
            arms[side]["opening_after_refinement_radians"] = float(
                data.qpos[m.joint(side + "_gripper").qposadr[0]]
            )
            arms[side]["role"] = role
        for a in range(m.nu):
            j = m.actuator_trnid[a, 0]
            data.ctrl[a] = data.qpos[m.jnt_qposadr[j]]
        data.qvel[:] = 0
        mujoco.mj_forward(m, data)
        mating = None
        if task in self.resting and phase in self.resting[task]:
            rest = self.resting[task][phase]
            receiver = (
                "base_side_cover" if task == "upper_arm_module" else "upper_arm_module"
            )
            rb = m.body(receiver).id
            tb = m.body(task).id
            far = data.xpos[tb] + data.xmat[tb].reshape(3, 3) @ rest["far_point"]
            receiver_local = (far - data.xpos[rb]) @ data.xmat[rb].reshape(3, 3)
            mating = {
                "receiver_body": receiver,
                "far_point_piece_local_m": rest["far_point"].tolist(),
                "contact_receiver_local_m": receiver_local.tolist(),
                "held_servo_body": f"target_servo_{2 if task == 'upper_arm_module' else 3}_pivot",
                "held_servo_axis_local": [0, 0, -1],
                "far_contact_gap_m": rest["far_contact_gap_m"],
                "receiver_frame_is_world_vertical": task == "upper_arm_module",
            }
        return data, {
            "held_body_sag": sag,
            "mating": mating,
            "id": definition["id"],
            "arms": arms,
            "qualification": "UNQUALIFIED kinematic review",
            "pin_registration": "inferred, not a verified machined hole",
            "seating_reference": "common positioned CAD frame, not contact-validated",
        }

    def export(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        root = self.layout.xml_root
        root.set("model", self.plan["profile"])
        xml = ET.tostring(root, encoding="unicode")
        (output / "scene.xml").write_text(xml)
        signature = hashlib.sha256(xml.encode()).hexdigest()
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        bank = output / "stages"
        bank.mkdir(exist_ok=True)
        receipts = []
        tasks = [s["id"] for s in self.plan["states"]]
        for index, definition in enumerate(self.plan["states"]):
            data, receipt = self.state(definition)
            state = np.empty(mujoco.mj_stateSize(self.model, spec))
            mujoco.mj_getState(self.model, data, state, spec)
            # Each stage is its own task in the REVIEW viewer; these labels are not
            # a change to policy tasks or the success-guided sampler.
            np.savez_compressed(
                bank / (definition["id"] + "_train.npz"),
                states=[state],
                controls=[data.ctrl.copy()],
                seed=[index],
                family=[0],
            )
            receipt.update(
                definition=definition,
                training_eligible=False,
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                control_sha256=hashlib.sha256(data.ctrl.tobytes()).hexdigest(),
            )
            receipts.append(receipt)
            print(
                definition["id"],
                {
                    s: round(a["grasp_position_error_m"] * 1000, 2)
                    for s, a in receipt["arms"].items()
                },
                flush=True,
            )
        (bank / "manifest.json").write_text(
            json.dumps(
                {
                    "model_signature": signature,
                    "state_spec": int(spec),
                    "task_names": tasks,
                    "family_names": ["Unqualified photo reset candidate"],
                    "reset_distribution": "photo_review",
                    "display_name": "WowRobo · photo reset sequence · UNQUALIFIED",
                    "default_viewer_collection": True,
                    "stage_by_task": {s["id"]: s for s in self.plan["states"]},
                    "focus_by_task": {
                        t: {"target": [0, 0, 0.11], "radius": 0.46} for t in tasks
                    },
                },
                indent=2,
            )
            + "\n"
        )
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "profile": self.plan["profile"],
                    "model_signature": signature,
                    "bank": "stages",
                    "count": len(tasks),
                    "qualification": "UNQUALIFIED kinematic review; not a training bank",
                },
                indent=2,
            )
            + "\n"
        )
        (output / "stage-receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        (output / "reset-contract.json").write_text(
            json.dumps(self.plan, indent=2) + "\n"
        )
        return signature
