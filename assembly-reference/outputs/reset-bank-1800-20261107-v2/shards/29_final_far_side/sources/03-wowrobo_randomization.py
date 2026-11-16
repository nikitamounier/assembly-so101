"""Coherent stage-specific kinematic candidates; physical admission is external.

Never treats IK convergence or a saved CAD pose as a loaded, settled grasp.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from assembly_policy.so101.wowrobo_photo_resets import ROOT, PhotoResetSequence
from assembly_policy.so101.wowrobo_reset_framework import (
    check_library,
    difficulty_for_slot,
    world_grasp,
)


class WowRoboRandomizer:
    def __init__(self, assets=None, nominal_bank=None, config=None):
        if config is None:
            config = ROOT / "configs/wowrobo_randomization.json"
        self.config = (
            json.loads(Path(config).read_text())
            if not isinstance(config, dict)
            else config
        )
        self.assets = Path(
            assets
            or self.config.get("component_assets", "assets_local/wowrobo/components-v2")
        )
        if not self.assets.is_absolute():
            self.assets = ROOT / self.assets
        self.nominal_bank = Path(
            nominal_bank
            or self.config.get("nominal_bank", "assets_local/wowrobo/photo-resets-v3")
        )
        if not self.nominal_bank.is_absolute():
            self.nominal_bank = ROOT / self.nominal_bank
        self.sequence = PhotoResetSequence(self.assets)
        self.model = self.sequence.model
        if self.config.get("use_nominal_model", False):
            self.model = mujoco.MjModel.from_xml_string(
                (self.nominal_bank / "scene.xml").read_text()
            )
            self.sequence.model = self.model
            self.sequence.layout.model = self.model
        self.grasp_library = None
        if self.config.get("grasp_library"):
            path = Path(self.config["grasp_library"])
            path = path if path.is_absolute() else ROOT / path
            self.grasp_library = check_library(json.loads(path.read_text()))
            signature = hashlib.sha256(
                (self.nominal_bank / "scene.xml").read_bytes()
            ).hexdigest()
            if self.grasp_library["model_signature"] != signature:
                raise ValueError("Grasp library and nominal model identities differ")
        self.definitions = {d["id"]: d for d in self.sequence.plan["states"]}
        self.nominal_receipts = {
            d["id"]: d
            for d in json.loads((self.nominal_bank / "stage-receipts.json").read_text())
        }
        manifest_path = self.nominal_bank / "stages/manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected = manifest.get("component_manifest_sha256")
            if (
                expected
                and expected
                != hashlib.sha256(
                    (self.assets / "manifest.json").read_bytes()
                ).hexdigest()
            ):
                raise ValueError(
                    "Reference bank CAD assets do not match selected component manifest"
                )

    def restore(self, stage_id):
        data = mujoco.MjData(self.model)
        with np.load(self.nominal_bank / "stages" / f"{stage_id}_train.npz") as saved:
            mujoco.mj_setState(
                self.model,
                data,
                saved["states"][0],
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            data.ctrl[:] = saved["controls"][0]
        mujoco.mj_forward(self.model, data)
        return data

    def _transform(self, data, piece, rotation, translation):
        b = self.model.body(piece).id
        r = rotation @ data.xmat[b].reshape(3, 3)
        p = rotation @ data.xpos[b] + translation
        self.sequence.pose(data, piece, r, p)
        mujoco.mj_forward(self.model, data)

    @staticmethod
    def _disk(rng, radius):
        a = rng.uniform(-np.pi, np.pi)
        return radius * np.sqrt(rng.random()) * np.array([np.cos(a), np.sin(a)])

    def sample(self, stage_id, seed, split="train", difficulty=None, solve_arms=True):
        if split not in ("train", "eval"):
            raise ValueError("split must be train or eval")
        definition = self.definitions[stage_id]
        cfg = self.config["stages"][stage_id]
        key = f"{split}:{stage_id}:{int(seed)}".encode()
        effective_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        rng = np.random.default_rng(effective_seed)
        if difficulty is None:
            difficulty = difficulty_for_slot(self.config["sampling"], seed, split, rng)
        level = float(difficulty)
        if not 0 <= level <= 1:
            raise ValueError("difficulty must lie in [0,1]")
        data = self.restore(stage_id)
        old = self.restore(stage_id)
        m, seq = self.model, self.sequence
        task = definition["task"]
        number = int(stage_id.split("_", 1)[0])
        receipt = {
            "stage_id": stage_id,
            "seed": int(seed),
            "split": split,
            "effective_seed": effective_seed,
            "difficulty": level,
            "training_eligible": False,
            "qualification": "UNQUALIFIED kinematic candidate; settling/contact/reward admission required",
            "transforms": [],
            "ik": {},
            "cad_revision": self.config.get("cad_revision"),
            "constraints_pending": [
                "loaded_grasp",
                "settling",
                "nonterminal_reward",
                "native_boundaries",
            ],
        }
        low, high = self.config["global"]["friction"]["multiplier_range"]
        receipt["friction_multipliers"] = {
            k: float(rng.uniform(low, high))
            for k in ("finger_part", "part_table", "part_part")
        }
        attached = []
        swivel = 0.0
        if number >= self.config["global"]["connector"]["installed_start_id"]:
            swivel = float(
                rng.uniform(-1, 1)
                * level
                * self.config["global"]["connector"]["yaw_deg"]
            )
            shaft = seq.layout.manifest["servo_axes"]["1"]
            axis = seq.rotation @ np.array(shaft["axis"])
            pivot = seq.origin + seq.rotation @ np.array(shaft["position_m"])
            r = Rotation.from_rotvec(axis * np.deg2rad(swivel)).as_matrix()
            t = pivot - r @ pivot
            attached = ["shoulder_connector"]
            if number >= 17:
                attached.append("assembly_pin")
            if number >= 25:
                attached.append("upper_arm_module")
            following_receiver = (
                (task == "assembly_pin" and number >= 13)
                or (task == "upper_arm_module" and number >= 21)
                or (task == "forearm_module" and number >= 29)
            )
            if following_receiver and task not in attached:
                attached.append(task)
            for piece in attached:
                self._transform(data, piece, r, t)
            if self.config.get("minimum_reset_required", False):
                address = m.joint("target_servo_1_shaft").qposadr[0]
                data.qpos[address] = old.qpos[address] + np.deg2rad(swivel)
            receipt["transforms"].append(
                {
                    "kind": "connector_swivel",
                    "pieces": attached,
                    "rotation": r.tolist(),
                    "translation_m": t.tolist(),
                    "pivot_world_m": pivot.tolist(),
                }
            )
        receipt["connector_swivel_degrees"] = swivel
        receipt["connector_rotated_pieces"] = attached

        # Move loose pieces about their grip centers, recomputing tangent yaw.
        pickup = cfg["category"] in ("loose", "pickup")
        pieces = seq.order if task == "initial_scene" else [task]
        if pickup:
            for piece in pieces:
                side = "left" if seq.order.index(piece) % 2 == 0 else "right"
                center, _ = seq.grip(data, piece)
                delta = np.r_[
                    rng.uniform(-1, 1, 2) * cfg.get("translation_xy_m", 0) * level, 0.0
                ]
                radial = center[:2] - seq.layout.axes[side][:2]
                new_radial = radial + delta[:2]
                yaw = np.arctan2(new_radial[1], new_radial[0]) - np.arctan2(
                    radial[1], radial[0]
                )
                tangent_noise = (
                    float(
                        rng.uniform(-1, 1) * cfg.get("tangent_yaw_noise_deg", 0) * level
                    )
                    if cfg.get("tangent_yaw_noise_deg", 0)
                    else 0.0
                )
                yaw += np.deg2rad(tangent_noise)
                receipt.setdefault("tangent_yaw_noise_degrees", {})[piece] = (
                    tangent_noise
                )
                r = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
                self._transform(data, piece, r, center + delta - r @ center)
        elif cfg["category"] == "transport":
            center, _ = seq.grip(data, task)
            delta = (
                np.r_[
                    rng.uniform(-1, 1, 2) * cfg.get("translation_xy_m", 0),
                    rng.uniform(-1, 1) * cfg.get("z_m", 0),
                ]
                * level
            )
            radial = center[:2] - seq.layout.axes[definition["active_arm"]][:2]
            yaw = np.arctan2(
                (radial + delta[:2])[1], (radial + delta[:2])[0]
            ) - np.arctan2(radial[1], radial[0])
            yaw += np.deg2rad(cfg.get("angle_deg", 0)) * rng.uniform(-1, 1) * level
            r = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
            self._transform(data, task, r, center + delta - r @ center)
            # Vary sag around the approved8deg while preserving its held case.
            bounds = cfg.get("sag_deg")
            if bounds:
                extra = (rng.uniform(*bounds) - 8) * level
                if abs(extra) > 1e-12:
                    seq.sag_held_link(
                        data, task, abs(extra)
                    ) if extra > 0 else self._unsag(data, task, -extra)
        elif cfg["category"] in (
            "aligned",
            "entry",
            "partial",
            "near",
            "far_side_supported",
        ):
            receiver_name = (
                "upper_arm_module"
                if task == "forearm_module"
                else "shoulder_connector"
                if number >= 12
                else "base_servo"
            )
            receiver = m.body(receiver_name).id
            rr = data.xmat[receiver].reshape(3, 3)
            axis = (
                np.array([-1.0, 0, 0])
                if task == "shoulder_connector"
                else -(rr[:, 2])
                if task == "forearm_module"
                else np.array([0.0, 0, -1.0])
            )
            frame = self.nominal_receipts[stage_id].get("insertion_frame")
            if frame:
                frame_receiver = m.body(frame["receiver_body"]).id
                frame_rr = data.xmat[frame_receiver].reshape(3, 3)
                axis = frame_rr @ np.asarray(frame["axis_receiver_local"], dtype=float)
                receipt["insertion_frame"] = frame
            axis /= np.linalg.norm(axis)
            tangent = rr[:, 0] - axis * np.dot(rr[:, 0], axis)
            if np.linalg.norm(tangent) < 0.1:
                tangent = rr[:, 1] - axis * np.dot(rr[:, 1], axis)
            tangent /= np.linalg.norm(tangent)
            other = np.cross(axis, tangent)
            lat = self._disk(rng, cfg.get("lateral_radius_m", 0) * level)
            axial = rng.uniform(-1, 1) * cfg.get("axial_noise_m", 0) * level
            stand = cfg.get("stand_off_m")
            if stand:
                nominal = {
                    "base_side_cover": 0.055,
                    "shoulder_connector": 0.065,
                    "assembly_pin": 0.032,
                }.get(task, 0)
                axial = (rng.uniform(*stand) - nominal) * level
            evidence = self.nominal_receipts[stage_id].get("mating")
            if evidence and self.config.get("cad_revision"):
                evidence = dict(
                    evidence,
                    nominal_support_gap_m=self.nominal_receipts[stage_id]
                    .get("support_contact", {})
                    .get("gap_m", 0.0),
                )
            if evidence:
                # Rotate around the far-end contact, counterrotating the held
                # case so its receiving-axis alignment is preserved.
                receipt["resting_contact"] = self._supported(
                    data, task, evidence, axial, rng, level, cfg, axis=axis
                )
                receipt["projected_noise_constraints"] = [
                    "Engaged lateral/tilt noise projected to contact-preserving hinge/edge motion; physical fit pending"
                ]
            else:
                self._transform(
                    data,
                    task,
                    np.eye(3),
                    lat[0] * tangent + lat[1] * other - axis * axial,
                )
                angle = np.deg2rad(cfg.get("angle_deg", 0)) * level
                center, _ = seq.grip(data, task)
                rv = rng.uniform(-1, 1, 3) * angle
                r = Rotation.from_rotvec(rv).as_matrix()
                self._transform(data, task, r, center - r @ center)
            if evidence and receipt["resting_contact"].get("support_pass") is False:
                receipt["candidate_rejected"] = True
                receipt.setdefault("rejection_reasons", []).append(
                    "Far-end support gap exceeds 0.1 mm"
                )
            if evidence:
                projection_limit = cfg.get(
                    "lateral_radius_m", cfg.get("rest_slide_m", 0.001)
                )
                if (
                    receipt["resting_contact"]["projection_distance_m"]
                    > projection_limit + 1e-6
                ):
                    receipt["candidate_rejected"] = True
                    receipt["rejection_reasons"] = [
                        "Receiver-edge projection exceeds allowed pocket lateral tolerance"
                    ]
            receipt["insertion_noise"] = {
                "axial_m": float(axial),
                "requested_lateral_m": lat.tolist(),
                "applied_lateral_m": [0.0, 0.0] if evidence else lat.tolist(),
                "receiver_body": receiver_name,
            }

        # Preserve each real hand/object relationship rather than adding joint
        # noise independently to a hand which is carrying or supporting a part.
        held_pickup = cfg["category"] != "pickup" or bool(
            rng.random() < self.config["global"]["pickup_held_fraction"]
        )
        gap_bounds = self.config["global"][
            "pin_pickup_approach_gap_m"
            if task == "assembly_pin"
            else "pickup_approach_gap_m"
        ]
        approach_gap = 0.0 if held_pickup else float(rng.uniform(*gap_bounds) * level)
        receipt["pickup_requires_hold"] = held_pickup
        receipt["pickup_approach_gap_m"] = approach_gap
        for side in ("left", "right"):
            role = definition[side + "_role"]
            joints = [
                m.joint(side + "_" + n).qposadr[0]
                for n in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                )
            ]
            if role == "nearby_open":
                for address in joints:
                    j = int(np.flatnonzero(m.jnt_qposadr == address)[0])
                    data.qpos[address] = np.clip(
                        old.qpos[address]
                        + rng.uniform(-1, 1)
                        * np.deg2rad(
                            self.config["global"]["waiting_arm_joint_noise_deg"]
                        )
                        * level,
                        *m.jnt_range[j],
                    )
                continue
            piece = (
                task
                if role == "grasp_active"
                else {
                    "hold_connector": "shoulder_connector",
                    "hold_base_assembly": "base_servo",
                    "hold_upper_arm": "upper_arm_module",
                }[role]
            )
            support_override = (
                self.nominal_receipts[stage_id]
                .get("support_grasp_by_side", {})
                .get(side)
                if role != "grasp_active"
                else None
            )
            if support_override:
                piece = support_override["piece"]
            servo = (
                {"upper_arm_module": 2, "forearm_module": 3}.get(piece)
                if role == "grasp_active"
                else None
            )
            anchor = m.body(f"target_servo_{servo}_pivot" if servo else piece).id
            site = m.site(side + "_gripperframe").id
            r0 = old.xmat[anchor].reshape(3, 3)
            r1 = data.xmat[anchor].reshape(3, 3)
            change = r1 @ r0.T
            target_r = change @ old.site_xmat[site].reshape(3, 3)
            target_p = data.xpos[anchor] + change @ (
                old.site_xpos[site] - old.xpos[anchor]
            )
            spec = (
                dict(support_override["grasp_calibration"])
                if support_override
                else dict(
                    seq.side_grasps[piece]
                    if role == "grasp_active" or role == "hold_connector"
                    else seq.installed_base_grip[1]
                    if role == "hold_base_assembly"
                    else seq.installed_upper_grip[1]
                )
            )
            opening = float(old.qpos[m.joint(side + "_gripper").qposadr[0]])
            if self.grasp_library:
                key = self.grasp_library["bindings"][stage_id][side]
                # Retry other witnessed grips on the SAME rigid motor/body.
                # Supports stay on their named feature; every alternate still
                # goes through IK and native bilateral retention admission.
                retry = int(seed) // self.config["sampling"][f"{split}_per_stage"]
                if role == "grasp_active" and servo and retry > 0:
                    alternatives = [
                        k
                        for k, value in self.grasp_library["grasps"].items()
                        if value["anchor"] == m.body(anchor).name
                        and value["source_witness_pass"]
                    ]
                    if alternatives:
                        key = str(rng.choice(sorted(alternatives)))
                record = self.grasp_library["grasps"][key]
                if record["anchor"] != m.body(anchor).name:
                    raise ValueError(
                        "Grasp must follow its motor case or printed-body anchor"
                    )
                target_p, target_r = world_grasp(
                    data.xpos[anchor], r1, record["transform"]
                )
                spec = dict(record["calibration"])
                opening = spec["opening_radians"]
                receipt.setdefault("grasp_library", {})[side] = {
                    "id": key,
                    "anchor": record["anchor"],
                    "source_witness_pass": record["source_witness_pass"],
                    "candidate_requires_physical_validation": True,
                }
            spec["opening_radians"] = opening
            slide = rng.uniform(-1, 1) * cfg.get("grasp_slide_m", 0) * level
            if support_override and support_override.get("stable_support", True):
                slide = 0.0
            slide_axis = r1[:, 2] if servo or piece == "assembly_pin" else r1[:, 1]
            target_p += slide * slide_axis
            if role == "grasp_active":
                target_p -= approach_gap * target_r[:, 0]
            grasp_angle = (
                np.deg2rad(cfg.get("grasp_angle_deg", 0)) * rng.uniform(-1, 1) * level
            )
            if support_override and support_override.get("stable_support", True):
                grasp_angle = 0.0
            target_r = (
                Rotation.from_rotvec(target_r[:, 0] * grasp_angle).as_matrix()
                @ target_r
            )
            center = target_p + target_r @ np.array(spec["center_offset_site"])
            seq.previous[side] = old.qpos[joints].copy()
            if solve_arms:
                info = seq.solve_arm(
                    data,
                    side,
                    center,
                    target_r[:, 2],
                    target_r[:, 0],
                    opening,
                    rng,
                    grasp=spec,
                    horizontal_approach=piece == "shoulder_connector",
                )
                receipt["ik"][side] = info
            else:
                receipt["ik"][side] = {"pending": True}
        mujoco.mj_forward(m, data)
        data.qvel[:] = 0
        for a in range(m.nu):
            address = m.jnt_qposadr[m.actuator_trnid[a, 0]]
            preload = (
                old.ctrl[a] - old.qpos[address]
                if self.config.get("inherit_equilibrium_control_offsets", False)
                else 0.0
            )
            data.ctrl[a] = np.clip(
                data.qpos[address] + preload, *m.actuator_ctrlrange[a]
            )
        if self.nominal_receipts[stage_id].get("support_grasp_by_side"):
            receipt["support_grasp_by_side"] = self.nominal_receipts[stage_id][
                "support_grasp_by_side"
            ]
        receipt["ik_position_pass"] = solve_arms and all(
            v.get("grasp_position_error_m", 0) <= 0.002 for v in receipt["ik"].values()
        )
        nominal_receipt = self.nominal_receipts[stage_id]
        if nominal_receipt.get("registration_pass") is False:
            receipt["candidate_rejected"] = True
            receipt.setdefault("rejection_reasons", []).extend(
                "reference_registration:" + str(reason)
                for reason in nominal_receipt.get(
                    "registration_rejection_reasons",
                    nominal_receipt.get(
                        "rejection_reasons", ["unresolved reference registration"]
                    ),
                )
            )
        return data, receipt

    def _unsag(self, data, task, degrees):
        m = self.model
        before = data.qpos.copy()
        servo = 2 if task == "upper_arm_module" else 3
        joint = m.joint(f"target_servo_{servo}_shaft")
        self.sequence.sag_held_link(data, task, degrees)
        angle = data.qpos[joint.qposadr[0]] - before[joint.qposadr[0]]
        data.qpos[:] = before
        mujoco.mj_forward(m, data)
        case = m.body(f"target_servo_{servo}_pivot").id
        position = data.xpos[case].copy()
        rotation = Rotation.from_rotvec(data.xaxis[joint.id] * angle).as_matrix()
        self._transform(data, task, rotation, position - rotation @ position)
        data.qpos[joint.qposadr[0]] -= angle
        mujoco.mj_forward(m, data)
        correction = position - data.xpos[case]
        self._transform(data, task, np.eye(3), correction)

    def _far(self, data, task):
        b = self.model.body(task).id
        return (
            data.xpos[b]
            + data.xmat[b].reshape(3, 3)
            @ self.sequence.resting[task]["far_side_supported"]["far_point"]
        )

    def _hinge(self, data, task, angle, pivot):
        servo = 2 if task == "upper_arm_module" else 3
        j = self.model.joint(f"target_servo_{servo}_shaft")
        r = Rotation.from_rotvec(data.xaxis[j.id] * angle).as_matrix()
        self._transform(data, task, r, pivot - r @ pivot)
        data.qpos[j.qposadr[0]] -= angle
        mujoco.mj_forward(self.model, data)

    def _supported_repaired(self, data, task, evidence, axial, rng, level, cfg, axis):
        from assembly_policy.so101.wowrobo_repaired_resets import fit_case_support_hinge

        m = self.model
        servo = 2 if task == "upper_arm_module" else 3
        case = m.body(f"target_servo_{servo}_pivot").id
        joint = m.joint(f"target_servo_{servo}_shaft")
        qadr = int(joint.qposadr[0])
        preferred = float(data.qpos[qadr])
        receiver = m.body(evidence["receiver_body"]).id
        insertion = np.asarray(axis, dtype=float)
        before = data.xpos[case].copy()
        slide = rng.uniform(-1, 1) * cfg.get("rest_slide_m", 0) * level
        fit = None
        if level > 0:
            delta = insertion * axial + data.xmat[receiver].reshape(3, 3)[:, 0] * slide
            self._transform(data, task, np.eye(3), delta)
            fit = fit_case_support_hinge(m, data, task, evidence, preferred)
        root = m.body(task).id
        local = np.asarray(evidence["far_point_piece_local_m"])
        far = data.xpos[root] + data.xmat[root].reshape(3, 3) @ local
        gap = (
            float(fit["witness_distance_m"])
            if fit
            else float(evidence["nominal_support_gap_m"])
        )
        return {
            "method": "registered case path with real internal-hinge support fit",
            "receiver_body": evidence["receiver_body"],
            "far_point_piece_local_m": local.tolist(),
            "contact_world_m": far.tolist(),
            "projection_distance_m": 0.0,
            "support_gap_m": gap,
            "support_tolerance_m": 0.0001,
            "support_pass": gap <= 0.0001,
            "edge_slide_m": float(slide),
            "hinge_delta_radians": float(data.qpos[qadr] - preferred),
            "requested_axial_m": float(axial),
            "actual_axial_m": float(np.dot(data.xpos[case] - before, insertion)),
            "case_support_fit": fit,
        }

    def _supported(self, data, task, evidence, axial, rng, level, cfg, axis=None):
        if self.config.get("cad_revision"):
            return self._supported_repaired(
                data, task, evidence, axial, rng, level, cfg, axis
            )
        import trimesh

        from assembly_policy.so101.wowrobo_boundaries import world_mesh

        m = self.model
        root = m.body(task).id
        local = np.array(evidence["far_point_piece_local_m"])
        far = data.xpos[root] + data.xmat[root].reshape(3, 3) @ local
        receiver = m.body(evidence["receiver_body"]).id
        slide = rng.uniform(-1, 1) * cfg.get("rest_slide_m", 0) * level
        guess = far + slide * data.xmat[receiver].reshape(3, 3)[:, 0]
        meshes = []
        for gid in range(m.ngeom):
            body = int(m.geom_bodyid[gid])
            while body and body != receiver:
                body = int(m.body_parentid[body])
            if body == receiver and m.geom_type[gid] == 7:
                meshes.append(world_mesh(m, data, gid))
        mesh = trimesh.util.concatenate(meshes)
        projected, dist, _ = trimesh.proximity.closest_point(mesh, [guess])
        if level > 0:
            self._transform(data, task, np.eye(3), projected[0] - far)
            far = projected[0]
        servo = 2 if task == "upper_arm_module" else 3
        case = m.body(f"target_servo_{servo}_pivot").id
        insertion = -data.xmat[receiver].reshape(3, 3)[:, 2]
        before = data.qpos.copy()
        position = data.xpos[case].copy()
        self._hinge(data, task, 1e-4, far)
        derivative = np.dot(data.xpos[case] - position, insertion) / 1e-4
        data.qpos[:] = before
        mujoco.mj_forward(m, data)
        theta = np.clip(
            axial / derivative if abs(derivative) > 1e-5 else 0,
            -np.deg2rad(6),
            np.deg2rad(6),
        )
        self._hinge(data, task, float(theta), far)
        actual = float(np.dot(data.xpos[case] - position, insertion))
        return {
            "receiver_body": evidence["receiver_body"],
            "far_point_piece_local_m": local.tolist(),
            "contact_world_m": far.tolist(),
            "projection_distance_m": float(dist[0]),
            "edge_slide_m": float(slide),
            "hinge_delta_radians": float(theta),
            "requested_axial_m": float(axial),
            "actual_axial_m": actual,
        }
