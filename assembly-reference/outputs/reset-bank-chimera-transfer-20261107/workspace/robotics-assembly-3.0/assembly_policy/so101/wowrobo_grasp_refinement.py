"""Collision-aware refinement of review grasps against voxelized native meshes.

This is geometric authoring, not contact dynamics or grasp-retention validation.
"""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares


class MeshField:
    def __init__(self, mesh, cache):
        pitch = 0.001
        key = hashlib.sha256(
            mesh.vertices.tobytes() + mesh.faces.tobytes()
        ).hexdigest()[:20]
        path = cache / (key + ".npz")
        if path.exists():
            with np.load(path) as f:
                self.sdf = f["sdf"]
                self.origin = f["origin"]
                self.pitch = float(f["pitch"])
        else:
            vox = mesh.voxelized(pitch, method="ray").fill()
            occupancy = np.pad(vox.matrix, 2)
            self.origin = vox.transform[:3, 3] - 2 * pitch
            self.pitch = pitch
            self.sdf = (
                (distance_transform_edt(~occupancy) - distance_transform_edt(occupancy))
                * 0.001
                + 0.0005
            ).astype(np.float32)
            np.savez_compressed(
                path, sdf=self.sdf, origin=self.origin, pitch=self.pitch
            )
        self.last = np.array(self.sdf.shape) - 1

    def query(self, points):
        coord = (points - self.origin) / self.pitch
        clipped = np.clip(coord, 0, self.last)
        distance = map_coordinates(self.sdf, clipped.T, order=1, mode="nearest")
        return distance + np.linalg.norm(coord - clipped, axis=1) * self.pitch


class GraspRefiner:
    def __init__(self, model, cache):
        self.model = model
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.targets = []
        self.fields = {}
        self.meshes = {}
        for gid in range(model.ngeom):
            if model.geom_type[gid] != 7:
                continue
            mid = model.geom_dataid[gid]
            if mid not in self.meshes:
                first, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                face, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                self.meshes[mid] = trimesh.Trimesh(
                    model.mesh_vert[first : first + n],
                    model.mesh_face[face : face + nf],
                    process=False,
                )
            name = model.geom(gid).name or ""
            if (
                name.startswith(
                    (
                        "servo_",
                        "base_servo_printed",
                        "base_side_cover_printed",
                        "shoulder_connector_printed",
                        "upper_arm_module_printed",
                        "forearm_module_printed",
                    )
                )
                or name == "pin"
            ):
                self.targets.append(gid)
                if mid not in self.fields:
                    self.fields[mid] = MeshField(self.meshes[mid], cache)
        self.hand = {}
        for side in ["left", "right"]:
            root = model.body(side + "_gripper").id
            chosen = []
            for gid in range(model.ngeom):
                if model.geom_type[gid] != 7:
                    continue
                b = int(model.geom_bodyid[gid])
                while b and b != root:
                    b = int(model.body_parentid[b])
                if b != root:
                    continue
                mesh = self.meshes[model.geom_dataid[gid]]
                count = min(3000, max(600, int(mesh.area / 0.0008**2)))
                surface, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
                feature = mesh.vertices
                if len(feature) > 600:
                    feature = feature[np.linspace(0, len(feature) - 1, 600, dtype=int)]
                pts = np.vstack([feature, surface])
                if model.geom_dataid[gid] not in self.fields:
                    self.fields[model.geom_dataid[gid]] = MeshField(mesh, cache)
                chosen.append((gid, pts))
            self.hand[side] = chosen

    def hand_clearance(self, data, side, ghosts=()):
        m = self.model
        points = np.vstack(
            [
                p @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                for g, p in self.hand[side]
            ]
        )
        clearance = points[:, 2].copy()
        for g in self.targets:
            local = (points - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
            clearance = np.minimum(
                clearance, self.fields[m.geom_dataid[g]].query(local)
            )
        for g, pos, r in ghosts:
            clearance = np.minimum(
                clearance, self.fields[m.geom_dataid[g]].query((points - pos) @ r)
            )
        return float(clearance.min())

    def refine(
        self,
        data,
        side,
        center,
        span,
        target_names,
        spec,
        ghosts=(),
        joint_reference=None,
        approach_direction=None,
        horizontal_approach=False,
    ):
        m = self.model
        joints = [
            m.joint(side + "_" + n).id
            for n in [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]
        ]
        addresses = m.jnt_qposadr[joints]
        limits = m.jnt_range[joints].copy()
        if joint_reference is not None:
            # A held insertion stage cannot regrasp via a different IK branch.
            limits[:5, 0] = np.maximum(limits[:5, 0], joint_reference - 0.6)
            limits[:5, 1] = np.minimum(limits[:5, 1], joint_reference + 0.6)
        start = data.qpos[addresses].copy()
        if approach_direction is not None:
            approach_direction = np.asarray(approach_direction, dtype=float)
            approach_direction /= np.linalg.norm(approach_direction)
        site = m.site(side + "_gripperframe").id
        fixed = m.body(side + "_gripper").id
        moving = m.body(side + "_moving_jaw_so101_v1").id
        target = [g for g in self.targets if m.geom(g).name in target_names]
        if not target:
            raise ValueError("A grasp must name its actual object geometry")
        pads = [[], []]
        initial_frame = data.site_xmat[site].reshape(3, 3)
        for gid, _ in self.hand[side]:
            b = int(m.geom_bodyid[gid])
            which = 0 if b == fixed else 1 if b == moving else None
            if which is None:
                continue
            mesh = self.meshes[m.geom_dataid[gid]]
            world = (
                mesh.vertices @ data.geom_xmat[gid].reshape(3, 3).T
                + data.geom_xpos[gid]
            )
            local = (world - data.site_xpos[site]) @ initial_frame
            normals = (
                mesh.vertex_normals
                @ data.geom_xmat[gid].reshape(3, 3).T
                @ initial_frame
            )
            mask = (
                (local[:, 0] > -0.035)
                & (local[:, 0] < 0.004)
                & (np.abs(local[:, 1]) < 0.010)
            )
            mask &= normals[:, 2] > 0.25 if which == 0 else normals[:, 2] < -0.25
            pts = mesh.vertices[mask]
            if len(pts) > 80:
                pts = pts[np.linspace(0, len(pts) - 1, 80, dtype=int)]
            if len(pts):
                pads[which].append((gid, pts))
        if not all(pads):
            return {"qualified": False, "reason": "no interior finger pad samples"}

        def world_points(gid, points):
            return points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]

        def distances(points, geoms):
            best = np.full(len(points), 1e3)
            for g in geoms:
                local = (points - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
                best = np.minimum(best, self.fields[m.geom_dataid[g]].query(local))
            return best

        initial_hand = np.vstack([world_points(g, p) for g, p in self.hand[side]])
        low, high = initial_hand.min(0) - 0.03, initial_hand.max(0) + 0.03
        reciprocal_points = []
        transforms = [
            (g, data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy())
            for g in self.targets
        ] + list(ghosts)
        for g, pos, rotation in transforms:
            mesh = self.meshes[m.geom_dataid[g]]
            source = np.vstack([mesh.vertices, mesh.triangles_center])
            world = source @ rotation.T + pos
            world = world[np.all((world >= low) & (world <= high), axis=1)]
            if len(world) > 1200:
                world = world[np.linspace(0, len(world) - 1, 1200, dtype=int)]
            if len(world):
                reciprocal_points.append(world)
        reciprocal_points = (
            np.vstack(reciprocal_points) if reciprocal_points else np.zeros((0, 3))
        )
        offset = np.array(spec["center_offset_site"])

        def evaluate(q, receipt=False):
            data.qpos[addresses] = q
            mujoco.mj_forward(m, data)
            points = np.vstack([world_points(g, p) for g, p in self.hand[side]])
            clearance = np.minimum(distances(points, self.targets), points[:, 2])
            for g, pos, rotation in ghosts:
                local = (points - pos) @ rotation
                clearance = np.minimum(
                    clearance, self.fields[m.geom_dataid[g]].query(local)
                )
            reciprocal = np.full(len(reciprocal_points), 1e3)
            for g, _ in self.hand[side]:
                local = (reciprocal_points - data.geom_xpos[g]) @ data.geom_xmat[
                    g
                ].reshape(3, 3)
                reciprocal = np.minimum(
                    reciprocal, self.fields[m.geom_dataid[g]].query(local)
                )
            pad_points = [
                np.vstack([world_points(g, p) for g, p in pad]) for pad in pads
            ]
            pad_distances = [distances(points, target) for points in pad_points]
            contacts = np.array([values.min() for values in pad_distances])
            touched = np.array(
                [
                    points[np.argmin(values)]
                    for points, values in zip(pad_points, pad_distances)
                ]
            )
            frame = data.site_xmat[site].reshape(3, 3)
            signed_positions = (touched - center) @ frame[:, 2]
            width = spec["object_span_m"]
            opposition = np.array(
                [
                    max(signed_positions[0] + width * 0.25, 0),
                    max(width * 0.25 - signed_positions[1], 0),
                ]
            )
            region = np.zeros(2)
            if target_names == ["pin"]:
                body = m.body("assembly_pin").id
                pin_local = (touched - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                region = np.maximum(0.004 - pin_local[:, 2], 0) + np.maximum(
                    pin_local[:, 2] - 0.024, 0
                )
            frame = data.site_xmat[site].reshape(3, 3)
            grip = data.site_xpos[site] + frame @ offset
            pitch_error = float(np.degrees(np.arcsin(np.clip(abs(frame[2, 0]), 0, 1))))
            axis_error = (
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(frame[:, 0], approach_direction), -1, 1)
                        )
                    )
                )
                if approach_direction is not None
                else None
            )
            if receipt:
                return {
                    "approach_alignment_error_degrees": axis_error,
                    "approach_pitch_error_degrees": pitch_error,
                    "approach_constraint_error_degrees": pitch_error
                    if horizontal_approach
                    else axis_error,
                    "minimum_sdf_clearance_m": float(clearance.min()),
                    "reciprocal_sdf_clearance_m": float(reciprocal.min())
                    if len(reciprocal)
                    else 1.0,
                    "lookahead_obstacle_count": len(ghosts),
                    "pad_surface_distances_m": contacts.tolist(),
                    "opposed_pad_positions_m": signed_positions.tolist(),
                    "contact_points_world_m": touched.tolist(),
                    "pin_region_violation_m": region.tolist(),
                    "grip_center_shift_m": float(np.linalg.norm(grip - center)),
                    "method": "1mm native-mesh SDF, sampled hand/camera surfaces; requires exact-mesh audit",
                }
            return np.r_[
                3000 * np.minimum(clearance - 0.0008, 0),
                2000 * np.minimum(reciprocal - 0.0002, 0),
                400 * (contacts - 0.0006),
                400 * opposition,
                800 * region,
                20 * (grip - center),
                0.3 * (frame[:, 2] - span),
                (0.3 if horizontal_approach else 20)
                * (frame[:, 0] - approach_direction)
                if approach_direction is not None
                else np.empty(0),
                [20 * frame[2, 0]] if horizontal_approach else np.empty(0),
                0.005 * (q - start),
            ]

        initial = evaluate(start, True)
        candidates = [start]
        for delta in [-np.pi, np.pi]:
            seed = start.copy()
            seed[4] += delta
            if limits[4, 0] + 0.001 < seed[4] < limits[4, 1] - 0.001:
                candidates.append(seed)
        best = None
        for seed in candidates:
            fit = least_squares(
                evaluate,
                np.clip(seed, limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
                bounds=(limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
                max_nfev=150,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
                diff_step=1e-4,
            )
            score = np.linalg.norm(evaluate(fit.x))
            if best is None or score < best[0]:
                best = (score, fit.x.copy())
            info = evaluate(fit.x, True)
            if (
                info["minimum_sdf_clearance_m"] >= 0.0004
                and info["reciprocal_sdf_clearance_m"] >= -0.0002
                and max(info["pad_surface_distances_m"]) < 0.0018
                and (
                    info["approach_constraint_error_degrees"] is None
                    or info["approach_constraint_error_degrees"] < 3.0
                )
            ):
                break
        final = evaluate(best[1], True)
        final["geometric_refinement_pass"] = bool(
            final["minimum_sdf_clearance_m"] >= 0.0004
            and final["reciprocal_sdf_clearance_m"] >= -0.0002
            and max(final["pad_surface_distances_m"]) < 0.0018
            and max(final["pin_region_violation_m"]) < 0.0005
            and (
                final["approach_constraint_error_degrees"] is None
                or final["approach_constraint_error_degrees"] < 3.0
            )
        )
        final["target_geom_names"] = list(target_names)
        final["initial"] = initial
        final["qualified"] = False
        return final
