"""Reciprocal sampled triangle audit for kinematic review snapshots.

Native meshes and their spatial indexes are reused across snapshots. This is a
finite surface-sample check, not continuous collision or force certification.
"""

from itertools import combinations

import numpy as np
import trimesh

from assembly_policy.so101.wowrobo_layout import mesh_vertices

THRESHOLD_M = 0.0005
_TARGET_ROOTS = {
    "base_servo",
    "base_side_cover",
    "shoulder_connector",
    "upper_arm_module",
    "forearm_module",
    "assembly_pin",
}


def world_mesh(model, data, geom):
    """Compatibility helper for callers needing an independent world mesh."""
    mid = model.geom_dataid[geom]
    first, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return trimesh.Trimesh(
        mesh_vertices(model, data, geom),
        model.mesh_face[first : first + count],
        process=False,
    )


def penetration(mesh, points, max_samples=512):
    """Maximum sampled inside-to-surface distance; subsample AFTER local AABB."""
    if max_samples < 512:
        raise ValueError("Boundary audits require at least 512 samples per direction")
    points = np.asarray(points)
    bounds = mesh.bounds
    selected = points[
        np.all((points >= bounds[0] - 1e-8) & (points <= bounds[1] + 1e-8), axis=1)
    ]
    if not len(selected):
        return 0.0
    if len(selected) > max_samples:
        selected = selected[np.linspace(0, len(selected) - 1, max_samples, dtype=int)]
    try:
        import rtree  # noqa: F401
    except ImportError:
        pass
    else:
        inside = mesh.contains(selected)
        if not np.any(inside):
            return 0.0
        _, distance, _ = trimesh.proximity.closest_point(mesh, selected[inside])
        return float(distance.max())

    # Slow dependency-free fallback with the same sample budget, not a convex hull.
    inside = []
    triangles = mesh.triangles
    for start in range(0, len(selected), 8):
        p = selected[start : start + 8]
        a = triangles[None, :, 0] - p[:, None]
        b = triangles[None, :, 1] - p[:, None]
        c = triangles[None, :, 2] - p[:, None]
        na, nb, nc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        numerator = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        denominator = (
            na * nb * nc
            + np.einsum("pfi,pfi->pf", a, b) * nc
            + np.einsum("pfi,pfi->pf", b, c) * na
            + np.einsum("pfi,pfi->pf", c, a) * nb
        )
        winding = np.sum(2 * np.arctan2(numerator, denominator), axis=1)
        inside.extend(p[np.abs(winding) > 2 * np.pi])
    worst = 0.0
    for start in range(0, len(inside), 8):
        _, distance, _ = trimesh.proximity.closest_point_naive(
            mesh, np.asarray(inside[start : start + 8])
        )
        worst = max(worst, float(distance.max()))
    return worst


class NativeBoundaryAudit:
    """Reuse one instance for every snapshot of the same MuJoCo model.

    Includes every helper triangle mesh, including cameras. Target constituents
    in one free-body assembly are excluded from target-target checks. Decorative
    non-mesh geoms, within-helper/non-EEF cross-helper collisions, and continuous
    edges are not tested.
    """

    def __init__(self, model, *, max_samples=512, threshold_m=THRESHOLD_M):
        if max_samples < 512:
            raise ValueError("max_samples must be >= 512")
        self.model = model
        self.max_samples = max_samples
        self.threshold_m = threshold_m
        self.meshes = {}
        self._direction_cache = {}
        self.samples = {}
        self.info = {}
        self.helper = []
        self.targets = []
        self.eef = {"left": [], "right": []}
        self.moving_helper = []
        for gid in range(model.ngeom):
            if model.geom_type[gid] != 7:
                continue
            bid = int(model.geom_bodyid[gid])
            ancestors = []
            current = bid
            while current:
                ancestors.append(model.body(current).name or "")
                current = int(model.body_parentid[current])
            side = next(
                (s for s in ("left", "right") if s + "_base" in ancestors), None
            )
            root = next((n for n in ancestors if n in _TARGET_ROOTS), None)
            if side is None and root is None:
                continue
            mid = int(model.geom_dataid[gid])
            if mid not in self.meshes:
                vstart, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fstart, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                mesh = trimesh.Trimesh(
                    np.array(model.mesh_vert[vstart : vstart + vn]),
                    np.array(model.mesh_face[fstart : fstart + fn]),
                    process=False,
                )
                self.meshes[mid] = mesh
                self.samples[mid] = np.vstack(
                    (mesh.vertices, mesh.triangles_center[::3])
                )
            self.info[gid] = {
                "geom_id": gid,
                "geom_name": model.geom(gid).name or "",
                "body_id": bid,
                "body_name": model.body(bid).name or "",
                "mesh_id": mid,
                "mesh_name": model.mesh(mid).name or "",
                "piece": root,
                "helper_side": side,
            }
            if root:
                self.targets.append(gid)
            if side:
                self.helper.append(gid)
                if model.body(bid).name != side + "_base":
                    self.moving_helper.append(gid)
                if any(
                    n in ancestors for n in (side + "_gripper", side + "_camera_rig")
                ):
                    self.eef[side].append(gid)

    def _direction(self, data, source, target):
        a, b = self.info[source]["mesh_id"], self.info[target]["mesh_id"]
        ra = data.geom_xmat[source].reshape(3, 3)
        rb = data.geom_xmat[target].reshape(3, 3)
        # Row vectors: native source -> world -> native target. Rigid distances
        # are unchanged, and the target's triangle Rtree remains reusable.
        relative_rotation = rb.T @ ra
        relative_position = (data.geom_xpos[source] - data.geom_xpos[target]) @ rb
        key = (
            source,
            target,
            np.round(relative_rotation, 12).tobytes(),
            np.round(relative_position, 12).tobytes(),
        )
        if key not in self._direction_cache:
            points = self.samples[a] @ relative_rotation.T + relative_position
            self._direction_cache[key] = penetration(
                self.meshes[b], points, self.max_samples
            )
        return self._direction_cache[key]

    def _bounds(self, data, gids):
        result = {}
        for gid in gids:
            mesh = self.meshes[self.info[gid]["mesh_id"]]
            vertices = mesh.vertices @ data.geom_xmat[gid].reshape(3, 3).T
            vertices += data.geom_xpos[gid]
            result[gid] = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        return result

    def _pairs(self, data, pairs, bounds):
        rows = []
        for a, b in pairs:
            if np.any(bounds[a][1] < bounds[b][0]) or np.any(
                bounds[b][1] < bounds[a][0]
            ):
                continue
            ab, ba = self._direction(data, a, b), self._direction(data, b, a)
            rows.append(
                {
                    "a": self.info[a],
                    "b": self.info[b],
                    "a_in_b_m": ab,
                    "b_in_a_m": ba,
                    "penetration_m": max(ab, ba),
                    "rejected": max(ab, ba) > self.threshold_m,
                }
            )
        return rows

    def audit(self, data, *, hand_hand=True, target_target=True, table_height_m=0.0):
        bounds = self._bounds(data, self.info)
        helper_target = self._pairs(
            data, ((a, b) for a in self.helper for b in self.targets), bounds
        )
        target_pairs = (
            self._pairs(
                data,
                (
                    (a, b)
                    for a, b in combinations(self.targets, 2)
                    if self.info[a]["piece"] != self.info[b]["piece"]
                ),
                bounds,
            )
            if target_target
            else []
        )
        hand_pairs = (
            self._pairs(
                data,
                ((a, b) for a in self.eef["left"] for b in self.eef["right"]),
                bounds,
            )
            if hand_hand
            else []
        )
        table = [
            dict(
                self.info[g],
                minimum_z_m=float(bounds[g][0, 2]),
                clearance_m=float(bounds[g][0, 2] - table_height_m),
            )
            for g in self.moving_helper
        ]
        table_min = min(table, key=lambda row: row["clearance_m"], default=None)
        all_pairs = helper_target + target_pairs + hand_pairs
        rejected = any(p["rejected"] for p in all_pairs) or any(
            row["clearance_m"] < -self.threshold_m for row in table
        )
        reasons = [
            name
            for name, rows in (
                ("helper_target", helper_target),
                ("target_target", target_pairs),
                ("hand_hand", hand_pairs),
            )
            if any(row["rejected"] for row in rows)
        ]
        if any(row["clearance_m"] < -self.threshold_m for row in table):
            reasons.append("moving_helper_table")
        return {
            "candidate_rejection_reasons": reasons,
            "coverage": {
                "helper_mesh_geoms": len(self.helper),
                "target_mesh_geoms": len(self.targets),
                "native_meshes": len(self.meshes),
                "eef_and_camera_mesh_geoms": sum(map(len, self.eef.values())),
                "non_mesh_geoms_tested": False,
            },
            "method": "reciprocal native-triangle surface sampling",
            "max_samples_per_direction_after_aabb": self.max_samples,
            "threshold_m": self.threshold_m,
            "helper_target": helper_target,
            "target_target": target_pairs,
            "hand_hand": hand_pairs,
            "moving_helper_table": table,
            "table_minimum": table_min,
            "max_sampled_penetration_m": max(
                (p["penetration_m"] for p in all_pairs), default=0.0
            ),
            "candidate_rejected": bool(rejected),
            "training_eligible": False,
            "limitations": "Finite surface samples; mesh geoms only; no within-helper or non-EEF cross-helper collision checks, dynamics, forces, or continuous collision qualification.",
        }


def audit_grippers(model, data):
    """Legacy list API; preserve 0.5 mm callers while including camera meshes."""
    audit = NativeBoundaryAudit(model)
    bounds = audit._bounds(data, audit.info)
    rows = audit._pairs(
        data,
        ((a, b) for side in audit.eef.values() for a in side for b in audit.targets),
        bounds,
    )
    return [
        dict(row, helper=row["a"]["body_name"], target=row["b"]["geom_name"])
        for row in rows
    ]
