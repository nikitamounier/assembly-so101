"""Exterior point pushes with zero applied moment about each moving body's COM.

The convex envelope is used only to choose real source-mesh vertices where a
distributed push can act. It never replaces collision geometry. Cartesian drag
acts at the actual body COM, avoiding origin-offset moments from free-joint drag.
"""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


class BalancedPush:
    def __init__(self, model, data, active_bodies, axis, damping=10.0):
        self.model = model
        self.axis = np.asarray(axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.damping = float(damping)
        self.bodies = []
        total = sum(float(model.body_mass[b]) for b in active_bodies)
        for bid in sorted(active_bodies):
            mass = float(model.body_mass[bid])
            if mass <= 0:
                continue
            points = []
            rb = data.xmat[bid].reshape(3, 3)
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] != bid:
                    continue
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = model.geom_dataid[gid]
                first = model.mesh_vertadr[mid]
                count = model.mesh_vertnum[mid]
                v = model.mesh_vert[first : first + count].astype(float)
                world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                points.append((world - data.xpos[bid]) @ rb)
            if not points:
                raise ValueError(
                    f"No source surface for moving body {model.body(bid).name}"
                )
            vertices = np.unique(np.vstack(points), axis=0)
            hull = ConvexHull(vertices)
            self.bodies.append(
                {
                    "id": bid,
                    "name": model.body(bid).name,
                    "mass_fraction": mass / total,
                    "vertices": vertices,
                    "faces": hull.simplices,
                    "normals": hull.equations[:, :3],
                    "offsets": hull.equations[:, 3],
                    "com": model.body_ipos[bid].copy(),
                }
            )
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.max_drive_moment = 0.0
        self.max_line_error = 0.0

    def patches(self, data):
        """Return upstream surface vertices and weights for the current axis."""
        result = []
        for body in self.bodies:
            bid = body["id"]
            rb = data.xmat[bid].reshape(3, 3)
            local_axis = rb.T @ self.axis
            denom = body["normals"] @ local_axis
            numer = body["normals"] @ body["com"] + body["offsets"]
            upstream = np.flatnonzero(denom < -1e-12)
            distances = numer[upstream] / denom[upstream]
            t = float(np.min(distances))
            hit = body["com"] - t * local_axis
            candidates = upstream[np.abs(distances - t) < 1e-8]
            found = None
            for face in candidates:
                triangle = body["vertices"][body["faces"][face]]
                basis = np.column_stack(
                    (triangle[1] - triangle[0], triangle[2] - triangle[0])
                )
                uv, *_ = np.linalg.lstsq(basis, hit - triangle[0], rcond=None)
                weights = np.array([1 - uv.sum(), uv[0], uv[1]])
                if weights.min() >= -1e-7:
                    weights = np.maximum(weights, 0)
                    weights /= weights.sum()
                    if np.linalg.norm(weights @ triangle - hit) < 1e-8:
                        found = triangle, weights
                        break
            if found is None or t < -1e-8:
                raise ValueError(f"No balanced exterior push patch for {body['name']}")
            triangle, weights = found
            world = triangle @ rb.T + data.xpos[bid]
            center = weights @ world
            com = data.xipos[bid]
            moment = np.cross(center - com, self.axis)
            self.max_line_error = max(
                self.max_line_error, float(np.linalg.norm(moment))
            )
            result.append((body, world, weights))
        return result

    def apply(self, data, magnitude):
        """Apply mass-proportional surface pushes and COM Cartesian drag."""
        model = self.model
        data.qfrc_applied[:] = 0
        records = []
        net_drive = np.zeros(3)
        net_moment = np.zeros(3)
        whole_com = sum(b["mass_fraction"] * data.xipos[b["id"]] for b in self.bodies)
        for body, points, weights in self.patches(data):
            bid = body["id"]
            drive = magnitude * body["mass_fraction"] * self.axis
            for point, weight in zip(points, weights, strict=True):
                if weight <= 1e-10:
                    continue
                force = weight * drive
                mujoco.mj_applyFT(
                    model, data, force, np.zeros(3), point, bid, data.qfrc_applied
                )
                net_drive += force
                net_moment += np.cross(point - whole_com, force)
                records.append(
                    {
                        "body": body["name"],
                        "point_world_m": point.tolist(),
                        "force_N": force.tolist(),
                    }
                )
            mujoco.mj_jacBodyCom(model, data, self.jp, self.jr, bid)
            velocity = self.jp @ data.qvel
            drag = -self.damping * body["mass_fraction"] * velocity
            # A Cartesian viscous field, not a free-joint force at an offset origin.
            mujoco.mj_applyFT(
                model, data, drag, np.zeros(3), data.xipos[bid], bid, data.qfrc_applied
            )
        self.max_drive_moment = max(
            self.max_drive_moment, float(np.linalg.norm(net_moment))
        )
        if np.linalg.norm(net_drive - magnitude * self.axis) > 1e-9:
            raise RuntimeError("Balanced push resultant changed")
        if np.linalg.norm(net_moment) > 1e-8:
            raise RuntimeError("Balanced push has an unintended turning moment")
        return records

    def description(self):
        return {
            "method": "mass-proportional parallel forces on upstream source-mesh vertices; per-body line through COM",
            "drag": "mass-proportional Cartesian velocity drag at each actual body COM",
            "bodies": [
                {k: b[k] for k in ("name", "mass_fraction")} for b in self.bodies
            ],
            "max_drive_moment_about_whole_com_Nm": self.max_drive_moment,
            "max_force_line_distance_to_body_com_m": self.max_line_error,
            "collision_geometry_changed": False,
        }
