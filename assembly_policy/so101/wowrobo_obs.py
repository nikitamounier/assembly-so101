"""Task-local current-assembly point observations, shared by policy and viewer.

Fixed source-surface samples: 128 active, 256 receiver, 64 actual fingers.
Reference seating transforms only select source triangles once; returned samples
always follow their real bodies, never a target pose or current contact subset.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import igl
import mujoco
import numpy as np
import torch
from scipy.spatial import cKDTree

from assembly_policy.rl.obs import sample_surface
from assembly_policy.so101.obs import PointTables as OrderedPointTables
from assembly_policy.so101.obs import farthest, surface_budget

GRIPS = tuple(s + "_" + b for s in ("left", "right") for b in ("gripper", "moving_jaw_so101_v1"))
WOW_TASKS = ("base_side_cover", "shoulder_connector", "assembly_pin", "upper_arm_module", "forearm_module")
WOW_BASE = "base_servo"
CONFIG = Path(__file__).resolve().parents[2] / "configs/wowrobo_point_features.json"


def kinematic_fingerprint(model):
    """Bind body topology, reference frames and articulated axes, not just shape."""
    fields = (
        model.geom_bodyid,
        model.body_parentid,
        model.body_pos,
        model.body_quat,
        model.jnt_bodyid,
        model.jnt_type,
        model.jnt_pos,
        model.jnt_axis,
        model.qpos0,
    )
    return hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in fields)).hexdigest()


def mesh_fingerprint(model, gid):
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return hashlib.sha256(
        model.mesh_vert[va : va + vn].tobytes()
        + model.mesh_face[fa : fa + fn].tobytes()
        + model.geom_pos[gid].tobytes()
        + model.geom_quat[gid].tobytes()
    ).hexdigest()


class PointTables:
    __call__ = OrderedPointTables.__call__

    def __init__(self, model, config=None, feature_budget=None):
        config = json.loads(CONFIG.read_text()) if config is None else config
        if config.get("kinematic_fingerprint", kinematic_fingerprint(model)) != kinematic_fingerprint(model):
            raise ValueError("Point feature reference frames differ; regenerate the observation contract")
        for item in config.get("mesh_fingerprints", []):
            gid = item["id"]
            if gid >= model.ngeom or mesh_fingerprint(model, gid) != item["sha256"]:
                raise ValueError(
                    f"Point feature mesh differs: {item['name'] or gid}; regenerate the observation contract"
                )
        self.names = tuple(model.body(i).name for i in range(1, model.nbody))
        name_index = {n: i for i, n in enumerate(self.names)}
        rng = np.random.default_rng(0)
        data = mujoco.MjData(model)
        self.receiver_budget = {}
        self.mating_counts = {}
        self.coverage = {}
        self.provenance = []
        geometry = {}
        for gid in range(model.ngeom):
            if (
                model.geom_group[gid] not in (0, 2)
                or model.geom_rgba[gid, 3] <= 0
                or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v = np.asarray(model.mesh_vert[va : va + vn], float)
            f = np.asarray(model.mesh_face[fa : fa + fn], np.int64)
            tri = v[f]
            area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
            geometry[gid] = (v, f, area)

        def geoms(name, welded=False):
            root = model.body(name).id
            bids = {root}
            for bid in range(root + 1, model.nbody):
                if int(model.body_parentid[bid]) in bids:
                    bids.add(bid)
            return [
                g
                for g in geometry
                if (
                    int(model.body_weldid[model.geom_bodyid[g]]) == int(model.body_weldid[root])
                    if welded
                    else int(model.geom_bodyid[g]) in bids
                )
            ]

        def world(gid, points):
            return points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]

        def bundle(gid, native):
            bid = int(model.geom_bodyid[gid])
            body = data.body(bid)
            local = (world(gid, native) - body.xpos) @ body.xmat.reshape(3, 3)
            return dict(
                local=local,
                ids=np.full(len(native), name_index[model.body(bid).name], int),
                geom=np.full(len(native), gid, int),
                xyz=world(gid, native),
            )

        def join(rows):
            return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

        def subset(row, ids):
            return {k: v[ids] for k, v in row.items()}

        def fps(row, count):
            if len(row["xyz"]) < count:
                raise ValueError(f"Insufficient source surface candidates: {len(row['xyz'])} < {count}")
            chosen = farthest(row["xyz"], count)
            lookup = {tuple(p): i for i, p in enumerate(row["xyz"])}
            return subset(row, np.array([lookup[tuple(p)] for p in chosen]))

        def samples(gids, n=4096, region=None):
            rows = []
            for gid in gids:
                v, f, _ = geometry[gid]
                selected = f
                if region is not None:
                    # Test vertices as well as centroids to retain small openings
                    # cut into a larger source triangle; reject off-region samples.
                    mask = region(world(gid, v))
                    fm = np.any(mask[f], axis=1) | region(world(gid, v[f].mean(1)))
                    selected = f[fm]
                if not len(selected):
                    continue
                native = sample_surface(v, selected, n, rng)
                row = bundle(gid, native)
                if region is not None:
                    row = subset(row, region(row["xyz"]))
                if len(row["xyz"]):
                    rows.append(row)
            if not rows:
                raise ValueError("No source triangles in configured mating region")
            return join(rows)

        def area_samples(gids, count):
            quotas = surface_budget([geometry[g][2] for g in gids], count, minimum=0)
            return join(
                [subset(samples([g], max(32, int(n))), np.arange(int(n))) for g, n in zip(gids, quotas) if n]
            )

        points = []
        which = []
        for index, task in enumerate(WOW_TASKS):
            spec = config["tasks"][task]
            if len(spec["reference_qpos"]) != model.nq:
                raise ValueError("Point feature reference/model layout mismatch")
            data.qpos[:] = spec["reference_qpos"]
            mujoco.mj_kinematics(model, data)
            receiver = data.body(spec["receiver"])
            rg = receiver.xmat.reshape(3, 3)
            target_r = rg @ np.asarray(spec["rotation_receiver_local"])
            target_p = (
                receiver.xpos
                + rg @ np.asarray(spec["center_receiver_local"])
                - target_r @ np.asarray(spec["mesh_center_local"])
            )
            ag = model.geom(spec["geom"]).id
            av, af, _ = geometry[ag]
            target_v = av @ target_r.T + target_p
            lower, upper = target_v.min(0), target_v.max(0)

            def distance(p):
                result = np.full(len(p), np.inf)
                keep = np.all((p >= lower - 0.018) & (p <= upper + 0.018), axis=1)
                if keep.any():
                    result[keep] = np.sqrt(
                        igl.point_mesh_squared_distance(
                            np.ascontiguousarray(p[keep]),
                            np.ascontiguousarray(target_v),
                            np.ascontiguousarray(af),
                        )[0]
                    )
                return result

            receivers = (WOW_BASE, *WOW_TASKS[:index])
            receiver_gids = [g for n in receivers for g in geoms(n)]
            active_gids = geoms(task)
            nf = int(feature_budget if feature_budget is not None else spec["feature_count"])
            if not 24 <= nf <= 96:
                raise ValueError("Mating-feature budget must be between 24 and 96")
            active_all = area_samples(active_gids, 4096)
            receiver_all = area_samples(receiver_gids, 8192)
            # Evaluate active samples at their seat only to choose mating surfaces;
            # stored coordinates remain attached to each original physical body.
            delta = target_r @ data.geom_xmat[ag].reshape(3, 3).T
            seated_active = (active_all["xyz"] - data.geom_xpos[ag]) @ delta.T + target_p
            if spec.get("pin"):
                bid = model.body(task).id
                root = data.body(bid)
                body_r = delta @ root.xmat.reshape(3, 3)
                body_p = (root.xpos - data.geom_xpos[ag]) @ delta.T + target_p
                body_vertices = (target_v - body_p) @ body_r
                zmax = body_vertices[:, 2].max()
                shoulder = zmax - spec["shaft_length_m"]
                axis = body_r[:, 2]
                mouth = body_p + body_r @ np.array([0, 0, shoulder])

                def cylinder(p):
                    q = p - mouth
                    z = q @ axis
                    r = np.linalg.norm(q - z[:, None] * axis, axis=1)
                    return r, z

                rim = lambda p: (cylinder(p)[0] < 0.0028) & (abs(cylinder(p)[1]) < 0.0008)
                bore = lambda p: (
                    (cylinder(p)[0] < 0.00165)
                    & (cylinder(p)[1] > 0.0008)
                    & (cylinder(p)[1] < spec["bore_depth_m"] - 0.0004)
                )
                cap = lambda p: (
                    (cylinder(p)[0] < 0.0013) & (abs(cylinder(p)[1] - spec["bore_depth_m"]) < 0.0004)
                )
                split = [nf // 3, nf // 2, nf - nf // 3 - nf // 2]
                region_pools = [samples(receiver_gids, 8192, region=region) for region in (rim, bore, cap)]
                features = join([fps(pool, n) for pool, n in zip(region_pools, split)])
                feature_pool = join(region_pools)
                active_z = (seated_active - body_p) @ body_r[:, 2]
                # 8 tip + 24 shaft/shoulder samples; remainder spans the grip.
                tip = fps(subset(active_all, active_z > zmax - 0.0006), 8)
                shaft = fps(
                    subset(active_all, (active_z > shoulder - 0.0005) & (active_z < zmax - 0.0006)), 24
                )
                active_feature = join([tip, shaft])
                near_mask = distance(receiver_all["xyz"]) < 0.012
                pin_r, pin_z = cylinder(features["xyz"])
                self.coverage[task] = dict(
                    feature_regions=dict(zip(("rim", "bore", "blind_stop"), split)),
                    mouth_world_m=mouth.tolist(),
                    max_feature_radius_mm=float(pin_r.max() * 1000),
                    axial_range_mm=[float(pin_z.min() * 1000), float(pin_z.max() * 1000)],
                )
            else:
                feature_gids = [model.geom(n).id for n in spec["feature_geoms"]]
                feature_pool = samples(
                    feature_gids, 4096, region=lambda p: distance(p) < spec.get("feature_distance_m", 0.0015)
                )
                features = fps(feature_pool, nf)
                near_mask = distance(receiver_all["xyz"]) < 0.012
                # Enrich active mating surfaces using their distance to the
                # receiver feature set at the canonical seat, not live contacts.
                active_distance = cKDTree(features["xyz"]).query(seated_active)[0]
                active_feature = fps(subset(active_all, active_distance < 0.006), 32)
                self.coverage[task] = dict(
                    max_feature_surface_distance_mm=float(distance(features["xyz"]).max() * 1000)
                )
            distances = cKDTree(features["xyz"]).query(feature_pool["xyz"])[0] * 1000
            self.coverage[task]["feature_probe_count"] = len(distances)
            self.coverage[task]["feature_probe_distance_mm"] = dict(
                p50=float(np.percentile(distances, 50)),
                p95=float(np.percentile(distances, 95)),
                maximum=float(distances.max()),
            )
            near = fps(subset(receiver_all, near_mask), 256 - nf - 64)
            # Coarse context retains every installed predecessor, with small-part
            # presence floors; it no longer consumes most of the receiving budget.
            coarse_counts = surface_budget([sum(geometry[g][2] for g in geoms(n)) for n in receivers], 64)
            coarse = join(
                [
                    fps(area_samples(geoms(n), max(1024, int(nc) * 16)), int(nc))
                    for n, nc in zip(receivers, coarse_counts)
                ]
            )
            used = {tuple(p) for p in active_feature["xyz"]}
            remaining = np.array([tuple(p) not in used for p in active_all["xyz"]])
            active = join([active_feature, fps(subset(active_all, remaining), 96)])
            receiving = join([features, near, coarse])
            distances = cKDTree(receiving["xyz"]).query(receiver_all["xyz"][near_mask])[0] * 1000
            feature_distances = cKDTree(receiving["xyz"]).query(feature_pool["xyz"])[0] * 1000
            self.coverage[task]["all_receiver_feature_probe_distance_mm"] = dict(
                p95=float(np.percentile(feature_distances, 95)), maximum=float(feature_distances.max())
            )
            self.coverage[task]["nearby_probe_distance_mm"] = dict(
                p95=float(np.percentile(distances, 95)), maximum=float(distances.max())
            )
            all_points = join([active, receiving])
            points.append(all_points["local"])
            which.append(all_points["ids"])
            self.provenance.append(all_points["geom"])
            self.receiver_budget[task] = dict(
                feature=nf, nearby=256 - nf - 64, context=64, active_feature=32, active_context=96
            )
            self.mating_counts[task] = dict(Counter(self.names[i] for i in receiving["ids"]))
        # Actual source finger surfaces; same 16 distal samples per finger.
        mujoco.mj_resetData(model, data)
        mujoco.mj_kinematics(model, data)
        fingers = []
        for name in GRIPS:
            row = area_samples(geoms(name, welded=True), 8192)
            site = data.site(name.split("_")[0] + "_gripperframe")
            tcp = (row["xyz"] - site.xpos) @ site.xmat.reshape(3, 3)
            chosen = fps(subset(row, tcp[:, 0] > -0.025), 16)
            body = data.body(name)
            fingers.append((chosen["xyz"] - body.xpos) @ body.xmat.reshape(3, 3))
        self.points = torch.tensor(np.asarray(points), dtype=torch.float32)
        self.which = torch.tensor(np.asarray(which), dtype=torch.long)
        self.grip = torch.tensor(np.asarray(fingers), dtype=torch.float32)
        self.n_pts = 448
        self.signature = hashlib.sha256(
            Path(__file__).read_bytes()
            + json.dumps(config, sort_keys=True).encode()
            + repr(feature_budget).encode()
            + self.points.numpy().tobytes()
            + self.which.numpy().tobytes()
            + self.grip.numpy().tobytes()
        ).hexdigest()
