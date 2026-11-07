"""Current WowRobo policy point tables, shared with reset inspection.

No older assembly model or assets are loaded. Deterministic sampling and the
ordered tensor encoder reuse the established policy math without changing it.
"""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from assembly_policy.rl.obs import sample_surface
from assembly_policy.so101.obs import PointTables as OrderedPointTables
from assembly_policy.so101.obs import farthest, surface_budget

GRIPS = tuple(s + "_" + b for s in ("left", "right") for b in ("gripper", "moving_jaw_so101_v1"))
WOW_TASKS = ("base_side_cover", "shoulder_connector", "assembly_pin", "upper_arm_module", "forearm_module")
WOW_BASE = "base_servo"


class PointTables:
    """Current WowRobo observation contract: 128 active + 256 receiving + 64 fingers.

    All surfaces come from the supplied current model. Articulated case samples
    retain their own body transforms. The viewer and future policy consumers call
    this same builder/encoder; publishing this contract does not attach a policy.
    """

    __call__ = OrderedPointTables.__call__

    def __init__(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        self.names = tuple(model.body(i).name for i in range(1, model.nbody))
        rng = np.random.default_rng(0)
        pools, areas = {}, {}

        def sample_body(name, descendants=False):
            key = (name, descendants)
            if key in pools:
                return pools[key], areas[key]
            root = model.body(name).id
            bids = {root}
            if descendants is True:
                for bid in range(root + 1, model.nbody):
                    if int(model.body_parentid[bid]) in bids:
                        bids.add(bid)
            chunks, ids, weights = [], [], []
            for gid in range(model.ngeom):
                bid = int(model.geom_bodyid[gid])
                # Use visible source surfaces, excluding markers and collision copies.
                member = (
                    (int(model.body_weldid[bid]) == int(model.body_weldid[root]))
                    if descendants == "welded"
                    else bid in bids
                )
                if not member or model.geom_group[gid] not in (0, 2) or model.geom_rgba[gid, 3] <= 0:
                    continue
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = model.geom_dataid[gid]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                v = model.mesh_vert[va : va + vn]
                f = model.mesh_face[fa : fa + fn]
                tri = v[f]
                area = (
                    np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
                )
                points = sample_surface(v, f, 8192, rng)
                world = points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                chunks.append(world)
                ids.extend([self.names.index(model.body(bid).name)] * len(points))
                weights.extend([area / len(points)] * len(points))
            if not chunks:
                raise ValueError(f"No source visual mesh for {name}")
            points = np.concatenate(chunks)
            picks = rng.choice(len(points), 8192, p=np.array(weights) / sum(weights))
            result = (points[picks], np.asarray(ids)[picks])
            pools[key], areas[key] = result, float(sum(weights))
            return result, areas[key]

        def choose(name, count):
            (world, ids), _ = sample_body(name, True)
            # FPS selects candidate coordinates, then retain their exact body IDs.
            chosen = farthest(np.c_[world], count)
            lookup = {tuple(p): i for i, p in enumerate(world)}
            indices = np.array([lookup[tuple(p)] for p in chosen])
            ids = ids[indices]
            local = np.array(
                [
                    (p - data.body(self.names[i]).xpos) @ data.body(self.names[i]).xmat.reshape(3, 3)
                    for p, i in zip(chosen, ids)
                ]
            )
            return local, ids

        points, which = [], []
        self.mating_counts = {}
        for index, task in enumerate(WOW_TASKS):
            active, active_ids = choose(task, 128)
            receivers = (WOW_BASE, *WOW_TASKS[:index])
            counts = surface_budget([sample_body(n, True)[1] for n in receivers], 256)
            self.mating_counts[task] = dict(zip(receivers, map(int, counts)))
            pairs = [(active, active_ids)] + [choose(n, int(c)) for n, c in zip(receivers, counts)]
            points.append(np.concatenate([p for p, _ in pairs]))
            which.append(np.concatenate([i for _, i in pairs]))
        # Sample the current scene's actual four finger bodies. A welded group
        # includes fixed descendants but excludes the articulated opposing jaw.
        fingers = []
        for name in GRIPS:
            (world, _), _ = sample_body(name, "welded")
            body = data.body(name)
            site = data.site(name.split("_")[0] + "_gripperframe")
            tcp = (world - site.xpos) @ site.xmat.reshape(3, 3)
            local = (world - body.xpos) @ body.xmat.reshape(3, 3)
            fingers.append(farthest(local[tcp[:, 0] > -0.025], 16))
        self.points = torch.tensor(np.asarray(points), dtype=torch.float32)
        self.which = torch.tensor(np.asarray(which), dtype=torch.long)
        self.grip = torch.tensor(np.asarray(fingers), dtype=torch.float32)
        self.n_pts = 448
        self.signature = hashlib.sha256(
            Path(__file__).read_bytes()
            + self.points.numpy().tobytes()
            + self.which.numpy().tobytes()
            + self.grip.numpy().tobytes()
        ).hexdigest()
