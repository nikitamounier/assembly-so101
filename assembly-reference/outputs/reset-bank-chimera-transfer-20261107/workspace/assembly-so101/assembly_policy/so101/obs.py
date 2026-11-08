"""Fixed ordered part/mating surface samples and distal finger geometry."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..rl.obs import quat_to_mat_torch, sample_surface
from .scene import ASSEMBLY_TASKS, OBS_BODIES, model

MATING_BODIES = {task: ("base", *ASSEMBLY_TASKS[:i]) for i, task in enumerate(ASSEMBLY_TASKS)}


def surface_budget(areas, total, minimum=4):
    """Area-proportional quotas, a small presence floor, and exact total."""
    areas = np.asarray(areas, dtype=float)
    floor = min(minimum, total // len(areas))
    remaining = total - floor * len(areas)
    raw = remaining * areas / areas.sum()
    counts = np.floor(raw).astype(int)
    extra = remaining - int(counts.sum())
    counts[np.argsort(-(raw - counts), kind="stable")[:extra]] += 1
    return counts + floor


def farthest(points, n):
    """Deterministic coverage, with indices reused for every control step."""
    if len(points) < n:
        raise ValueError("Insufficient surface candidates")
    chosen = []
    distance = np.full(len(points), np.inf)
    index = int(np.argmax(np.sum((points - points.mean(0)) ** 2, axis=1)))
    for _ in range(n):
        chosen.append(index)
        distance = np.minimum(distance, np.sum((points - points[index]) ** 2, axis=1))
        distance[chosen] = -1
        index = int(np.argmax(distance))
    return points[chosen]


def require_observation_signature(config, tables):
    if config.get("observation_signature") != tables.signature:
        raise ValueError("SO101 observation schema differs; train a new policy for part/mating/finger inputs")


class PointTables:
    """n_fix samples the receiving assembly for every task."""

    def __init__(self, task_names, n_part=128, n_fix=256, n_grip=64, seed=0, device="cpu"):
        if n_grip % 4:
            raise ValueError("n_grip must be divisible by four")
        m = model()
        rng = np.random.default_rng(seed)
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        surface_areas = {}

        def body_samples(name, n):
            bid = m.body(name).id
            chunks = []
            areas = []
            for gid in range(m.ngeom):
                if m.body_weldid[m.geom_bodyid[gid]] != m.body_weldid[bid]:
                    continue
                if m.geom_group[gid] != 2:
                    continue
                mid = m.geom_dataid[gid]
                if mid >= 0 and int(m.geom_type[gid]) in (
                    int(mujoco.mjtGeom.mjGEOM_MESH),
                    int(mujoco.mjtGeom.mjGEOM_SDF),
                ):
                    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                    p = sample_surface(
                        m.mesh_vert[va : va + vn], m.mesh_face[fa : fa + fn], max(4 * n, 32), rng
                    )
                    tri = m.mesh_vert[va : va + vn][m.mesh_face[fa : fa + fn]]
                    area = (
                        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
                        / 2
                    )
                else:
                    import trimesh

                    kind = int(m.geom_type[gid])
                    size = m.geom_size[gid]
                    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                        mesh = trimesh.creation.box(extents=2 * size)
                    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
                    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[16, 16])
                    else:
                        raise ValueError(f"Unhandled point geometry {kind} on {name}")
                    p = sample_surface(mesh.vertices, mesh.faces, max(4 * n, 32), rng)
                    area = mesh.area
                world = p @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                chunks.append((world - d.body(name).xpos) @ d.body(name).xmat.reshape(3, 3))
                areas.append(area)
            if not chunks:
                raise ValueError(f"No surface for {name}")
            p = np.concatenate(chunks)
            surface_areas[name] = float(sum(areas))
            weights = np.concatenate([np.full(len(c), area / len(c)) for c, area in zip(chunks, areas)])
            return p[rng.choice(len(p), n, replace=len(p) < n, p=weights / weights.sum())]

        names = OBS_BODIES
        points = []
        which = []
        self.mating_counts = {}
        for task in task_names:
            active = names.index(task)
            moving = body_samples(task, max(4096, n_part * 16))
            p = [*farthest(moving, n_part)]
            ids = [active] * n_part
            receivers = MATING_BODIES[task]
            pools = {name: body_samples(name, max(8192, n_fix * 32)) for name in receivers}
            counts = surface_budget([surface_areas[name] for name in receivers], n_fix)
            self.mating_counts[task] = {name: int(count) for name, count in zip(receivers, counts)}
            for name, count in self.mating_counts[task].items():
                p.extend(farthest(pools[name], count))
                ids.extend([names.index(name)] * count)
            points.append(p)
            which.append(ids)
        self.points = torch.tensor(np.array(points), dtype=torch.float32, device=device)
        self.which = torch.tensor(np.array(which), dtype=torch.long, device=device)
        grips = [
            side + "_" + part for side in ("left", "right") for part in ("gripper", "moving_jaw_so101_v1")
        ]
        grip = []
        for name in grips:
            candidates = body_samples(name, max(8192, n_grip * 64))
            body = d.body(name)
            site = d.site(name.split("_")[0] + "_gripperframe")
            world = candidates @ body.xmat.reshape(3, 3).T + body.xpos
            tcp = (world - site.xpos) @ site.xmat.reshape(3, 3)
            # Source TCP x runs along the fingers. Keep their distal 25 mm,
            # excluding the servo/palm shell without inventing pad geometry.
            candidates = candidates[tcp[:, 0] > -0.025]
            grip.append(farthest(candidates, n_grip // 4))
        self.grip = torch.tensor(np.array(grip), dtype=torch.float32, device=device)
        self.signature = hashlib.sha256(
            Path(__file__).read_bytes() + repr((list(task_names), n_part, n_fix, n_grip, seed)).encode()
        ).hexdigest()
        self.n_pts = n_part + n_fix + n_grip

    def __call__(self, task, tcp_p, tcp_R, part_p, part_q, finger):
        B = len(task)
        r = quat_to_mat_torch(part_q)
        ids = self.which[task]
        batch = torch.arange(B, device=task.device)[:, None]
        p = torch.einsum("bnij,bnj->bni", r[batch, ids], self.points[task]) + part_p[batch, ids]
        grip = torch.einsum("bgij,gpj->bgpi", tcp_R, self.grip) + tcp_p[:, :, None]
        world = torch.cat((p, grip.reshape(B, -1, 3)), 1)
        local = torch.einsum("bij,bnj->bni", tcp_R[:, 0].transpose(1, 2), world - tcp_p[:, 0, None])
        return (10 * local).reshape(B, -1)
