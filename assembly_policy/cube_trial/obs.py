"""Deterministic ordered cube, target-circle and real attached pad point tables."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from assembly_policy.rl.obs import _box_surface, quat_to_mat_torch

from .core import CUBE_SIZE, TARGET_RADIUS, model


def require_observation_signature(config, tables):
    if config.get("observation_signature") != tables.signature:
        raise ValueError("Cube point observation contract differs")


class PointTables:
    def __init__(self, task_names, n_part=256, n_fix=128, n_grip=64, seed=0, device="cpu"):
        if list(task_names) != ["cube_handoff"] or min(n_part, n_fix, n_grip) < 1 or n_grip % 4:
            raise ValueError("Cube task with positive point budgets and four equal finger budgets required")
        rng = np.random.default_rng(seed)
        cube = _box_surface([CUBE_SIZE / 2] * 3, n_part, rng)
        angle = np.arange(n_fix) * 2 * np.pi / n_fix
        target = np.c_[TARGET_RADIUS * np.cos(angle), TARGET_RADIUS * np.sin(angle), np.full(n_fix, 0.0007)]
        self.points = torch.tensor(np.r_[cube, target], dtype=torch.float32, device=device)
        self.which = torch.tensor([0] * n_part + [1] * n_fix, dtype=torch.long, device=device)
        m = model()
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        grip = []
        for s in ["left", "right"]:
            for p, b in [("fixed", "gripper"), ("moving", "moving_jaw_so101_v1")]:
                gid = m.geom(s + "_" + p + "_pad").id
                body = d.body(s + "_" + b)
                local = _box_surface(m.geom_size[gid], n_grip // 4, rng)
                world = local @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                grip.append((world - body.xpos) @ body.xmat.reshape(3, 3))
        self.grip = torch.tensor(np.array(grip), dtype=torch.float32, device=device)
        self.n_pts = n_part + n_fix + n_grip
        self.signature = hashlib.sha256(
            Path(__file__).read_bytes() + repr((list(task_names), n_part, n_fix, n_grip, seed)).encode()
        ).hexdigest()

    def __call__(self, task, tcp_p, tcp_R, part_p, part_q, finger):
        B = len(task)
        r = quat_to_mat_torch(part_q)
        world = torch.einsum("bnij,nj->bni", r[:, self.which], self.points) + part_p[:, self.which]
        grip = torch.einsum("bgij,gpj->bgpi", tcp_R, self.grip) + tcp_p[:, :, None]
        return (10 * torch.cat((world, grip.reshape(B, -1, 3)), 1)).reshape(B, -1)
