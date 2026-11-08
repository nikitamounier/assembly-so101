"""Current-assembly policy tensor export, bound atomically to the selected state."""

import hashlib

import numpy as np
import torch

from assembly_policy.so101.wowrobo_obs import GRIPS, WOW_TASKS, PointTables


class PolicyPointCloud:
    def __init__(self, model, scene, point_config=None):
        self.model, self.scene = model, scene
        self.tables = None
        self.point_config = point_config

    def snapshot(self, data, entry):
        task = (entry.get("stage") or {}).get("task", entry["task"])
        preview = self.scene == "wowrobo-layout"
        if preview:
            if task not in WOW_TASKS:
                return dict(
                    available=False,
                    message="Setup state: no active assembly task or policy input.",
                    preview=True,
                )
            if self.tables is None:
                self.tables = PointTables(self.model, config=self.point_config)
            tasks, names = WOW_TASKS, self.tables.names
        else:
            return dict(
                available=False, message="No policy point-cloud adapter is configured for this scene."
            )
        grip_p = np.asarray([data.body(n).xpos for n in GRIPS], np.float32)
        grip_r = np.asarray([data.body(n).xmat.reshape(3, 3) for n in GRIPS], np.float32)
        kwargs = dict(
            task=torch.tensor([tasks.index(task)]),
            tcp_p=torch.tensor(grip_p[None]),
            tcp_R=torch.tensor(grip_r[None]),
            part_p=torch.tensor(np.asarray([[data.body(n).xpos for n in names]], np.float32)),
            part_q=torch.tensor(np.asarray([[data.body(n).xquat for n in names]], np.float32)),
            finger=torch.zeros((1, 2)),
        )
        with torch.no_grad():
            tensor = self.tables(**kwargs)[0].numpy()
        ids = self.tables.which[tasks.index(task)].tolist()
        labels = [names[i] for i in ids] + [name for name in GRIPS for _ in range(16)]
        return dict(
            available=True,
            preview=preview,
            status="Current assembly observation · no policy connected"
            if preview
            else "Exact configured policy observation",
            task=task,
            count=self.tables.n_pts,
            shape=[self.tables.n_pts, 3],
            dtype="float32",
            frame="left_gripper",
            scale=10,
            frame_pose_world=dict(
                position=grip_p[0].tolist(),
                rotation_row_major=grip_r[0].reshape(-1).tolist(),
                rotation=grip_r[0].tolist(),
            ),
            units="metres × 10",
            points=tensor.reshape(-1, 3).tolist(),
            flat=tensor.tolist(),
            bodies=labels,
            groups=[
                dict(name="Active part", start=0, count=128, color="#dc7928"),
                dict(name="Receiving assembly", start=128, count=256, color="#178d84"),
                dict(name="Fingers", start=384, count=64, color="#567ddb"),
            ],
            sampling_budget=self.tables.receiver_budget[task],
            receiver_body_counts=self.tables.mating_counts[task],
            observation_signature=self.tables.signature,
            tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
            state_id=entry["id"],
        )
