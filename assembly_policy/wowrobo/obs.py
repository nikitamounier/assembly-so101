"""Trainer interface to the identical current-model point encoder used by viewer."""

import hashlib

import mujoco

from ..so101.wowrobo_obs import WOW_TASKS
from ..so101.wowrobo_obs import PointTables as CurrentPointTables
from .runtime import DEFAULT_REVIEW


class PointTables(CurrentPointTables):
    def __init__(
        self,
        task_names,
        n_part=128,
        n_fix=256,
        n_grip=64,
        seed=0,
        device="cpu",
        *,
        review_model=DEFAULT_REVIEW,
    ):
        if (n_part, n_fix, n_grip, seed) != (128, 256, 64, 0):
            raise ValueError("Current observation contract is fixed at 128+256+64 points, seed0")
        super().__init__(mujoco.MjModel.from_xml_path(str(review_model)))
        indices = [WOW_TASKS.index(t) for t in task_names]
        self.points = self.points[indices].to(device)
        self.which = self.which[indices].to(device)
        self.grip = self.grip.to(device)
        self.signature = hashlib.sha256((self.signature + repr(list(task_names))).encode()).hexdigest()


def require_observation_signature(config, tables):
    if config.get("observation_signature") != tables.signature:
        raise ValueError("Current assembly observation signature differs")
