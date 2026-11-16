"""Current-assembly seating reward and its exact positional tolerance boxes.

This geometric reward contract is shared with inspection. It does not certify
settling, retained grasps, material clearance, or hardware calibration.
"""

import json
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[2] / "configs/wowrobo_reward_regions.json"


def seating_error(model, data, goal):
    receiver = data.body(goal["receiver"])
    rr = receiver.xmat.reshape(3, 3)
    rotation = rr @ np.asarray(goal["rotation_receiver_local"])
    center = receiver.xpos + rr @ np.asarray(goal["center_receiver_local"])
    gid = model.geom(goal["geom"]).id
    actual_rotation = data.geom_xmat[gid].reshape(3, 3)
    actual_center = data.geom_xpos[gid] + actual_rotation @ np.asarray(goal["mesh_center_local"])
    position_error = rotation.T @ (actual_center - center)
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation.T @ actual_rotation) - 1) / 2, -1, 1)))
    return center, rotation, position_error, float(angle)


class RewardRegions:
    def __init__(self, model, signature, config=None):
        self.model = model
        self.config = json.loads(CONFIG.read_text()) if config is None else config
        self.compatible = self.config["model_signature"] == signature
        self.position_tolerance = self.config["position_tolerance_m"]
        self.orientation_tolerance = self.config["orientation_tolerance_degrees"]

    def set_tolerances(self, position_mm, orientation_degrees):
        position_mm, orientation_degrees = float(position_mm), float(orientation_degrees)
        if not (0.01 <= position_mm <= 5 and 0.1 <= orientation_degrees <= 10):
            raise ValueError("Position must be 0.01–5 mm and orientation 0.1–10 degrees")
        self.position_tolerance = position_mm / 1000
        self.orientation_tolerance = orientation_degrees

    def snapshot(self, data, entry):
        task = (entry.get("stage") or {}).get("task", entry["task"])
        tasks = list(self.config["tasks"])
        if not self.compatible or task not in tasks:
            return dict(available=False, message="No seating reward for this setup or model.")
        checks = []
        for name in tasks[: tasks.index(task) + 1]:
            goal = self.config["tasks"][name]
            center, rotation, error, angle = seating_error(self.model, data, goal)
            passed = bool(
                np.max(np.abs(error)) <= self.position_tolerance and angle <= self.orientation_tolerance
            )
            checks.append(
                dict(
                    task=name, position_error_m=error.tolist(), orientation_error_degrees=angle, passed=passed
                )
            )
            if name == task:
                box = dict(
                    center=center.tolist(),
                    rotation_row_major=rotation.reshape(-1).tolist(),
                    size=goal["size_m"],
                )
        return dict(
            available=True,
            task=task,
            receiver=self.config["tasks"][task]["receiver"],
            reward=int(all(c["passed"] for c in checks)),
            checks=checks,
            target_box=box,
            noise_box=dict(
                center=box["center"],
                rotation_row_major=box["rotation_row_major"],
                size=[2 * self.position_tolerance] * 3,
            ),
            position_tolerance_mm=self.position_tolerance * 1000,
            orientation_tolerance_degrees=self.orientation_tolerance,
            definition=self.config["definition"],
            source="Verified full-insertion endpoint; receiver-relative target",
            limitations="Geometric seating reward; contact safety and settling are not evaluated by this overlay.",
        )

    def reward(self, data, task):
        """Binary reward consumed by clients of this current-model contract."""
        return self.snapshot(data, dict(task=task)).get("reward", 0)
