"""Reproducible source-preserving cover/receiver fit registration.

The small pose correction removes the artificial common-body-frame alignment
assumption. Geometry and contact laws are unchanged. The configuration records
the finite material-overlap audit and the limits of the guided fixture evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/wowrobo_cover_fit.json"


def rotation(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def registered_pose(receiver_position, receiver_quaternion, config=None):
    """Return cover root pose and insertion axis for a positioned receiver."""
    config = config or json.loads(CONFIG.read_text())
    receiver_r = rotation(receiver_quaternion)
    cover_r = receiver_r @ rotation(config["reference_relative_quaternion_wxyz"])
    position = np.asarray(receiver_position) + receiver_r @ np.asarray(
        config["reference_relative_position_m"]
    )
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, cover_r.ravel())
    axis = receiver_r @ np.asarray(config["insertion_axis_receiver_local"])
    return position, quaternion, axis


def write_alignment(source_alignment, source_scene, output):
    """Create a new fit-aligned full-state contract without changing its input."""
    source_alignment, source_scene, output = map(
        Path, (source_alignment, source_scene, output)
    )
    previous = json.loads(source_alignment.read_text())
    config = json.loads(CONFIG.read_text())
    model = mujoco.MjModel.from_xml_path(str(source_scene))
    data = mujoco.MjData(model)
    data.qpos[:] = previous["reference_qpos"]
    mujoco.mj_forward(model, data)
    receiver = data.body(config["receiver_body"])
    position, quaternion, axis = registered_pose(receiver.xpos, receiver.xquat, config)
    qa = int(model.joint(config["free_joint"]).qposadr[0])
    target = np.concatenate((position, quaternion))
    start = target.copy()
    start[:3] -= float(config["stand_off_m"]) * axis
    qstart = np.asarray(previous["start_qpos"]).copy()
    qtarget = np.asarray(previous["reference_qpos"]).copy()
    original_target = qtarget[qa : qa + 7].copy()
    qstart[qa : qa + 7] = start
    qtarget[qa : qa + 7] = target
    result = {
        "step": "cover",
        "piece": config["piece"],
        "receiver_body": config["receiver_body"],
        "mating_body": config["piece"],
        "mating_geom": "base_side_cover_printed_0",
        "root_joint": config["free_joint"],
        "qpos_address": qa,
        "axis_world": axis.tolist(),
        "authored_axial_travel_m": float(config["stand_off_m"]),
        "start_qpos": qstart.tolist(),
        "reference_qpos": qtarget.tolist(),
        "aligned_root_qpos": start.tolist(),
        "final_root_qpos": target.tolist(),
        "source_target_root_qpos": original_target.tolist(),
        "source_root_translation_delta_world_m": (
            target[:3] - original_target[:3]
        ).tolist(),
        "fit_configuration": config,
        "fit_configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_alignment": str(source_alignment.resolve()),
        "source_alignment_sha256": hashlib.sha256(
            source_alignment.read_bytes()
        ).hexdigest(),
        "source_scene_sha256": hashlib.sha256(source_scene.read_bytes()).hexdigest(),
        "scope": "Source-preserving rigid fit correction; contact law unchanged.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_alignment(
        arguments.source_alignment, arguments.source_scene, arguments.output
    )
