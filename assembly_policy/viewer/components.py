"""Present generated CAD components through the existing two-renderer viewer."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def component_collection(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "review.xml"))
    data = mujoco.MjData(model)
    with np.load(folder / "review_pose.npz", allow_pickle=False) as pose:
        data.qpos[:] = pose["qpos"]
        data.ctrl[:] = pose["ctrl"]
    mujoco.mj_forward(model, data)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    signature = hashlib.sha256((folder / "review.xml").read_bytes() + state.tobytes()).hexdigest()
    directory = folder / "viewer-bank"
    directory.mkdir(exist_ok=True)
    tasks = ["workcell", *manifest["pieces"]]
    focus = {"workcell": {"target": [0, -0.04, 0.12], "radius": 0.56}}
    for name, piece in manifest["pieces"].items():
        center = data.xpos[model.body(name).id] + np.array(piece["center_m"])
        extent = np.ptp(np.array(piece["bounds_m"]), axis=0)
        focus[name] = {"target": center.tolist(), "radius": max(0.035, float(max(extent)) * 0.65)}
    for name in tasks:
        np.savez_compressed(directory / f"{name}_train.npz", states=state[None], seed=[0], family=[0])
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_signature": signature,
                "state_spec": int(spec),
                "task_names": tasks,
                "family_names": ["Photo-based CAD review"],
                "reset_distribution": "cad_review",
                "display_name": "WowRobo · corrected components + camera-equipped followers",
                "default_viewer_collection": True,
                "focus_by_task": focus,
            },
            indent=2,
        )
        + "\n"
    )
    return model, signature, directory


def layout_collection(folder):
    """Load a geometry-qualified layout bank without fabricating training resets."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    signature = hashlib.sha256((folder / "scene.xml").read_bytes()).hexdigest()
    if signature != manifest["model_signature"]:
        raise ValueError("Layout model signature differs from its bank")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    return model, signature, folder / manifest["bank"]
