"""Version a reviewed bank with an explicit installed-cover supporting grasp."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def apply(source, output, grasp_path):
    source, output, grasp_path = map(Path, (source, output, grasp_path))
    if output.exists():
        raise ValueError("Use a new output directory; reference banks are immutable")
    candidate = json.loads(grasp_path.read_text())
    q = np.asarray(candidate["q_left"], dtype=float)
    calibration = candidate.get("grasp_calibration", candidate.get("spec"))
    if q.shape != (6,) or not np.isfinite(q).all() or not calibration:
        raise ValueError("Incomplete native cover grasp")
    override = copy.deepcopy(
        candidate.get(
            "support_grasp",
            {"piece": "base_side_cover", "grasp_calibration": calibration},
        )
    )
    override["piece"] = "base_side_cover"
    override["stable_support"] = True
    override["grasp_calibration"] = calibration
    override["source_grasp_receipt"] = str(grasp_path.resolve())
    override["qualification"] = (
        "Two-sided native finger geometry checked; loaded retention remains separately tested"
    )
    shutil.copytree(source, output)
    m = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    left_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    addresses = np.asarray(
        [m.joint("left_" + name).qposadr[0] for name in left_names], int
    )
    joint_ids = [m.joint("left_" + name).id for name in left_names]
    clipped = np.clip(q, m.jnt_range[joint_ids, 0], m.jnt_range[joint_ids, 1])
    if np.max(np.abs(clipped - q)) > 1e-6:
        raise ValueError("Reviewed grasp lies outside native joint limits")
    q = clipped
    actuators = [
        int(np.flatnonzero(m.actuator_trnid[:, 0] == m.joint("left_" + name).id)[0])
        for name in left_names
    ]
    receipts = json.loads((output / "stage-receipts.json").read_text())
    manifest = json.loads((output / "stages/manifest.json").read_text())
    affected = []
    proof = []

    def update(path):
        with np.load(path) as z:
            values = {name: z[name].copy() for name in z.files}
        before = values["states"][0].copy()
        controls = values["controls"][0].copy()
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, before, mujoco.mjtState(manifest["state_spec"]))
        old_q = d.qpos.copy()
        d.qpos[addresses] = q
        d.ctrl[:] = controls
        d.ctrl[actuators] = q
        mujoco.mj_forward(m, d)
        mask = np.ones(m.nq, bool)
        mask[addresses] = False
        assert np.array_equal(d.qpos[mask], old_q[mask]), "Target or right arm changed"
        ctrl_mask = np.ones(m.nu, bool)
        ctrl_mask[actuators] = False
        assert np.array_equal(d.ctrl[ctrl_mask], controls[ctrl_mask]), (
            "Other controls changed"
        )
        state = np.empty(
            mujoco.mj_stateSize(m, mujoco.mjtState(manifest["state_spec"]))
        )
        mujoco.mj_getState(m, d, state, mujoco.mjtState(manifest["state_spec"]))
        values["states"][0] = state
        values["controls"][0] = d.ctrl
        np.savez_compressed(path, **values)
        proof.append(
            {
                "file": str(path.relative_to(output)),
                "only_left_coordinates_changed": True,
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "control_sha256": hashlib.sha256(d.ctrl.tobytes()).hexdigest(),
            }
        )
        return state, d

    for receipt in receipts:
        stage = receipt["id"]
        definition = receipt["definition"]
        if definition["left_role"] != "hold_base_assembly":
            continue
        state, d = update(output / "stages" / f"{stage}_train.npz")
        receipt.setdefault("support_grasp_by_side", {})["left"] = copy.deepcopy(
            override
        )
        receipt["definition"]["left_support_piece"] = "base_side_cover"
        if str(receipt["definition"].get("photo_review_note", "")).startswith("Left support is lower"):
            receipt["definition"].pop("photo_review_note", None)
        receipt["arms"]["left"] = copy.deepcopy(candidate.get("info", {}))
        receipt["arms"]["left"].update(
            anchor_body="base_side_cover",
            role="hold_base_assembly",
            grasp_calibration=calibration,
            opening_radians=float(q[-1]),
            joint_positions=q[:5].tolist(),
            support_piece="base_side_cover",
        )
        receipt["state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
        receipt["control_sha256"] = hashlib.sha256(d.ctrl.tobytes()).hexdigest()
        receipt["definition"]["description"] += (
            " Left grips the installed base side cover by its opposed exterior side walls."
        )
        manifest["stage_by_task"][stage] = copy.deepcopy(receipt["definition"])
        manifest["stage_by_task"][stage]["support_grasp_by_side"] = {
            "left": copy.deepcopy(override)
        }
        affected.append(stage)
    update(output / "completed/24_upper_seated_train.npz")
    manifest["display_name"] = "Repaired CAD · cover support · reference states"
    manifest["randomized_display_name"] = (
        "Repaired CAD · cover support · reset variations"
    )
    manifest["support_grasp_revision"] = "installed-cover-sidewalls-v1"
    # Prior material target checks remain historical; fresh helper-aware audits follow.
    manifest.pop("nominal_validation_by_task", None)
    (output / "stages/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "stage-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    contract = json.loads((output / "reset-contract.json").read_text())
    for definition in contract["states"]:
        if definition["id"] in affected:
            definition["left_support_piece"] = "base_side_cover"
            if str(definition.get("photo_review_note", "")).startswith("Left support is lower"):
                definition.pop("photo_review_note", None)
    (output / "reset-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    result = {
        "source_bank": str(source.resolve()),
        "grasp_receipt": str(grasp_path.resolve()),
        "affected_stage_ids": affected,
        "left_qpos": q.tolist(),
        "support_grasp": override,
        "parity": proof,
        "training_eligible": False,
    }
    (output / "cover-support-change.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    top = json.loads((output / "manifest.json").read_text())
    top.update(
        profile="wowrobo-" + output.name,
        source_nominal_bank=str(source.resolve()),
        support_grasp_revision="installed-cover-sidewalls-v1",
    )
    (output / "manifest.json").write_text(json.dumps(top, indent=2) + "\n")
    with (output / "README.md").open("a") as stream:
        stream.write(
            "\n\n## Installed-cover support correction\n\nLeft supports the installed base side cover throughout states17,18,20,21,22,23,25 and private24. Only the six LEFT coordinates/controls change; every target and RIGHT coordinate is bitwise preserved. See cover-support-change.json and the referenced native finger contact audit. Loaded retention remains separately qualified.\n"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "assets_local/wowrobo/photo-resets-repaired-v1",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets_local/wowrobo/photo-resets-repaired-cover-support-v1",
    )
    p.add_argument("--grasp", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(apply(args.source, args.output, args.grasp), indent=2))
