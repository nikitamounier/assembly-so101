"""Preparation-only support receipt diagnostic. Never edits the active bank.

Run from the canonical repository. Re-select native printed far-end support
witnesses in each actual saved state; do not move a part, clear velocities,
change controls, or treat geometry as new dynamic qualification.
"""
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_repaired_resets import RepairedResetSequence

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/reset-bank-1800-20261107-v2/references"
OUT = Path(__file__).resolve().parent
STAGES = ["21_upper_far_side", "22_upper_half", "23_upper_near", "29_final_far_side", "30_final_half", "31_final_near"]
receipts = json.loads((SOURCE / "stage-receipts.json").read_text())
by_stage = {r["id"]: r for r in receipts}
model = mujoco.MjModel.from_xml_string((SOURCE / "scene.xml").read_text())
fitter = object.__new__(RepairedResetSequence)
fitter.model = model
fitter.old_receipts = by_stage
fitter.definitions = {r["id"]: r["definition"] for r in receipts}
results = {}
for stage in STAGES:
    old = by_stage[stage]
    data = mujoco.MjData(model)
    archive = SOURCE / "stages" / (stage + "_train.npz")
    with np.load(archive) as saved:
        mujoco.mj_setState(model, data, saved["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = saved["controls"][0]
    mujoco.mj_forward(model, data)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    support = fitter._support(data, stage)
    assert np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qvel, qvel)
    assert np.array_equal(data.ctrl, ctrl)
    proposed = copy.deepcopy(old)
    proposed["mating"] = support["mating_evidence"]
    proposed["support_contact"] = support
    proposed["support_receipt_recalibration"] = {
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "qpos_unchanged": True,
        "qvel_unchanged": True,
        "controls_unchanged": True,
        "source_qvel_norm": float(np.linalg.norm(qvel)),
        "geometry_only": True,
        "native_minimum_witness_preserved": bool(old.get("minimum_validity",{}).get("minimum_validity_pass",False)),
        "scope": "Recomputed contact evidence in actual witnessed body frames; no new physical admission claim",
    }
    if not support["valid"]:
        proposed["support_receipt_recalibration"]["warning"] = "Actual minimum-valid moving reference does not meet legacy 0.1mm instantaneous far-support gap; do not snap it or falsely claim support"
    (OUT / (stage + ".json")).write_text(json.dumps(proposed, indent=2) + "\n")
    result = {
        "actual_far_patch_gap_m": support["gap_m"],
        "old_claimed_gap_m": old["support_contact"]["gap_m"],
        "old_fixed_point_actual_gap_m": support["old_photo_witness_gap_m"],
        "valid_support_geometry": support["valid"],
        "far_local_change_m": float(np.linalg.norm(np.asarray(proposed["mating"]["far_point_piece_local_m"]) - old["mating"]["far_point_piece_local_m"])),
        "actual_qvel_norm": float(np.linalg.norm(qvel)),
        "upward_receiver_normal": support["upward_receiver_normal"],
    }
    results[stage] = result
    print(stage, json.dumps(result), flush=True)
(OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
