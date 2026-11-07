"""Measured repaired registrations, real hinge freedom and coherent grip replay."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_repaired_resets import fit_case_support_hinge

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "assets_local/wowrobo/photo-resets-repaired-v1"


@pytest.fixture(scope="module")
def bank():
    if not (BANK / "scene.xml").exists():
        pytest.skip("Build the local corrected-CAD nominal bank first")
    model = mujoco.MjModel.from_xml_path(str(BANK / "scene.xml"))
    receipts = {r["id"]: r for r in json.loads((BANK / "stage-receipts.json").read_text())}
    return model, receipts


def restore(model, folder, stage):
    data = mujoco.MjData(model)
    with np.load(BANK / folder / f"{stage}_train.npz") as z:
        mujoco.mj_setState(model, data, z["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = z["controls"][0]
    mujoco.mj_forward(model, data)
    return data


def relative(data, parent, child):
    p, r = data.body(parent).xpos, data.body(parent).xmat.reshape(3, 3)
    q, s = data.body(child).xpos, data.body(child).xmat.reshape(3, 3)
    return r.T @ (q-p), r.T @ s


def test_all_approved_ids_have_updated_grips_and_real_far_support(bank):
    model, receipts = bank
    original = {r["id"] for r in json.loads((ROOT / "assets_local/wowrobo/photo-resets-v3/stage-receipts.json").read_text())}
    assert set(receipts) == original
    assert len(receipts) == 25
    manifest = json.loads((BANK / "manifest.json").read_text())
    assert manifest["model_signature"] == hashlib.sha256((BANK / "scene.xml").read_bytes()).hexdigest()
    assert manifest["cad_revision"] == "insertion-fix-final-common-v1"
    for stage, receipt in receipts.items():
        assert receipt["registration_pass"], stage
        assert receipt["training_eligible"] is False
        for info in receipt["arms"].values():
            assert info["grasp_position_error_m"] < .0001
            assert info["span_vector_error"] < .02
        support = receipt["support_contact"]
        if support:
            assert support["valid"]
            assert support["gap_m"] < .00001
            assert support["upward_receiver_normal"][2] > .35
            assert receipt["mating"]["far_point_piece_local_m"] == support["far_point_piece_local_m"]
        data = restore(model, "stages", stage)
        for actuator in range(model.nu):
            address = model.jnt_qposadr[model.actuator_trnid[actuator, 0]]
            assert data.ctrl[actuator] == data.qpos[address]


def test_public_hinge_fitter_keeps_case_and_nonactive_coordinates(bank):
    model, receipts = bank
    stage = "23_upper_near"
    data = restore(model, "stages", stage)
    before = data.qpos.copy()
    p = data.body("target_servo_2_pivot").xpos.copy()
    r = data.body("target_servo_2_pivot").xmat.copy()
    address = model.joint("target_servo_2_shaft").qposadr[0]
    info = fit_case_support_hinge(model, data, "upper_arm_module", receipts[stage]["mating"], float(before[address]) + .01)
    np.testing.assert_allclose(data.body("target_servo_2_pivot").xpos, p, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.body("target_servo_2_pivot").xmat, r, atol=1e-12, rtol=0)
    mask = np.ones(model.nq, dtype=bool)
    root = model.joint("upper_arm_module_free").qposadr[0]
    mask[root:root+7] = False
    mask[address] = False
    np.testing.assert_array_equal(before[mask], data.qpos[mask])
    assert abs(info["hinge_delta_from_photo_rad"]) <= .12
    assert info["case_position_error_m"] < 1e-12


def test_completed_upper_case_uses_successful_endpoint_with_free_articulation(bank):
    model, _ = bank
    data = restore(model, "completed", "24_upper_seated")
    reference = mujoco.MjData(model)
    with np.load(ROOT / "outputs/insertion-fix-20261107/upper/canonical-common-v1/actual-endpoint-full51.npz") as z:
        reference.qpos[:] = z["qpos"]
    mujoco.mj_forward(model, reference)
    for actual, expected in zip(relative(data, "shoulder_connector", "target_servo_2_pivot"), relative(reference, "shoulder_connector", "target_servo_2_pivot")):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    hinge = model.joint("target_servo_2_shaft").qposadr[0]
    # Printed link articulates to its support; the successfully fitted case stays exact.
    assert abs(data.qpos[hinge] - reference.qpos[hinge]) > .1
    following = restore(model, "stages", "26_final_floor_grasp")
    for actual, expected in zip(relative(following, "shoulder_connector", "target_servo_2_pivot"), relative(data, "shoulder_connector", "target_servo_2_pivot")):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
