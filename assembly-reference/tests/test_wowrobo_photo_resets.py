"""Photo-contract and exported native-pose checks, not training qualification.

The generator is deliberately not run here: a reset-authoring optimization is
not a regression test. Replay the final review export and audit its real poses.
Set WOWROBO_PHOTO_RESET_BANK to inspect another explicitly selected export.
"""

import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from assembly_policy.so101.wowrobo_boundaries import audit_grippers, world_mesh
from assembly_policy.so101.wowrobo_photo_resets import PLAN

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_LIMIT_M = 0.0005
ORDER = [
    "base_side_cover",
    "shoulder_connector",
    "assembly_pin",
    "upper_arm_module",
    "forearm_module",
]


def test_photo_contract_coverage_and_nonactive_arm_roles():
    plan = json.loads(PLAN.read_text())
    states = plan["states"]
    by = {s["id"]: s for s in states}
    assert len(by) == len(states) == 25
    assert not {2, 6, 11, 16, 19, 24, 27, 32}.intersection(
        int(stage["id"].split("_")[0]) for stage in states
    )
    assert sum(not s["terminal"] for s in states) == 25
    assert sum(s["terminal"] for s in states) == 0
    assert all(not s["training_reset_candidate"] for s in states if s["terminal"])
    assert {p for s in states for p in s["reference_images"]} == {
        f"IMG_{n}"
        for n in [
            1607,
            1608,
            1609,
            1610,
            1611,
            1612,
            1613,
            1614,
            1615,
            1616,
            1617,
            1618,
            1621,
            1622,
            1623,
            1624,
        ]
    }
    for state in states:
        if state["task"] == "assembly_pin":
            assert state["right_role"] == "hold_connector"
        if (
            state["task"] == "upper_arm_module"
            and state["phase"] != "support_transition"
        ):
            assert state["left_role"] == "hold_base_assembly"
        if state["task"] == "forearm_module" and state["phase"] != "support_transition":
            assert state["right_role"] == "hold_upper_arm"
    assert by["17_left_support"]["left_role"] == "hold_base_assembly"
    assert by["17_left_support"]["right_role"] == "hold_connector"
    assert by["25_right_support"]["left_role"] == "hold_base_assembly"
    assert by["25_right_support"]["right_role"] == "hold_upper_arm"
    assert by["18_upper_floor_grasp"]["phase"] == "floor_grasp"
    assert by["26_final_floor_grasp"]["phase"] == "floor_grasp"
    requirements = plan["grasp_requirements"]
    assert requirements["upper_arm_module"]["case_only"]
    assert requirements["upper_arm_module"]["preserve_piece_orientation_in_states"] == [
        18,
        19,
    ]
    assert requirements["assembly_pin"]["thin_shaft_excluded"]
    assert requirements["assembly_pin"]["diameter_m"] == pytest.approx(0.012)
    assert requirements["forearm_module"]["wrist_correction_degrees"] == 90


@pytest.fixture(scope="module")
def export():
    folder = Path(
        os.environ.get(
            "WOWROBO_PHOTO_RESET_BANK",
            ROOT / "assets_local/wowrobo/photo-resets-v3",
        )
    )
    required = [
        folder / name for name in ("manifest.json", "scene.xml", "stage-receipts.json")
    ]
    if not all(path.is_file() for path in required):
        pytest.skip(f"Final photo-reset export is not available: {folder}")
    plan = json.loads(PLAN.read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["profile"] == plan["profile"], (
        "Do not audit a superseded review export"
    )
    assert (
        manifest["model_signature"]
        == hashlib.sha256((folder / "scene.xml").read_bytes()).hexdigest()
    )
    bank = folder / manifest["bank"]
    bank_manifest = json.loads((bank / "manifest.json").read_text())
    receipts = json.loads((folder / "stage-receipts.json").read_text())
    by_id = {receipt["id"]: receipt for receipt in receipts}
    assert set(by_id) == {stage["id"] for stage in plan["states"]}
    assert not any(receipt["training_eligible"] for receipt in receipts)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    assert bank_manifest["state_spec"] == int(spec)

    def restore(stage_id):
        with np.load(bank / f"{stage_id}_train.npz", allow_pickle=False) as archive:
            assert len(archive["states"]) == 1
            state = archive["states"][0]
            controls = archive["controls"][0]
        receipt = by_id[stage_id]
        assert hashlib.sha256(state.tobytes()).hexdigest() == receipt["state_sha256"]
        assert (
            hashlib.sha256(controls.tobytes()).hexdigest() == receipt["control_sha256"]
        )
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state, spec)
        data.ctrl[:] = controls
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.qpos))
        np.testing.assert_array_equal(data.qvel, 0)
        return data

    return model, plan, by_id, restore


@pytest.fixture(scope="module")
def frozen_seated(export):
    """Removed success resets remain physical anchors in the frozen v2 bank."""
    model, _, _, _ = export
    folder = Path(
        os.environ.get(
            "WOWROBO_PHOTO_RESET_BANK",
            ROOT / "assets_local/wowrobo/photo-resets-v3",
        )
    )
    patch = json.loads((folder / "pose-patch.json").read_text())
    base = Path(patch["base"])
    manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["model_signature"] == patch["model_signature"]
    receipts = json.loads((base / "stage-receipts.json").read_text())
    seated = {}
    for receipt in receipts:
        definition = receipt["definition"]
        if not definition["terminal"]:
            continue
        with np.load(
            base / manifest["bank"] / f"{receipt['id']}_train.npz", allow_pickle=False
        ) as archive:
            state = archive["states"][0]
        assert hashlib.sha256(state.tobytes()).hexdigest() == receipt["state_sha256"]
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        seated[definition["task"]] = data
    assert set(seated) == set(ORDER)
    return seated


def test_export_controls_and_prerequisites(export, frozen_seated):
    model, plan, receipts, restore = export
    seated = frozen_seated
    for stage in plan["states"]:
        data = restore(stage["id"])
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator, 0]
            assert data.ctrl[actuator] == pytest.approx(
                data.qpos[model.jnt_qposadr[joint]], abs=1e-10
            )
        assert receipts[stage["id"]]["definition"] == stage
        if stage["task"] not in ORDER:
            continue
        for predecessor in ORDER[: ORDER.index(stage["task"])]:
            body = model.body(predecessor).id
            np.testing.assert_allclose(
                data.xpos[body], seated[predecessor].xpos[body], atol=1e-9
            )
            np.testing.assert_allclose(
                data.xmat[body], seated[predecessor].xmat[body], atol=1e-9
            )


def _assert_native_pad_contacts(
    model, data, side, target_name, source_body_name, face_axis, gap_limit_m=0.0018
):
    """Use every inner-pad vertex and face center, not the coarse SDF's80 samples."""
    frame = data.site_xmat[model.site(side + "_gripperframe").id].reshape(3, 3)
    site = data.site_xpos[model.site(side + "_gripperframe").id]
    source = model.body(source_body_name).id
    source_rotation = data.xmat[source].reshape(3, 3)
    target = world_mesh(model, data, model.geom(target_name).id)
    proofs = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != 7:
            continue
        body_name = model.body(model.geom_bodyid[geom]).name
        mesh_name = model.mesh(model.geom_dataid[geom]).name
        if body_name == side + "_gripper" and "wrist_roll_follower" in mesh_name:
            sign = 1
        elif body_name == side + "_moving_jaw_so101_v1":
            sign = -1
        else:
            continue
        mesh = world_mesh(model, data, geom)
        points = np.vstack([mesh.vertices, mesh.triangles_center])
        normals = np.vstack([mesh.vertex_normals, mesh.face_normals])
        local = (points - site) @ frame
        normal_site = normals @ frame
        inner = (
            (local[:, 0] > -0.035)
            & (local[:, 0] < 0.004)
            & (np.abs(local[:, 1]) < 0.010)
            & (normal_site[:, 2] * sign > 0.25)
        )
        points, normals = points[inner], normals[inner]
        assert len(points), body_name
        _, distance, triangles = trimesh.proximity.closest_point(target, points)
        target_normals = target.face_normals[triangles]
        normal_dot = np.einsum("ij,ij->i", target_normals, normals)
        normal_source = target_normals @ source_rotation
        # Permit a bevel on the intended side face, while requiring the
        # requested source-frame face normal to dominate. Antiparallel
        # normals exclude a near miss at the back of a finger.
        eligible = (np.abs(normal_source[:, face_axis]) > 0.8) & (normal_dot < -0.4)
        assert np.any(eligible), (
            f"{body_name} has no opposed intended side-face contact"
        )
        candidates = np.flatnonzero(eligible)
        best = candidates[np.argmin(distance[candidates])]
        assert distance[best] <= gap_limit_m, (
            body_name,
            float(distance[best]),
            float(normal_dot[best]),
        )
        proofs.append((float(distance[best]), normal_source[best, face_axis]))
    assert len(proofs) == 2
    assert proofs[0][1] * proofs[1][1] < 0, (
        "Fingers must approach opposite intended side faces"
    )
    return proofs


def _assert_native_case_pad_contacts(model, data, side, servo, gap_limit_m=0.0018):
    return _assert_native_pad_contacts(
        model,
        data,
        side,
        f"servo_{servo}_case",
        f"target_servo_{servo}_pivot",
        1,
        gap_limit_m=gap_limit_m,
    )


def test_18_preserves_piece_orientation_and_pinches_black_case(export):
    model, _, receipts, restore = export
    initial = restore("00_loose")
    body = model.body("upper_arm_module").id
    manifest = json.loads(
        (ROOT / "assets_local/wowrobo/components-v2/manifest.json").read_text()
    )
    bounds = np.array(manifest["constituents"]["servo_2_case"]["bounds_m"])
    case = model.body("target_servo_2_pivot").id
    for stage_id in ["18_upper_floor_grasp"]:
        data = restore(stage_id)
        np.testing.assert_allclose(data.xmat[body], initial.xmat[body], atol=1e-9)
        arm = receipts[stage_id]["arms"]["right"]
        assert arm["span_site_axis"] == 2
        assert arm["grasp_calibration"]["target"] == "servo_2_case"
        check = arm["boundary_refinement"]
        _assert_native_case_pad_contacts(model, data, "right", 2)
        points = np.array(check["contact_points_world_m"])
        local = (points - data.xpos[case]) @ data.xmat[case].reshape(3, 3)
        # The free front/back case faces are Y-normal; X-normal shaft-end
        # faces are occupied by the printed yoke and are the wrong target.
        np.testing.assert_allclose(np.sort(local[:, 1]), bounds[:, 1], atol=0.002)
        assert np.all(local[:, 0] >= bounds[0, 0] - 0.002)
        assert np.all(local[:, 0] <= bounds[1, 0] + 0.002)
        assert np.all(local[:, 2] >= bounds[0, 2] - 0.002)
        assert np.all(local[:, 2] <= bounds[1, 2] + 0.002)


@pytest.mark.parametrize(
    "stage_id",
    ["26_final_floor_grasp", "29_final_far_side", "30_final_half"],
)
def test_final_piece_wrist_pinches_servo_case_faces(export, stage_id):
    model, _, receipts, restore = export
    data = restore(stage_id)
    arm = receipts[stage_id]["arms"]["left"]
    assert arm["span_site_axis"] == 2
    assert arm["grasp_calibration"]["target"] == "servo_3_case"
    check = arm["boundary_refinement"]
    _assert_native_case_pad_contacts(model, data, "left", 3)
    case = model.body("target_servo_3_pivot").id
    points = np.array(check["contact_points_world_m"])
    local = (points - data.xpos[case]) @ data.xmat[case].reshape(3, 3)
    manifest = json.loads(
        (ROOT / "assets_local/wowrobo/components-v2/manifest.json").read_text()
    )
    bounds = np.array(manifest["constituents"]["servo_3_case"]["bounds_m"])
    np.testing.assert_allclose(np.sort(local[:, 1]), bounds[:, 1], atol=0.002)


def test_pin_pad_contacts_stay_on_upper_grip_cylinder(export):
    model, plan, receipts, restore = export
    body = model.body("assembly_pin").id
    for stage in plan["states"]:
        if stage["task"] != "assembly_pin":
            continue
        data = restore(stage["id"])
        check = receipts[stage["id"]]["arms"]["left"]["boundary_refinement"]
        assert check["geometric_refinement_pass"], stage["id"]
        points = np.array(check["contact_points_world_m"])
        local = (points - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        assert np.all((local[:, 2] >= 0.001) & (local[:, 2] <= 0.027)), stage["id"]
        np.testing.assert_allclose(
            np.linalg.norm(local[:, :2], axis=1), 0.006, atol=0.002
        )
        if stage["phase"] != "pickup":
            axis = data.xmat[body].reshape(3, 3) @ np.array([0, 0, 1])
            assert np.dot(axis, [0, 0, -1]) > np.cos(np.deg2rad(1))


def _descends_from(model, body, ancestor):
    while body and body != ancestor:
        body = int(model.body_parentid[body])
    return body == ancestor


@pytest.mark.parametrize("stage_index", [21, 22, 23, 29, 30, 31])
def test_far_side_is_on_receiver_and_case_alignment_uses_receiver_frame(
    export, stage_index
):
    model, plan, receipts, restore = export
    definition = next(
        state for state in plan["states"] if state["id"].startswith(f"{stage_index:02d}_")
    )
    data = restore(definition["id"])
    evidence = receipts[definition["id"]]["mating"]
    piece = model.body(definition["task"]).id
    receiver = model.body(evidence["receiver_body"]).id
    case = model.body(evidence["held_servo_body"]).id
    receiver_rotation = data.xmat[receiver].reshape(3, 3)
    point = data.xpos[piece] + data.xmat[piece].reshape(3, 3) @ np.array(
        evidence["far_point_piece_local_m"]
    )
    expected = data.xpos[receiver] + receiver_rotation @ np.array(
        evidence["contact_receiver_local_m"]
    )
    np.testing.assert_allclose(point, expected, atol=1e-7)
    distances = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != 7 or not _descends_from(
            model, int(model.geom_bodyid[geom]), receiver
        ):
            continue
        mesh = world_mesh(model, data, geom)
        _, distance, _ = trimesh.proximity.closest_point(mesh, [point])
        distances.append(float(distance[0]))
    assert distances and min(distances) <= BOUNDARY_LIMIT_M, definition["id"]
    case_down = data.xmat[case].reshape(3, 3) @ [0, 0, -1]
    receiver_down = receiver_rotation @ [0, 0, -1]
    assert np.dot(case_down, receiver_down) > np.cos(np.deg2rad(1)), definition["id"]
    if stage_index >= 29:
        # A world-vertical case would be wrong here: the receiving upper arm
        # is tilted and its pocket frame supplies the insertion direction.
        assert np.dot(receiver_down, [0, 0, -1]) < np.cos(np.deg2rad(5))


@pytest.mark.parametrize(
    "stage_id",
    ["18_upper_floor_grasp", "25_right_support", "30_final_half"],
)
def test_critical_gripper_boundaries_retain_half_mm_limit(export, stage_id):
    model, _, _, restore = export
    results = audit_grippers(model, restore(stage_id))
    failures = [pair for pair in results if pair["penetration_m"] > BOUNDARY_LIMIT_M]
    assert not failures, f"{stage_id}: {failures}"


@pytest.mark.parametrize(
    "before,after",
    [
        ("29_final_far_side", "30_final_half"),
        ("30_final_half", "31_final_near"),
    ],
)
def test_final_insertion_retains_camera_side_of_held_grasp(export, before, after):
    model, _, _, restore = export
    site = model.site("left_gripperframe").id
    first = restore(before).site_xmat[site].reshape(3, 3)[:, 1]
    second = restore(after).site_xmat[site].reshape(3, 3)[:, 1]
    # Site Y points to the camera side of the native gripper. A180-degree
    # wrist-roll branch switch changes that hemisphere and breaks a retained
    # grasp even when both isolated snapshots have good surface clearance.
    assert np.dot(first, second) > 0, (
        f"Held-grasp wrist branch flips: {before} -> {after}"
    )


@pytest.mark.parametrize("stage_index", [8, 9, 10, 12, 13, 14, 15, 17])
def test_connector_push_is_parallel_horizontal_and_retains_its_grasp(
    export, stage_index
):
    model, plan, _, restore = export
    stage_id = next(
        state["id"] for state in plan["states"] if state["id"].startswith(f"{stage_index:02d}_")
    )
    data = restore(stage_id)
    body = model.body("shoulder_connector").id
    site = model.site("right_gripperframe").id
    receiver_rotation = data.xmat[body].reshape(3, 3)
    hand_rotation = data.site_xmat[site].reshape(3, 3)
    expected = receiver_rotation @ np.array([0, 1, 0])
    approach = hand_rotation[:, 0]
    # Horizontal means parallel to the table. The native asymmetric jaw gap
    # and five arm DOFs require a small planar yaw to center the pinch; that
    # must not be confused with the wrong downward push seen in the old pose.
    approach_xy = approach[:2] / np.linalg.norm(approach[:2])
    expected_xy = expected[:2] / np.linalg.norm(expected[:2])
    assert np.dot(approach_xy, expected_xy) > np.cos(np.deg2rad(10)), stage_id
    assert abs(approach[2]) < np.sin(np.deg2rad(3)), stage_id
    if stage_index >= 12:
        initial = restore("12_pin_pickup")
        initial_receiver = initial.xmat[body].reshape(3, 3)
        local_position = receiver_rotation.T @ (data.site_xpos[site] - data.xpos[body])
        initial_position = initial_receiver.T @ (
            initial.site_xpos[site] - initial.xpos[body]
        )
        local_rotation = receiver_rotation.T @ hand_rotation
        initial_rotation = initial_receiver.T @ initial.site_xmat[site].reshape(3, 3)
        np.testing.assert_allclose(local_position, initial_position, atol=1e-6)
        np.testing.assert_allclose(local_rotation, initial_rotation, atol=1e-5)
        jaw = model.joint("right_gripper").qposadr[0]
        assert data.qpos[jaw] == pytest.approx(initial.qpos[jaw], abs=1e-6)


@pytest.mark.parametrize("stage_index", [7, 8, 9, 10])
def test_connector_exterior_side_pinches_and_half_mm_boundaries(export, stage_index):
    model, plan, _, restore = export
    stage_id = next(
        state["id"] for state in plan["states"] if state["id"].startswith(f"{stage_index:02d}_")
    )
    data = restore(stage_id)
    _assert_native_pad_contacts(
        model, data, "right", "shoulder_connector_printed_1", "shoulder_connector", 0
    )
    results = audit_grippers(model, data)
    failures = [pair for pair in results if pair["penetration_m"] > BOUNDARY_LIMIT_M]
    assert not failures, f"{stage_id}: {failures}"


def test_25_right_support_has_tight_bilateral_native_pad_contact(export):
    model, _, _, restore = export
    _assert_native_pad_contacts(
        model,
        restore("25_right_support"),
        "right",
        "upper_arm_module_printed_0",
        "upper_arm_module",
        0,
        gap_limit_m=0.0003,
    )


def test_right_support_joint_pose_is_retained_through_final_insertion(export):
    model, plan, _, restore = export
    joints = [
        model.joint("right_" + name).qposadr[0]
        for name in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    ]
    initial = restore("25_right_support").qpos[joints].copy()
    for stage in plan["states"]:
        if int(stage["id"].split("_")[0]) < 25:
            continue
        assert stage["right_role"] == "hold_upper_arm"
        np.testing.assert_allclose(
            restore(stage["id"]).qpos[joints],
            initial,
            atol=1e-9,
            rtol=0,
            err_msg=f"Right supporting grasp changes in {stage['id']}",
        )


@pytest.mark.parametrize("stage_id", ["20_upper_transport", "28_final_transport"])
def test_transport_link_sags_while_case_and_helper_poses_remain_fixed(export, stage_id):
    model, _, receipts, restore = export
    folder = Path(
        os.environ.get(
            "WOWROBO_PHOTO_RESET_BANK",
            ROOT / "assets_local/wowrobo/photo-resets-v3",
        )
    )
    patch = json.loads((folder / "pose-patch.json").read_text())
    # Preserve the actual preceding v3 support pose, which is tighter than v2.
    # The archived reference must share the current native geometry.
    assert patch["model_signature"] == hashlib.sha256(
        (folder / "scene.xml").read_bytes()
    ).hexdigest()
    reference = folder / patch["sag_reference_folder"] / f"{stage_id}_train.npz"
    with np.load(reference, allow_pickle=False) as archive:
        state = archive["states"][0]
    assert hashlib.sha256(state.tobytes()).hexdigest() == patch[
        "sag_reference_state_sha256"
    ][stage_id]
    baseline = mujoco.MjData(model)
    mujoco.mj_setState(model, baseline, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, baseline)
    data = restore(stage_id)
    evidence = receipts[stage_id]["held_body_sag"]
    root_name, servo, side = (
        ("upper_arm_module", 2, "right")
        if stage_id == "20_upper_transport"
        else ("forearm_module", 3, "left")
    )
    assert evidence["piece"] == root_name
    assert evidence["held_servo_body"] == f"target_servo_{servo}_pivot"
    root = model.body(root_name).id
    case = model.body(evidence["held_servo_body"]).id
    far_local = np.array(evidence["far_point_piece_local_m"])
    before = baseline.xpos[root] + baseline.xmat[root].reshape(3, 3) @ far_local
    after = data.xpos[root] + data.xmat[root].reshape(3, 3) @ far_local
    assert np.linalg.norm(data.xmat[root] - baseline.xmat[root]) > 0.05
    assert after[2] < before[2] - 0.001, stage_id
    assert abs(evidence["angle_degrees"]) == pytest.approx(8)
    assert evidence["far_height_before_m"] == pytest.approx(before[2], abs=1e-10)
    assert evidence["far_height_after_m"] == pytest.approx(after[2], abs=1e-10)
    np.testing.assert_allclose(data.xpos[case], baseline.xpos[case], atol=1e-10, rtol=0)
    np.testing.assert_allclose(data.xmat[case], baseline.xmat[case], atol=1e-10, rtol=0)
    # Only the target free root and its compensating shaft may change. This
    # independently preserves both helper arms and every other loose component.
    allowed = np.zeros(model.nq, dtype=bool)
    root_address = model.joint(root_name + "_free").qposadr[0]
    allowed[root_address : root_address + 7] = True
    allowed[model.joint(f"target_servo_{servo}_shaft").qposadr[0]] = True
    np.testing.assert_allclose(data.qpos[~allowed], baseline.qpos[~allowed], atol=1e-12, rtol=0)
    _assert_native_case_pad_contacts(model, data, side, servo)
    results = audit_grippers(model, data)
    failures = [pair for pair in results if pair["penetration_m"] > BOUNDARY_LIMIT_M]
    assert not failures, f"{stage_id}: {failures}"
