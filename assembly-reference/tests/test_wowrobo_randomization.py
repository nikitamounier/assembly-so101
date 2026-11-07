"""Distribution and native transform regressions, never training admission."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/wowrobo_randomization.json"
NOMINAL = ROOT / "assets_local/wowrobo/photo-resets-v3"


def test_distribution_covers_approved_starts_and_independent_bank_budget():
    config = json.loads(CONFIG.read_text())
    plan = json.loads((ROOT / "configs/wowrobo_reset_sequence.json").read_text())
    assert set(config["stages"]) == {stage["id"] for stage in plan["states"]}
    assert len(config["stages"]) == 25
    budget = config["sampling"]
    assert budget["train_per_stage"] == 640
    assert budget["eval_per_stage"] == 128
    assert budget["train_total"] == 16000
    assert budget["eval_total"] == 3200
    assert [band["count_per_stage"] for band in budget["train_difficulty"]] == [
        256,
        256,
        128,
    ]
    assert budget["independent_train_eval_seeds"]
    assert not budget["duplicate_padding"]
    # Insertion/near-start tolerances must be materially narrower than loose placement.
    stages = config["stages"]
    assert (
        stages["00_loose"]["translation_xy_m"]
        > 100 * stages["15_pin_near"]["lateral_radius_m"]
    )
    assert (
        stages["04_cover_entry"]["lateral_radius_m"]
        > stages["05_cover_near"]["lateral_radius_m"]
    )
    assert config["global"]["friction"]["multiplier_range"] == [0.97, 1.03]
    assert not config["global"]["connector"]["loose_unattached_parts_inherit"]


@pytest.fixture(scope="module")
def randomizer():
    if not (NOMINAL / "scene.xml").is_file():
        pytest.skip("Requires the reviewed local v3 CAD assets")
    from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

    return WowRoboRandomizer()


def nominal(randomizer, stage):
    data = mujoco.MjData(randomizer.model)
    with np.load(NOMINAL / "stages" / (stage + "_train.npz")) as saved:
        mujoco.mj_setState(
            randomizer.model,
            data,
            saved["states"][0],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        data.ctrl[:] = saved["controls"][0]
    mujoco.mj_forward(randomizer.model, data)
    return data


def pose(model, data, body):
    index = model.body(body).id
    return data.xpos[index].copy(), data.xmat[index].reshape(3, 3).copy()


def relative(model, data, parent, child):
    pp, pr = pose(model, data, parent)
    cp, cr = pose(model, data, child)
    return pr.T @ (cp - pp), pr.T @ cr


def test_seed_replay_and_split_separation_without_model_mutation(randomizer):
    model = randomizer.model
    friction = model.geom_friction.copy()
    first, receipt = randomizer.sample(
        "25_right_support", 97, split="train", difficulty=0.6, solve_arms=False
    )
    second, replay = randomizer.sample(
        "25_right_support", 97, split="train", difficulty=0.6, solve_arms=False
    )
    _, evaluation = randomizer.sample(
        "25_right_support", 97, split="eval", difficulty=0.6, solve_arms=False
    )
    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.ctrl, second.ctrl)
    assert receipt["friction_multipliers"] == replay["friction_multipliers"]
    assert receipt["friction_multipliers"] != evaluation["friction_multipliers"]
    assert set(receipt["friction_multipliers"]) == {
        "finger_part",
        "part_table",
        "part_part",
    }
    assert all(
        0.97 <= scale <= 1.03 for scale in receipt["friction_multipliers"].values()
    )
    np.testing.assert_array_equal(model.geom_friction, friction)
    assert receipt["training_eligible"] is False


def test_connector_swivel_is_coherent_and_leaves_base_and_loose_piece_fixed(randomizer):
    model = randomizer.model
    stage = "25_right_support"
    before = nominal(randomizer, stage)
    data, receipt = randomizer.sample(stage, 143, difficulty=0.8, solve_arms=False)
    assert 0 < abs(receipt["connector_swivel_degrees"]) <= 3
    for body in ["base_servo", "base_side_cover", "forearm_module"]:
        for actual, expected in zip(pose(model, data, body), pose(model, before, body)):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    for descendant in ["assembly_pin", "upper_arm_module"]:
        for actual, expected in zip(
            relative(model, data, "shoulder_connector", descendant),
            relative(model, before, "shoulder_connector", descendant),
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert not np.allclose(
        pose(model, data, "shoulder_connector")[1],
        pose(model, before, "shoulder_connector")[1],
    )


def test_connector_has_no_installed_swivel_before_its_installation(randomizer):
    data, receipt = randomizer.sample(
        "01_cover_pickup", 81, difficulty=0.5, solve_arms=False
    )
    before = nominal(randomizer, "01_cover_pickup")
    assert receipt["connector_swivel_degrees"] == 0
    for body in [
        "base_servo",
        "shoulder_connector",
        "assembly_pin",
        "upper_arm_module",
        "forearm_module",
    ]:
        for actual, expected in zip(
            pose(randomizer.model, data, body), pose(randomizer.model, before, body)
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_sampled_connector_keeps_real_horizontal_approach(randomizer):
    model = randomizer.model
    data, receipt = randomizer.sample(
        "08_connector_aligned", 137, difficulty=0.05, solve_arms=True
    )
    frame = data.site_xmat[model.site("right_gripperframe").id].reshape(3, 3)
    assert abs(frame[2, 0]) < np.sin(np.deg2rad(3))
    assert frame[2, 1] > 0, "Connector wrist camera must remain on the upper side"
    receiver = model.body("shoulder_connector").id
    expected = data.xmat[receiver].reshape(3, 3) @ [0, 1, 0]
    actual_xy = frame[:2, 0] / np.linalg.norm(frame[:2, 0])
    expected_xy = expected[:2] / np.linalg.norm(expected[:2])
    assert actual_xy @ expected_xy > np.cos(np.deg2rad(10))
    assert receipt["training_eligible"] is False


def test_sampled_transport_retains_actual_case_gripper_relationship(randomizer):
    model = randomizer.model
    before = nominal(randomizer, "20_upper_transport")
    data, receipt = randomizer.sample(
        "20_upper_transport", 151, difficulty=0.05, solve_arms=True
    )
    case = model.body("target_servo_2_pivot").id
    site = model.site("right_gripperframe").id

    def local_grasp(state):
        rotation = state.xmat[case].reshape(3, 3)
        position = rotation.T @ (state.site_xpos[site] - state.xpos[case])
        orientation = rotation.T @ state.site_xmat[site].reshape(3, 3)
        return position, orientation

    actual_position, actual_rotation = local_grasp(data)
    nominal_position, nominal_rotation = local_grasp(before)
    assert np.linalg.norm(actual_position - nominal_position) < 0.002
    relative_rotation = nominal_rotation.T @ actual_rotation
    angle = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
    assert angle < np.deg2rad(3)
    assert receipt["training_eligible"] is False


@pytest.mark.parametrize("stage", ["21_upper_far_side", "29_final_far_side"])
def test_sampled_far_end_still_touches_receiver_and_follows_its_axis(randomizer, stage):
    import trimesh

    from assembly_policy.so101.wowrobo_boundaries import world_mesh

    model = randomizer.model
    data, receipt = randomizer.sample(stage, 277, difficulty=0.7, solve_arms=False)
    evidence = randomizer.nominal_receipts[stage]["mating"]
    task = randomizer.definitions[stage]["task"]
    body = model.body(task).id
    receiver = model.body(evidence["receiver_body"]).id
    far = data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.array(
        evidence["far_point_piece_local_m"]
    )
    gaps = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != 7:
            continue
        ancestor = int(model.geom_bodyid[geom])
        while ancestor and ancestor != receiver:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != receiver:
            continue
        _, distance, _ = trimesh.proximity.closest_point(
            world_mesh(model, data, geom), [far]
        )
        gaps.append(float(distance[0]))
    assert gaps and min(gaps) <= 0.0005, (stage, gaps)
    held = model.body(evidence["held_servo_body"]).id
    case_axis = data.xmat[held].reshape(3, 3) @ [0, 0, -1]
    receiver_axis = data.xmat[receiver].reshape(3, 3) @ [0, 0, -1]
    assert case_axis @ receiver_axis > np.cos(np.deg2rad(1))
    assert receipt["training_eligible"] is False


def test_zero_difficulty_keeps_all_approved_nominal_geometry(randomizer):
    for stage in randomizer.definitions:
        before = nominal(randomizer, stage)
        data, receipt = randomizer.sample(stage, 173, difficulty=0, solve_arms=False)
        for body in [
            "base_servo",
            "base_side_cover",
            "shoulder_connector",
            "assembly_pin",
            "upper_arm_module",
            "forearm_module",
        ]:
            for actual, expected in zip(
                pose(randomizer.model, data, body), pose(randomizer.model, before, body)
            ):
                np.testing.assert_allclose(actual, expected, atol=1e-10, err_msg=stage)
        assert receipt["training_eligible"] is False


def test_loose_tangent_yaw_noise_changes_orientation_about_same_grip_center(randomizer):

    from scipy.spatial.transform import Rotation

    # Compare the actual transform to the known nominal+tangent transformation,
    # rather than merely checking the sampler's reported draw.
    before = nominal(randomizer, "00_loose")
    data, receipt = randomizer.sample("00_loose", 519, difficulty=1, solve_arms=False)
    draws = receipt["tangent_yaw_noise_degrees"]
    assert set(draws) == set(randomizer.sequence.order)
    assert all(abs(a) <= 2 for a in draws.values())
    assert any(abs(a) > 0.05 for a in draws.values())
    for i, piece in enumerate(randomizer.sequence.order):
        side = "left" if i % 2 == 0 else "right"
        c0, _ = randomizer.sequence.grip(before, piece)
        c1, _ = randomizer.sequence.grip(data, piece)
        shoulder = randomizer.sequence.layout.axes[side]
        a0 = np.arctan2(*(c0[:2] - shoulder[:2])[::-1])
        a1 = np.arctan2(*(c1[:2] - shoulder[:2])[::-1])
        yaw = a1 - a0 + np.deg2rad(draws[piece])
        rb = pose(randomizer.model, before, piece)[1]
        expected = Rotation.from_rotvec([0, 0, yaw]).as_matrix() @ rb
        np.testing.assert_allclose(
            pose(randomizer.model, data, piece)[1], expected, atol=1e-10
        )
