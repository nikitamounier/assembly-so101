import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from assembly_policy.so101.wowrobo_reset_framework import (
    check_library,
    difficulty_for_slot,
    grasp_id,
    relative_grasp,
    resize_sampling,
    world_grasp,
)


def sampling():
    return {
        "train_per_stage": 640,
        "eval_per_stage": 128,
        "train_difficulty": [
            {"probability": 0.4, "count_per_stage": 256, "interval": [0, 1 / 3]},
            {"probability": 0.4, "count_per_stage": 256, "interval": [1 / 3, 2 / 3]},
            {"probability": 0.2, "count_per_stage": 128, "interval": [2 / 3, 1]},
        ],
    }


def test_grasp_follows_motor_case_under_receiver_rotation():
    case_r = Rotation.from_euler("xyz", [12, 23, -41], degrees=True).as_matrix()
    case_p = np.array([0.1, -0.2, 0.03])
    site_r = Rotation.from_euler("xyz", [18, -7, 54], degrees=True).as_matrix()
    site_p = np.array([0.12, -0.17, 0.08])
    grasp = relative_grasp(case_p, case_r, site_p, site_r)
    r = Rotation.from_euler("zy", [9, -4], degrees=True).as_matrix()
    t = np.array([0.003, -0.004, 0.005])
    p, orientation = world_grasp(r @ case_p + t, r @ case_r, grasp)
    np.testing.assert_allclose(p, r @ site_p + t, atol=1e-14)
    np.testing.assert_allclose(orientation, r @ site_r, atol=1e-14)
    # No bracket angle is a parameter: the case, not the passive bracket,
    # carries the grasp transform.
    p2, _ = world_grasp(case_p, case_r, grasp)
    np.testing.assert_allclose(p2, site_p)


def test_1800_quota_and_retry_curriculum():
    spec = resize_sampling(sampling())
    assert [b["count_per_stage"] for b in spec["train_difficulty"]] == [24, 24, 12]
    assert 25 * (spec["train_per_stage"] + spec["eval_per_stage"]) == 1800
    for attempt in [0, 1, 17]:
        for slot in range(60):
            value = difficulty_for_slot(
                spec,
                slot + attempt * 60,
                "train",
                np.random.default_rng(attempt + slot),
            )
            band = 0 if slot < 24 else 1 if slot < 48 else 2
            assert band / 3 <= value <= (band + 1) / 3
    for slot in range(12):
        v = difficulty_for_slot(
            spec, slot + 12 * 3, "eval", np.random.default_rng(slot)
        )
        assert slot / 12 <= v < (slot + 1) / 12


def test_curriculum_does_not_silently_reuse_old_640_counts():
    spec = sampling()
    spec["train_per_stage"] = 60
    with pytest.raises(ValueError, match="sum"):
        difficulty_for_slot(spec, 0, "train", np.random.default_rng(1))
    resized = resize_sampling(sampling(), 7, 2)
    assert sum(b["count_per_stage"] for b in resized["train_difficulty"]) == 7


def test_library_detects_tampering_and_missing_bindings():
    record = {
        "anchor": "target_servo_2_pivot",
        "transform": {"position": [0.01, 0.02, 0.03], "rotation": np.eye(3).tolist()},
        "calibration": {"span_site_axis": 2, "opening_radians": 0.13},
        "source_witness_pass": False,
    }
    key = grasp_id(record)
    library = {
        "schema_version": 1,
        "grasps": {key: record},
        "bindings": {"stage": {"right": key}},
    }
    assert check_library(library)["grasps"][key]["source_witness_pass"] is False
    broken = copy.deepcopy(library)
    broken["grasps"][key]["transform"]["position"][0] += 0.001
    with pytest.raises(ValueError, match="identity"):
        check_library(broken)
    broken = copy.deepcopy(library)
    broken["bindings"]["stage"]["right"] = "absent"
    with pytest.raises(ValueError, match="Missing"):
        check_library(broken)


def test_merge_rejects_duplicate_states_across_splits(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from assembly_policy.so101.wowrobo_bank import write_json, write_npz

    script = (
        Path(__file__).resolve().parents[1] / "scripts/build_wowrobo_reset_framework.py"
    )
    spec = importlib.util.spec_from_file_location("framework_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    stage = "00_loose"
    plan = {
        "stage_ids": [stage],
        "train_per_stage": 1,
        "eval_per_stage": 1,
        "target_total": 2,
    }
    write_json(tmp_path / "plan.json", plan)
    shard = tmp_path / "shards" / stage / "qualified"
    write_json(
        shard / "manifest.json",
        {"complete": True, "model_signature": "same", "state_spec": 32767},
    )
    for split in ["train", "eval"]:
        write_npz(
            shard / f"{stage}_{split}.npz",
            states=np.array([[1.0, 2.0]]),
            controls=np.array([[0.0]]),
        )
    with pytest.raises(ValueError, match="Duplicate"):
        cli.merge(tmp_path, plan)
    assert not (tmp_path / "bank" / "manifest.json").exists()
    # A progress report cannot call partially filled quotas complete.
    write_json(
        tmp_path / "shards" / stage / "run.json",
        {
            "status": "incomplete_rejections",
            "progress": {"accepted_train": 1, "accepted_eval": 0},
        },
    )
    assert cli.status(tmp_path)["complete"] is False
    assert json.loads((tmp_path / "progress.json").read_text())["accepted"] == 1


def test_coordinator_rejects_another_owner(tmp_path):
    import fcntl
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts/build_wowrobo_reset_framework.py"
    )
    spec = importlib.util.spec_from_file_location("framework_lock_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with (tmp_path / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another coordinator"):
            cli.run(tmp_path, None)
