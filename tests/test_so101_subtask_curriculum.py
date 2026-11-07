import json
from pathlib import Path

import numpy as np
import pytest

from assembly_policy.so101.curriculum import FAMILIES, Builder, candidate_digest
from assembly_policy.so101.obs import PointTables
from assembly_policy.so101.scene import ASSEMBLY_TASKS, OBS_BODIES
from assembly_policy.so101.subtask_curriculum import publish


def test_balanced_publication_rejects_missing_groups(tmp_path):
    with pytest.raises(ValueError, match="Incomplete balanced group"):
        publish(tmp_path, tmp_path / "bank", dict(train_count=256, eval_count=64))
    assert not (tmp_path / "bank").exists()


def test_subtask_loose_retains_prefix_and_observes_actual_scene():
    b = Builder()
    try:
        points = PointTables(ASSEMBLY_TASKS)
        import torch

        for index, task in enumerate(ASSEMBLY_TASKS):
            for attempt in range(12):
                item = b.loose_candidate(task, 81000000 + 100 * index + attempt)
                assert item["predecessors"] == list(ASSEMBLY_TASKS[:index])
                accepted, receipt = b.qualify(item, warmup=False)
                if accepted is not None:
                    break
            assert accepted is not None, (task, receipt)
            assert receipt["support_mode"] == "table" and not receipt["active_squeeze"]
            assert receipt["predecessor_error_m"] < 0.001
            o = {k: torch.as_tensor(v) for k, v in b.env.observe().items()}
            pc = points(*[o[k] for k in ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")])
            assert pc.shape == (1, 1344) and torch.isfinite(pc).all()
            ids = points.which[index].cpu().numpy()
            for name, count in points.mating_counts[task].items():
                assert int((ids == OBS_BODIES.index(name)).sum()) == count
            assert b.m.neq == 0
    finally:
        b.close()


def test_family_evaluation_does_not_only_take_first_loose_rows():
    from assembly_policy.so101.evaluate import evaluation_indices

    labels = np.repeat(np.arange(5), 8)
    selected = evaluation_indices(labels, list(FAMILIES), 2)
    assert len(selected) == 10
    np.testing.assert_array_equal(np.bincount(labels[selected]), [2, 2, 2, 2, 2])
    with pytest.raises(ValueError, match="Missing evaluation family"):
        evaluation_indices(np.array([0]), ["partial"], 2)
    with pytest.raises(ValueError, match="Invalid evaluation family"):
        evaluation_indices(labels, ["loose", "loose"], 2)


def test_persisted_receipts_reject_bad_labels_limits_and_short_replay():
    import copy

    from assembly_policy.so101.subtask_curriculum import validate_row

    plan = dict(validator_identity={"code": "current"})
    receipt = dict(
        passed=True,
        ready_arm_checks=[dict(ok=True, sides=["right"])] * 26,
        validator_identity=plan["validator_identity"],
        diagnostic_timestep_s=0.0005,
        replay_seconds=1.0,
        support_mode="bilateral_grasp",
        tail_bilateral_samples=5,
        max_sampled_material_overlap_m=0.0001,
        peak_control_sampled_depth_m=0.0001,
        predecessor_error_m=0.0,
        max_grasp_slip_m=0.0001,
        max_grasp_rotation_slip_rad=0.001,
    )
    row = dict(
        side="left",
        task="motor",
        family="inhand",
        split="train",
        predecessors=[],
        state=[0.0, 1.0, 2.0],
        controls=[0.0] * 12,
        qualification=receipt,
        ready_arm=dict(sides=["right"], preparation=dict(ok=True)),
    )
    receipt["candidate_digest"] = candidate_digest(row)
    validate_row(row, "motor", "inhand", plan)
    for field, value in [
        ("max_sampled_material_overlap_m", 0.001),
        ("predecessor_error_m", 0.001),
        ("max_grasp_slip_m", float("nan")),
        ("replay_seconds", 0.04),
        ("tail_bilateral_samples", 0),
    ]:
        bad = copy.deepcopy(row)
        bad["qualification"][field] = value
        with pytest.raises(ValueError):
            validate_row(bad, "motor", "inhand", plan)
    with pytest.raises(ValueError, match="metadata"):
        validate_row(row, "motor", "partial", plan)
    bad = copy.deepcopy(row)
    bad["family"] = "partial"
    with pytest.raises(ValueError, match="qualification receipt"):
        validate_row(bad, "motor", "partial", plan)


def test_graceful_stop_skips_native_initialization(tmp_path, monkeypatch):
    from assembly_policy.so101 import subtask_curriculum as build

    (tmp_path / "STOP").write_text("stop")
    monkeypatch.setattr(build, "Builder", lambda: pytest.fail("Stopped worker constructed physics"))
    result = build.generate_group(("motor", "loose", dict(train_count=256, eval_count=64), str(tmp_path), []))
    assert result["stopped"] and not result["complete"]
    assert not list(tmp_path.glob("*.jsonl"))


def test_only_one_writer_can_run_a_build(tmp_path):
    from assembly_policy.so101.subtask_curriculum import build_lock

    with build_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with build_lock(tmp_path):
                pass
    with build_lock(tmp_path):
        pass


def test_balanced_publish_mapping_and_interrupted_publication(tmp_path, monkeypatch):
    """Exercise archive/SGS wiring with stub identities, without claiming physics."""
    import hashlib

    from assembly_policy.rl.sampler import ResetSampler
    from assembly_policy.so101 import subtask_curriculum as build
    from assembly_policy.train import load_banks

    folder = tmp_path / "build"
    folder.mkdir()
    identity = {"model_signature": "unit-test-model"}
    validator = {"code": "unit-test-validator"}
    plan = dict(
        environment=identity,
        validator_identity=validator,
        train_count=2,
        eval_count=2,
        generation_sha256=hashlib.sha256(Path(build.__file__).read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(build, "environment_identity", lambda: identity)
    monkeypatch.setattr(build, "validation_identity", lambda: validator)
    for ti, task in enumerate(ASSEMBLY_TASKS):
        for fi, family in enumerate(FAMILIES):
            rows = []
            for split, offset in [("train", 0), ("eval", 50)]:
                for n in range(2):
                    seed = 10000 * ti + 100 * fi + offset + n
                    q = dict(
                        passed=True,
                        ready_arm_checks=[
                            dict(ok=True, sides=["left", "right"] if family == "loose" else ["right"])
                        ]
                        * (33 if family == "pickup" else 26),
                        validator_identity=validator,
                        diagnostic_timestep_s=0.0005,
                        replay_seconds=1.28 if family == "pickup" else 1.0,
                        support_mode="table" if family == "loose" else "bilateral_grasp",
                        tail_bilateral_samples=5,
                        max_sampled_material_overlap_m=0.0001,
                        peak_control_sampled_depth_m=0.0001,
                        predecessor_error_m=0.0,
                        max_grasp_slip_m=0.0001,
                        max_grasp_rotation_slip_rad=0.001,
                    )
                    row = dict(
                        task=task,
                        family=family,
                        split=split,
                        seed=seed,
                        predecessors=list(ASSEMBLY_TASKS[:ti]),
                        state=[0.0, float(seed) + 1.0, 0.0],
                        controls=[0.0] * 12,
                        qualification=q,
                        side="left",
                        ready_arm=dict(
                            sides=["left", "right"] if family == "loose" else ["right"],
                            preparation=dict(ok=True),
                        ),
                    )
                    q["candidate_digest"] = candidate_digest(row)
                    if family == "partial":
                        row["halfstep_qualification"] = q | dict(diagnostic_timestep_s=0.00025)
                    rows.append(row)
            (folder / f"{task}_{family}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "bank"
    build.publish(folder, out, plan)
    states, tasks, families = load_banks(ASSEMBLY_TASKS, out)
    assert len(states) == 50
    weights = ResetSampler(len(states)).weights()
    weights /= weights.sum()
    for ti in range(5):
        assert np.isclose(weights[tasks == ti].sum(), 0.2)
        for fi in range(5):
            assert np.isclose(weights[(tasks == ti) & (families == fi)].sum(), 0.04)
    for task in ASSEMBLY_TASKS:
        with np.load(out / f"{task}_eval.npz") as z:
            assert len(z["states"]) == 2 and np.all(z["family"] == 0)
        with np.load(out / "family_eval" / f"{task}_eval.npz") as z:
            assert len(z["states"]) == 10
    calls = iter([identity, {"model_signature": "changed"}])
    monkeypatch.setattr(build, "environment_identity", lambda: next(calls))
    retry = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="source change during publication"):
        build.publish(folder, retry, plan)
    assert not retry.exists() and (tmp_path / ".retry.publishing").exists()
    monkeypatch.setattr(build, "environment_identity", lambda: identity)
    build.publish(folder, retry, plan)
    assert retry.exists() and not (tmp_path / ".retry.publishing").exists()


def test_duplicate_group_arguments_are_rejected_before_generation(tmp_path, monkeypatch):
    import sys

    from assembly_policy.so101 import subtask_curriculum as build

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subtask_curriculum",
            "--out",
            str(tmp_path / "bank"),
            "--tasks",
            "motor+motor",
            "--train-count",
            "2",
            "--eval-count",
            "2",
        ],
    )
    monkeypatch.setattr(build, "Builder", lambda: pytest.fail("Invalid selection constructed physics"))
    with pytest.raises(ValueError, match="duplicate task/family"):
        build.main()
    assert not (tmp_path / ".bank.building").exists()


def test_publication_requires_ready_preparation_and_successful_saved_replay():
    from assembly_policy.so101.subtask_curriculum import validate_row

    row = dict(
        task="motor",
        family="loose",
        split="train",
        side="left",
        predecessors=[],
        state=[0.0, 1.0],
        controls=[0.0] * 12,
    )
    with pytest.raises(ValueError, match="ready-arm preparation"):
        validate_row(row, "motor", "loose", {})
    row["ready_arm"] = dict(sides=["left", "right"], preparation=dict(ok=True))
    row["qualification"] = dict(passed=True, validator_identity={}, diagnostic_timestep_s=0.0005)
    row["qualification"]["candidate_digest"] = candidate_digest(row)
    with pytest.raises(ValueError, match="saved/replay checks"):
        validate_row(row, "motor", "loose", dict(validator_identity={}))
    row["qualification"]["ready_arm_checks"] = [dict(ok=True), dict(ok=False)]
    row["qualification"]["candidate_digest"] = candidate_digest(row)
    with pytest.raises(ValueError, match="saved/replay checks"):
        validate_row(row, "motor", "loose", dict(validator_identity={}))


def test_saved_state_and_controls_are_bound_to_receipts():
    from assembly_policy.so101.subtask_curriculum import validate_row

    row = dict(
        task="motor",
        family="loose",
        split="train",
        side="left",
        predecessors=[],
        state=[0.0, 1.0],
        controls=[0.0] * 12,
        ready_arm=dict(sides=["left", "right"], preparation=dict(ok=True)),
    )
    receipt = dict(
        passed=True,
        validator_identity={},
        diagnostic_timestep_s=0.0005,
        candidate_digest=candidate_digest(row),
    )
    row["qualification"] = receipt
    for field in ("state", "controls"):
        changed = row | {field: list(row[field])}
        changed[field][0] += 0.01
        with pytest.raises(ValueError, match="saved state/controls"):
            validate_row(changed, "motor", "loose", dict(validator_identity={}))


def test_direct_grasp_fallback_advances_independently_of_attempt_alias():
    from types import SimpleNamespace

    from assembly_policy.so101.subtask_curriculum import propose

    variants = []

    def candidate(task, family, seed, variant):
        variants.append(variant)
        return {}

    builder = SimpleNamespace(candidate=candidate, qualify=lambda row: (row, {}))
    for attempt in (3, 7):
        propose(builder, "horn", "near", 1000 + attempt, attempt, "train", [{}])
    assert variants == [0, 1]


def test_yoke_mating_uses_calibrated_fixed_receiver_proposals(monkeypatch):
    from types import SimpleNamespace

    from assembly_policy.so101 import subtask_curriculum as build

    seen = []

    def candidate(task, family, seed, variant, **kwargs):
        seen.append((family, variant, kwargs))
        return dict(calibrated=False)

    targets = []

    def calibrate(row, **kwargs):
        targets.append(kwargs["pad_target_m"])
        return row | dict(calibrated=True), {}

    def qualify(row):
        assert row["calibrated"]
        return row, {}

    builder = SimpleNamespace(candidate=candidate, calibrate_yoke_grasp=calibrate, qualify=qualify)
    monkeypatch.setattr(build, "yoke_gap", lambda _: 0.02127)
    for family in ("near", "partial"):
        build.propose(builder, "yoke", family, 1000, 0, "train", [{}])
    assert seen == [
        ("near", 32, dict(side_override="right", axis_gap=None, preserve_part_pose=False)),
        ("partial", 32, dict(side_override="right", axis_gap=0.02127, preserve_part_pose=True)),
    ]
    assert targets == [-0.000001, -0.000005]
