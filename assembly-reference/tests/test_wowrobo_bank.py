"""Bank accounting and actual resumable rejection-runner tests, not physics claims."""

import importlib
import json
from argparse import Namespace

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_bank import (
    changed_sources,
    digest,
    finite_json,
    sample_identity,
    seed_for_slot,
    write_json,
    write_npz,
)


def test_retry_seed_keeps_training_and_eval_strata():
    for split, quota in [("train", 640), ("eval", 128)]:
        for slot in range(quota):
            assert seed_for_slot(slot, 19, split, quota) % quota == slot
    with pytest.raises(ValueError):
        seed_for_slot(-1, 0, "train", 640)


def test_physical_duplicate_signature_and_finite_json():
    state = np.array([1.0, 2.0])
    a = {"finger_part": 0.98}
    assert sample_identity(state, a) == sample_identity(state.copy(), a)
    assert sample_identity(state, a) != sample_identity(state, {"finger_part": 1.01})
    assert finite_json({"a": np.array([float("inf"), float("nan"), 1.0])}) == {
        "a": [None, None, 1.0]
    }


def test_midrun_source_and_registration_creation_guard(tmp_path):
    source = tmp_path / "sampler.py"
    optional = tmp_path / "registration.json"
    source.write_text("original")
    guard = {source: digest(source.read_bytes()), optional: None}
    assert not changed_sources(guard)
    optional.write_text("{}")
    assert changed_sources(guard) == [str(optional)]
    optional.unlink()
    source.write_text("changed")
    assert changed_sources(guard) == [str(source)]


def test_full_runner_resamples_counts_only_admitted_and_resumes(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.build_wowrobo_randomized_bank")
    randomization = importlib.import_module(
        "assembly_policy.so101.wowrobo_randomization"
    )
    physics = importlib.import_module("assembly_policy.so101.wowrobo_physical_resets")
    nominal = tmp_path / "nominal"
    (nominal / "stages").mkdir(parents=True)
    xml = '<mujoco><worldbody><body><joint type="slide"/><geom size=".01"/></body></worldbody></mujoco>'
    (nominal / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    d = mujoco.MjData(m)
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    write_npz(nominal / "stages/00_loose_train.npz", states=[state], controls=[d.ctrl])
    write_json(
        nominal / "stages/manifest.json",
        {
            "model_signature": digest(xml.encode()),
            "state_spec": int(spec),
            "task_names": ["00_loose"],
            "family_names": ["test"],
            "stage_by_task": {},
        },
    )
    config = tmp_path / "config.json"
    write_json(config, {"sampling": {"train_per_stage": 2, "eval_per_stage": 1}})
    calls = []

    class Sampler:
        def __init__(self, *args, **kwargs):
            self.model = m

        def sample(self, stage, seed, split, difficulty=None):
            calls.append((stage, split, seed))
            data = mujoco.MjData(m)
            data.qpos[0] = seed * 0.01
            return data, {
                "seed": seed,
                "split": split,
                "ik_position_pass": True,
                "friction_multipliers": {
                    "finger_part": 1
                    + seed * 0.000001
                    + (0.001 if split == "eval" else 0)
                },
            }

    class Auditor:
        def __init__(self, *args, **kwargs):
            pass

        def audit(self, *args, **kwargs):
            raise AssertionError(
                "Physical admission must not repeat an ignored preliminary screen"
            )

    class Validator:
        def __init__(self, **kwargs):
            self.model = m
            self.scene_path = nominal / "scene.xml"
            self.profile = {"model_sha256": digest(xml.encode())}

        def validate(self, stage, data, metadata):
            result = np.empty(mujoco.mj_stateSize(m, spec))
            mujoco.mj_getState(m, data, result, spec)
            return {
                "training_eligible": metadata["seed"] != 0,
                "rejection_reasons": [] if metadata["seed"] else ["test_rejection"],
                "state_spec": int(spec),
                "settled_state": result,
                "settled_controls": data.ctrl.copy(),
                "validation_model_signature": digest(xml.encode()),
            }

    monkeypatch.setattr(randomization, "WowRoboRandomizer", Sampler)
    monkeypatch.setattr(script, "NativeBoundaryAudit", Auditor)
    monkeypatch.setattr(physics, "WowRoboPhysicalValidator", Validator)
    args = Namespace(
        assets=tmp_path,
        nominal=nominal,
        config=config,
        output=tmp_path / "out",
        validation="physical",
        full=True,
        train_per_stage=1,
        eval_per_stage=1,
        stages=None,
        difficulty=None,
        max_attempts=2,
        max_total_attempts=0,
        grid_manifest=None,
        physical_profile=None,
        pin_mesh=None,
        settle_seconds=0.15,
        hold_seconds=0.15,
    )
    assert script.build(args) == 0
    progress = json.loads((args.output / "progress.json").read_text())
    assert progress["accepted_train"] == 2
    assert progress["accepted_eval"] == 1
    assert progress["rejected_count"] == 2
    manifest = json.loads((args.output / "qualified/manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["model_signature"] == digest(xml.encode())
    with np.load(args.output / "qualified/00_loose_train.npz") as bank:
        assert len(bank["states"]) == 2
        assert bank["friction_categories"].tolist() == [
            "finger_part",
            "part_table",
            "part_part",
        ]
        assert bank["friction_multipliers"].shape == (2, 3)
        np.testing.assert_array_equal(bank["friction_multipliers"][:, 1:], 1.0)
        assert np.all(bank["friction_multipliers"][:, 0] > 1.0)
    original_calls = len(calls)
    assert script.build(args) == 0
    assert len(calls) == original_calls
    # Resume refuses corrupt immutable state data before recounting it.
    archive = next((args.output / "attempts").glob("*.npz"))
    write_npz(archive, states=[state + 1], controls=[d.ctrl])
    with pytest.raises(ValueError, match="hash mismatch"):
        script.build(args)

    # A divergent physics result is retained as a rejection, never published as
    # nonfinite viewer data or counted toward a training quota.
    original_validate = Validator.validate

    def diverged(self, stage, data, metadata):
        result = original_validate(self, stage, data, metadata)
        result["physical_input_state"] = result["settled_state"].copy()
        result["physical_input_controls"] = result["settled_controls"].copy()
        result["settled_state"][:] = np.nan
        result["training_eligible"] = True
        return result

    monkeypatch.setattr(Validator, "validate", diverged)
    args.output = tmp_path / "diverged"
    args.max_attempts = 1
    assert script.build(args) == 2
    bad_progress = json.loads((args.output / "progress.json").read_text())
    assert bad_progress["accepted_train"] == bad_progress["accepted_eval"] == 0
    for path in (args.output / "attempts").glob("*.json"):
        row = json.loads(path.read_text())
        assert "nonfinite_physical_state" in row["rejection_reasons"]
        with np.load(args.output / row["archive"]) as saved:
            assert np.isfinite(saved["states"]).all()
