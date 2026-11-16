"""Admission must certify the witnessed equilibrium snapshot, not another pose."""

import copy

import mujoco
import numpy as np
import pytest

import assembly_policy.so101.wowrobo_equilibrium_admission as admission
from assembly_policy.so101.wowrobo_equilibrium import integration_state


class PhysicalValidator:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string("""<mujoco>
        <worldbody><body><joint name="joint" type="hinge"/>
        <geom type="sphere" size=".01" mass=".1"/></body></worldbody>
        <actuator><position joint="joint" kp="10"/></actuator></mujoco>""")
        self.profile = {"model_sha256": "actual-physical-model"}
        self.scene_path = "physical.xml"
        self.calls = []
        self.pass_physics = True
        self.change_input = False

    def validate(self, stage, data, metadata):
        self.calls.append((stage, data.qpos.copy(), data.qvel.copy(), metadata))
        data = copy.copy(data)
        if self.change_input:
            data.qpos[0] += 0.01
        actual_input = integration_state(self.model, data)
        # The old physical validator may return its own later settled state.
        # That later state must never replace the independent hold-start witness.
        data.qpos[0] = 0.9
        return {
            "physical_admission_pass": self.pass_physics,
            "training_eligible": self.pass_physics,
            "rejection_reasons": [] if self.pass_physics else ["physical_material"],
            "metrics": {"material_pass": self.pass_physics},
            "physical_input_state": actual_input,
            "physical_input_controls": data.ctrl.copy(),
            "settled_state": integration_state(self.model, data),
            "settled_controls": data.ctrl.copy(),
            "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        }


@pytest.fixture
def setup(monkeypatch):
    validator = PhysicalValidator()
    source = mujoco.MjData(validator.model)
    source.qpos[0] = 0.1
    source.qvel[0] = 0.007
    source.ctrl[0] = 0.13
    witness = mujoco.MjData(validator.model)
    witness.qpos[0] = 0.2
    witness.qvel[0] = 0.003  # Deliberately retained, not cleared for publication.
    witness.ctrl[0] = 0.23
    saved = integration_state(validator.model, witness)
    state = {"equilibrium_pass": True, "rejection_reasons": []}
    calls = []

    def settle(v, stage, data, **kwargs):
        calls.append((v, stage, data.qvel.copy(), kwargs))
        return saved.copy(), witness.ctrl.copy(), dict(state), saved.copy()

    monkeypatch.setattr(admission, "settle_and_hold", settle)
    wrapper = admission.EquilibriumRequiredValidator(
        validator, settle_seconds=0.4, hold_seconds=0.2, gravity_preload=True
    )
    metadata = {
        "friction_multipliers": {
            "finger_part": 1.02,
            "part_table": 0.99,
            "part_part": 1.01,
        },
        "pickup_requires_hold": False,
        "seed": 12,
    }
    return validator, source, witness, saved, state, calls, wrapper, metadata


def test_failed_equilibrium_never_runs_or_admits_downstream_physics(setup):
    validator, source, _, saved, eq, calls, wrapper, metadata = setup
    eq.update(equilibrium_pass=False, rejection_reasons=["unloaded_finger_pad"])
    result = wrapper.validate("18_upper_floor_grasp", source, metadata)
    assert not result["physical_admission_pass"] and not result["training_eligible"]
    assert result["rejection_reasons"] == [
        "passive_equilibrium_required",
        "unloaded_finger_pad",
    ]
    assert not validator.calls
    assert result["equilibrium"] == result["metrics"]["equilibrium"]
    np.testing.assert_array_equal(result["settled_state"], saved)
    assert calls[0][3]["friction_multipliers"] == metadata["friction_multipliers"]
    assert calls[0][3]["pickup_requires_hold"] is False


def test_both_gates_preserve_exact_witness_velocity_and_metadata(setup):
    validator, source, witness, saved, _, _, wrapper, metadata = setup
    before = copy.deepcopy(metadata)
    result = wrapper.validate("18_upper_floor_grasp", source, metadata)
    assert result["physical_admission_pass"] and result["training_eligible"]
    assert result["rejection_reasons"] == []
    np.testing.assert_array_equal(result["settled_state"], saved)
    np.testing.assert_array_equal(result["settled_controls"], witness.ctrl)
    restored = mujoco.MjData(validator.model)
    mujoco.mj_setState(
        validator.model, restored, result["settled_state"], result["state_spec"]
    )
    np.testing.assert_array_equal(restored.qvel, witness.qvel)
    assert metadata == before == result["sampler_metadata"]
    assert validator.calls[0][3]["equilibrium_input_preserved"] is True
    assert "equilibrium_input_preserved" not in metadata
    assert wrapper.model is validator.model and wrapper.profile is validator.profile
    assert wrapper.scene_path == "physical.xml"


def test_equilibrium_pass_does_not_override_failed_material_gate(setup):
    validator, source, _, _, _, _, wrapper, metadata = setup
    validator.pass_physics = False
    result = wrapper.validate("18_upper_floor_grasp", source, metadata)
    assert result["equilibrium"]["equilibrium_pass"]
    assert not result["training_eligible"]
    assert result["rejection_reasons"] == ["physical_material"]


def test_secondary_pose_projection_cannot_qualify_a_different_start(setup):
    validator, source, _, saved, _, _, wrapper, metadata = setup
    validator.change_input = True
    result = wrapper.validate("18_upper_floor_grasp", source, metadata)
    assert not result["training_eligible"]
    assert "admission_input_changed_from_equilibrium" in result["rejection_reasons"]
    np.testing.assert_array_equal(result["settled_state"], saved)


def test_simulation_error_returns_complete_unqualified_integration_snapshot(
    setup, monkeypatch
):
    validator, source, _, _, _, _, wrapper, metadata = setup

    def fail(*args, **kwargs):
        raise RuntimeError("nonfinite settling")

    monkeypatch.setattr(admission, "settle_and_hold", fail)
    result = wrapper.validate("18_upper_floor_grasp", source, metadata)
    assert not result["training_eligible"] and not validator.calls
    assert "equilibrium_simulation_error" in result["rejection_reasons"]
    assert result["validation_model_signature"] == validator.profile["model_sha256"]
    np.testing.assert_array_equal(
        result["settled_state"], result["physical_input_state"]
    )
    np.testing.assert_array_equal(result["settled_controls"], source.ctrl)
