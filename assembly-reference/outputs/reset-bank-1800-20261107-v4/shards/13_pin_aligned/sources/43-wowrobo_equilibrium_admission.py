"""Require independently witnessed passive equilibrium before physical admission.

The saved reset is the actual equilibrium hold-start integration snapshot, with
its velocities, controls and warmstart intact. A separate physical validator still
owns material, loaded-grasp and reward checks; equilibrium alone never admits it.
"""

from __future__ import annotations

import copy
import hashlib

import mujoco
import numpy as np

from .wowrobo_equilibrium import integration_state, settle_and_hold


class MinimumResetValidator:
    """Initial-bank admission: material clearance and retained grasps; motion allowed."""

    def __init__(self, validator, *, settle_seconds=0.08, hold_seconds=0.2):
        self.validator = validator
        self.settle_seconds = settle_seconds
        self.hold_seconds = hold_seconds

    def __getattr__(self, name):
        return getattr(self.validator, name)

    def validate(self, stage, source, metadata):
        from .wowrobo_physical_resets import ASSEMBLY_ATTACHMENTS

        # Retain source mesh/Rtree caches; bound only memoized query results.
        # Recomputing the same geometric query does not alter admission.
        if len(self.validator.audit._direction_cache) > 20000:
            self.validator.audit._direction_cache.clear()
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = source.qpos
        d.qvel[:] = source.qvel
        d.ctrl[:] = source.ctrl
        for name, _, _, _, first in ASSEMBLY_ATTACHMENTS:
            d.eq_active[m.equality(name).id] = int(stage.split("_")[0]) >= first
        original = integration_state(m, d)
        original_ctrl = d.ctrl.copy()
        state, controls, witness, end = settle_and_hold(
            self.validator,
            stage,
            d,
            settle_seconds=self.settle_seconds,
            hold_seconds=self.hold_seconds,
            minimum_only=True,
            friction_multipliers=metadata.get("friction_multipliers"),
            pickup_requires_hold=metadata.get("pickup_requires_hold", True),
        )
        passed = witness["minimum_validity_pass"]
        return {
            "training_eligible": passed,
            "physical_admission_pass": passed,
            "valid_for_initial_reset_bank": passed,
            "rejection_reasons": witness["minimum_validity_rejection_reasons"],
            "metrics": {"minimum_validity": witness},
            "minimum_validity": witness,
            "minimum_reset_required": True,
            "stationarity_required": False,
            "profile": self.profile,
            "physics_profile": self.profile,
            "validation_model_signature": self.profile["model_sha256"],
            "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
            "physical_input_state": original,
            "physical_input_controls": original_ctrl,
            "settled_state": state,
            "settled_controls": controls,
            "qualification": "Initial reset minimum: no unintended overlaps and held parts retained; motion allowed",
            "replay_end_sha256": hashlib.sha256(end.tobytes()).hexdigest(),
        }


class EquilibriumRequiredValidator:
    """Adapt the existing bank validator without changing its source physics."""

    def __init__(
        self,
        validator,
        *,
        settle_seconds=2.0,
        hold_seconds=0.5,
        close_jaws=True,
        gravity_preload=False,
    ):
        if not all(np.isfinite(t) and t > 0 for t in (settle_seconds, hold_seconds)):
            raise ValueError("Equilibrium settling and hold windows must be positive")
        self.validator = validator
        self.settle_seconds = float(settle_seconds)
        self.hold_seconds = float(hold_seconds)
        self.close_jaws = bool(close_jaws)
        self.gravity_preload = bool(gravity_preload)

    def __getattr__(self, name):
        return getattr(self.validator, name)

    def _physical_source(self, source):
        # Incoming sampler data can belong to a nominal model with fewer
        # equalities. Never call mj_getState(physical_model, nominal_data).
        data = mujoco.MjData(self.model)
        data.qpos[:] = source.qpos
        data.qvel[:] = source.qvel
        data.ctrl[:] = source.ctrl
        if self.model.na:
            data.act[:] = source.act
        return data

    def validate(self, stage, source, metadata):
        model = self.model
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        original = self._physical_source(source)
        input_state = integration_state(model, original)
        input_controls = original.ctrl.copy()
        sampler_metadata = copy.deepcopy(metadata)
        try:
            saved, controls, equilibrium, replay_end = settle_and_hold(
                self.validator,
                stage,
                original,
                settle_seconds=self.settle_seconds,
                hold_seconds=self.hold_seconds,
                close_jaws=self.close_jaws,
                gravity_preload=self.gravity_preload,
                friction_multipliers=metadata.get("friction_multipliers"),
                pickup_requires_hold=metadata.get("pickup_requires_hold", True),
            )
        except (RuntimeError, ValueError) as error:
            saved, controls = input_state.copy(), input_controls.copy()
            replay_end = None
            equilibrium = {
                "stage": stage,
                "equilibrium_pass": False,
                "rejection_reasons": ["equilibrium_simulation_error"],
                "error": f"{type(error).__name__}: {error}",
                "training_eligible": False,
            }

        saved = np.asarray(saved).copy()
        controls = np.asarray(controls).copy()
        finite = np.isfinite(saved).all() and np.isfinite(controls).all()
        equilibrium_pass = (
            equilibrium.get("equilibrium_pass") is True
            and not equilibrium.get("rejection_reasons")
            and finite
        )
        reasons = []
        if not equilibrium_pass:
            reasons = ["passive_equilibrium_required"]
            reasons.extend(equilibrium.get("rejection_reasons", []))
            if not finite:
                reasons.append("nonfinite_equilibrium_state")
            physical = {
                "training_eligible": False,
                "physical_admission_pass": False,
                "metrics": {},
            }
        else:
            candidate = mujoco.MjData(model)
            mujoco.mj_setState(model, candidate, saved, spec)
            candidate.ctrl[:] = controls
            secondary_metadata = copy.deepcopy(metadata)
            secondary_metadata["equilibrium_input_preserved"] = True
            physical = dict(
                self.validator.validate(stage, candidate, secondary_metadata)
            )
            reasons.extend(physical.get("rejection_reasons", []))
            # A downstream registration/projection must not qualify a different
            # pose from the one whose unchanged-control hold was witnessed.
            admission_input = physical.get("physical_input_state")
            admission_spec = physical.get("state_spec", spec)
            if admission_input is not None:
                downstream = mujoco.MjData(model)
                mujoco.mj_setState(model, downstream, admission_input, admission_spec)
                same_pose = np.array_equal(downstream.qpos, candidate.qpos)
                same_velocity = np.array_equal(downstream.qvel, candidate.qvel)
                same_controls = np.array_equal(
                    physical.get("physical_input_controls"), controls
                )
                if not (same_pose and same_velocity and same_controls):
                    reasons.append("admission_input_changed_from_equilibrium")
            else:
                reasons.append("admission_input_equilibrium_identity_missing")
            if (
                not (
                    physical.get("physical_admission_pass") is True
                    and physical.get("training_eligible") is True
                )
                and not reasons
            ):
                reasons.append("physical_admission_not_passed")

        eligible = bool(
            equilibrium_pass
            and physical.get("physical_admission_pass") is True
            and physical.get("training_eligible") is True
            and not reasons
        )
        report = dict(physical)
        metrics = dict(physical.get("metrics", {}))
        metrics["equilibrium"] = equilibrium
        report.update(
            training_eligible=eligible,
            physical_admission_pass=eligible,
            rejection_reasons=sorted(set(reasons)),
            metrics=metrics,
            equilibrium=equilibrium,
            equilibrium_required=True,
            profile=self.profile,
            physics_profile=self.profile,
            validation_model_signature=self.profile["model_sha256"],
            state_spec=spec,
            physical_input_state=input_state,
            physical_input_controls=input_controls,
            settled_state=saved,
            settled_controls=controls,
            sampler_metadata=sampler_metadata,
            support_grasp_by_side=copy.deepcopy(
                metadata.get("support_grasp_by_side")
                or getattr(self.validator, "support_grasp_by_stage", {}).get(stage, {})
            ),
            active_equalities=[],
            saved_state_origin="equilibrium_hold_start",
            equilibrium_replay_end_sha256=hashlib.sha256(
                np.asarray(replay_end).tobytes()
            ).hexdigest()
            if replay_end is not None
            else None,
            qualification=(
                "Passive equilibrium and existing physical admission gates both pass"
                if eligible
                else "Not admitted; passive equilibrium and existing physical gates are both required"
            ),
        )
        if finite:
            candidate = mujoco.MjData(model)
            mujoco.mj_setState(model, candidate, saved, spec)
            report["active_equalities"] = [
                model.equality(i).name
                for i in range(model.neq)
                if candidate.eq_active[i]
            ]
        return report
