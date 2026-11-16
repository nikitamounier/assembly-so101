"""Episode-level sliding-friction draws applied to per-environment contacts.

MuJoCo regenerates contact friction on collision detection. Apply this hook on
EVERY fresh contact set, before constraint construction. Saved FULLPHYSICS states
do not contain these parameters: save the returned multipliers with reset metadata.
This module is not automatically wired into the training/contact backend.
"""

from collections.abc import Mapping

import mujoco
import numpy as np

CATEGORIES = ("finger_part", "part_table", "part_part")
_PARTS = {
    "base_servo",
    "base_side_cover",
    "shoulder_connector",
    "assembly_pin",
    "upper_arm_module",
    "forearm_module",
}


def sample_friction_multipliers(rng, fractional_range=0.03):
    """One draw per material-contact category, fixed for the entire episode."""
    if not 0 <= fractional_range < 1:
        raise ValueError("fractional_range must lie in [0,1)")
    return {
        key: float(rng.uniform(1 - fractional_range, 1 + fractional_range))
        for key in CATEGORIES
    }


def infer_geom_roles(model):
    """Infer review-model materials; explicit geom-ID roles can override these."""
    roles = {}
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if name in {"review_table", "table", "floor"}:
            roles[geom] = "table"
            continue
        body = int(model.geom_bodyid[geom])
        while body:
            name = model.body(body).name or ""
            if name in _PARTS:
                roles[geom] = "part"
                break
            if name in {
                "left_gripper",
                "right_gripper",
                "left_moving_jaw_so101_v1",
                "right_moving_jaw_so101_v1",
            }:
                roles[geom] = "finger"
                break
            # Cameras are separate materials, not fingertip pads.
            if "camera" in name:
                break
            body = int(model.body_parentid[body])
    return roles


def apply_contact_friction(
    model, data, multipliers, *, geom_roles=None, baseline_friction=None
):
    """Modify only data.contact sliding coefficients; return reusable baseline.

    Call once per freshly generated contact set. To reapply to the SAME set,
    provide the previous returned baseline; this prevents compounding draws.
    Torsional/rolling terms and shared model geom/pair parameters are untouched.
    Explicit pair coefficients and geom-priority mixing are already reflected in
    the generated baseline, so this scales the effective contact coefficient.
    """
    if not isinstance(multipliers, Mapping) or set(multipliers) != set(CATEGORIES):
        raise ValueError(
            f"Exactly these friction categories are required: {CATEGORIES}"
        )
    values = {k: float(multipliers[k]) for k in CATEGORIES}
    if any(not np.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("Friction multipliers must be finite and positive")
    roles = infer_geom_roles(model) if geom_roles is None else geom_roles
    current = data.contact.friction[: data.ncon]
    baseline = np.array(
        current if baseline_friction is None else baseline_friction,
        dtype=float,
        copy=True,
    )
    if baseline.shape != (data.ncon, 5) or not np.isfinite(baseline).all():
        raise ValueError("Baseline must match this fresh contact set, shape(ncon,5)")
    current[:] = baseline
    for i in range(data.ncon):
        a, b = (roles.get(int(g)) for g in data.contact.geom[i])
        category = None
        if {a, b} == {"finger", "part"}:
            category = "finger_part"
        elif {a, b} == {"table", "part"}:
            category = "part_table"
        elif a == b == "part":
            category = "part_part"
        if category:
            current[i, :2] = baseline[i, :2] * values[category]
    return baseline


def apply_friction_and_rebuild_constraints(
    model, data, multipliers, *, geom_roles=None, baseline_friction=None
):
    """Euler split-step hook, after mj_step1 and before mj_step2.

    mj_step1 has ALREADY assembled constraints. Contact edits alone are too late.
    Rebuild their Jacobians/islands/inertia and velocity references here. This
    deliberately supports the Euler pipeline only, not RK4/discrete/implicit
    pipelines or a claim of integration with custom native collision callbacks.
    Any control callback observing constraints must be invoked after this hook.
    """
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        raise ValueError("This verified rebuild hook supports Euler split steps only")
    baseline = apply_contact_friction(
        model,
        data,
        multipliers,
        geom_roles=geom_roles,
        baseline_friction=baseline_friction,
    )
    mujoco.mj_makeConstraint(model, data)
    mujoco.mj_island(model, data)
    mujoco.mj_projectConstraint(model, data)
    mujoco.mj_transmission(model, data)
    mujoco.mj_fwdVelocity(model, data)
    return baseline
