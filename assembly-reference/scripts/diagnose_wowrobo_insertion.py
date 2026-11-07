"""Matched contact/friction interventions from the same saved insertion stall."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode
from scripts.probe_wowrobo_aligned_force import SPECS, descendants

ROOT = Path(__file__).resolve().parents[1]


def configure(model, piece, receivers, contacts):
    active = descendants(model, piece)
    fixtures = set().union(*(descendants(model, p) for p in receivers))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.body_contype[:] = 0
    model.body_conaffinity[:] = 0
    if contacts:
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SDF:
                continue
            b = int(model.geom_bodyid[g])
            if b in active:
                model.geom_contype[g], model.geom_conaffinity[g] = 1, 2
            elif b in fixtures:
                model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
            model.body_contype[b] |= model.geom_contype[g]
            model.body_conaffinity[b] |= model.geom_conaffinity[g]
    return active


def reactions(model, data, active, axis):
    rows = []
    for k in range(data.ncon):
        con = data.contact[k]
        a, b = (int(x) for x in con.geom)
        sign = 1 if int(model.geom_bodyid[b]) in active else -1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, k, force)
        frame = con.frame.reshape(3, 3)
        normal = sign * force[0] * frame[0]
        friction = sign * (force[1:3] @ frame[1:3])
        rows.append(
            {
                "geom1": model.geom(a).name,
                "geom2": model.geom(b).name,
                "position_world_m": con.pos.tolist(),
                "distance_m": float(con.dist),
                "dimension": int(con.dim),
                "normal_force_N": float(force[0]),
                "local_contact_force": force.tolist(),
                "normal_axis_dot": float(sign * frame[0] @ axis),
                "normal_axial_reaction_N": float(normal @ axis),
                "friction_axial_reaction_N": float(friction @ axis),
            }
        )
    normal = sum(r["normal_axial_reaction_N"] for r in rows)
    friction = sum(r["friction_axial_reaction_N"] for r in rows)
    if abs(normal + friction - float(data.qfrc_constraint[0])) > 1e-7:
        raise RuntimeError("Contact-force decomposition disagrees with slide reaction")
    return {
        "normal_axial_N": normal,
        "friction_axial_N": friction,
        "generalized_contact_reaction_N": float(data.qfrc_constraint[0]),
        "contacts": rows,
    }


def run(args):
    source = ROOT / "outputs/linear-insertion-aligned-20261107" / args.step
    output = ROOT / "outputs/linear-insertion-diagnosis-20261107" / args.step
    output.mkdir(parents=True, exist_ok=True)
    load_backend(ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json")
    set_surface_mode(1)
    receipt = json.loads((source / "receipt.json").read_text())
    alignment = json.loads((source / "alignment.json").read_text())
    saved = np.load(source / "final-state.npz")
    axis = np.array(alignment["axis_world"], dtype=float)
    axis /= np.linalg.norm(axis)
    target = float(alignment["authored_axial_travel_m"])
    piece, receivers = SPECS[args.step][2], SPECS[args.step][4]
    cases = [
        ("baseline_0p5N", True, True, 0.5),
        ("no_friction_0p5N", True, False, 0.5),
        ("no_contacts_0p5N", False, False, 0.5),
        ("baseline_5N", True, True, 5.0),
        ("no_friction_5N", True, False, 5.0),
    ]
    results = []
    for name, contacts, friction, force in cases:
        started = time.monotonic()
        model = mujoco.MjModel.from_xml_path(str(source / "guide-scene.xml"))
        active = configure(model, piece, receivers, contacts)
        if not friction:
            model.geom_friction[:] = 0
            model.geom_condim[:] = 1
        data = mujoco.MjData(model)
        data.qpos[:] = saved["qpos"]
        data.qvel[:] = saved["qvel"]
        mujoco.mj_forward(model, data)
        push = BalancedPush(model, data, active, axis, damping=10)
        push.apply(data, force)
        mujoco.mj_forward(model, data)
        initial_reaction = reactions(model, data, active, axis)
        initial = float(data.qpos[0])
        peak_penetration = max(
            0.0, -min((float(c.dist) for c in data.contact), default=0)
        )
        reason = "duration elapsed"
        for _ in range(round(args.seconds / model.opt.timestep)):
            push.apply(data, force)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            peak_penetration = max(
                peak_penetration,
                max(0.0, -min((float(c.dist) for c in data.contact), default=0)),
            )
            if not np.isfinite(data.qpos).all():
                reason = "nonfinite state"
                break
            if float(data.qpos[0]) >= target:
                reason = "reference depth crossed; not seating proof"
                break
            if peak_penetration > 0.0005:
                reason = "0.5mm penetration guard"
                break
        push.apply(data, force)
        mujoco.mj_forward(model, data)
        result = {
            "case": name,
            "contacts_enabled": contacts,
            "friction_enabled": friction,
            "force_N": force,
            "initial_axial_mm": initial * 1000,
            "final_axial_mm": float(data.qpos[0]) * 1000,
            "additional_axial_mm": float(data.qpos[0] - initial) * 1000,
            "reference_axial_mm": target * 1000,
            "final_velocity_m_per_s": float(data.qvel[0]),
            "simulated_seconds": float(data.time),
            "wall_seconds": time.monotonic() - started,
            "peak_native_penetration_mm": peak_penetration * 1000,
            "stop": reason,
            "initial_reaction": initial_reaction,
            "final_reaction": reactions(model, data, active, axis),
        }
        results.append(result)
        np.savez_compressed(
            output / f"{name}-state.npz", qpos=data.qpos, qvel=data.qvel
        )
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if "reaction" not in k}),
            flush=True,
        )
    report = {
        "step": args.step,
        "design": "Identical saved stalled qpos/qvel, constant load, same COM drag/guide/grid/solver. Change only friction/contact mask/force as labeled.",
        "seconds_per_case_max": args.seconds,
        "aligned_receipt_sha256": hashlib.sha256(
            (source / "receipt.json").read_bytes()
        ).hexdigest(),
        "alignment_sha256": hashlib.sha256(
            (source / "alignment.json").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_video_axial_mm": receipt["final_axial_mm"],
        "cases": results,
        "limits": "5N can increase soft-contact penetration; crossing reference is not clearance or real seating proof. Contact-off is a mechanics control, not physical insertion.",
    }
    (output / "ablations.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=SPECS, required=True)
    parser.add_argument("--seconds", type=float, default=2)
    run(parser.parse_args())
