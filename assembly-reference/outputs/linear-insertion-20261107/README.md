# Current-pose linear-force insertion review — November 7, 2026

Five step owners produced independent probes from a frozen copy of the latest photo-reset v2 states. The input snapshot and SHA256 manifest are in `source-snapshot/`. Existing pose generation and training were not changed.

Each step has two force-driven native MuJoCo runs:

- **OFF:** the source scene's disabled-contact behavior. The directly forced piece can pass through solids; reaching the authored depth does not prove seating.
- **CONVEX:** native convex mesh contact diagnostic. Convexification fills concave cavities, so blockage can be a contact-model artifact. This is not qualified CAD insertion physics.

A force ramps to 0.5 N along the insertion axis. Integration uses 1 ms steps; translational damping is 10 N·s/m. The incoming component's free root and internal servo hinge remain dynamic. The receiver assembly and helper-arm joints are fixed by restoring their saved coordinates each integration step, and gravity is off. Force is applied directly to the component, not transmitted through a physically maintained robot grasp. No active-component pose interpolation or prescribed insertion trajectory is used. Safety/depth stops remove the displayed force and hold the last simulated frame for review.

Upper-arm force is world down at the servo 2 case. Final-module force is receiver-local down at servo 3, about 22.6° from world down. Their root translations are not insertion axes because the modules can rock.

Each mode directory contains `insertion.mp4`, three sampled frames, final state and `receipt.json` with frame-level force, held-point displacement and contact data. These probes establish neither grasp retention, qualified reset admission, learned policy success nor hardware behavior.

Reproduce from the canonical repository root:

```sh
../assembly-so101/.venv/bin/python scripts/probe_wowrobo_linear_force.py --step cover --contact-mode off --output outputs/linear-insertion-20261107/cover/off
../assembly-so101/.venv/bin/python scripts/probe_wowrobo_linear_force.py --step cover --contact-mode convex --output outputs/linear-insertion-20261107/cover/convex
```

Replace `cover` with `connector`, `pin`, `upper` or `final`. Native CGL rendering on this Mac needs a graphics-capable execution context.

## Recorded axial travel

| Step | Authored depth (mm) | Contacts off (mm) | Convex contacts (mm) |
|---|---:|---:|---:|
| cover | 55.00 | 55.06 | 21.90 |
| connector | 65.00 | 65.10 | 50.19 |
| pin | 32.00 | 32.03 | 8.41 |
| upper | 25.00 | 23.59 | 10.16 |
| final | 25.00 | 21.03 | 16.35 |

Scalar axial depth does not establish orientation, contact fit or seating. Upper/final safety-stop in both modes; other convex runs stop short of reference.

## MuJoCo runtime verification

Verified bindings/core/studio all4.0.0 and native engine4.0.0. The native dylib predates all10delivered clips. Details/SHA256: `mujoco-runtime-verification.json`. Initial run's14.25× result also includes task-specific surface-tree SDF plugins and sparse solver settings; WowRobo fixture probes have zero plugins and do not reproduce that qualified contact configuration.
