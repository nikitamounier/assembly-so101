# Motor-held, passive-bracket diagnostic — 2026-11-07

This is an actual MuJoCo simulation with gravity, a force/torque hold applied only to the servo-3 motor, a free incoming-module root and a passive shaft-3 hinge. No hand is present. Receiver and preceding assembly pieces are explicitly fixed. It is an isolated motion diagnostic, not bimanual grasp or training qualification.

Source geometry/contact fields are the frozen repaired `final-common-v1` snapshot. No source meshes, hinge axes or body inertias were changed. Existing receiver contact laws were retained. Two explicit own-bracket/motor SDF pairs supply the mechanical stop that default parent filtering previously suppressed. All other active-piece/receiver and table contacts use the existing native source-geometry backend; incoming motor/receiver fork retains the existing final mating law. Hinge range remains the original ±1.7rad, not a new angular clamp. Initial hinge is zero; no later hinge coordinate is assigned or targeted.

The motor controller applies Cartesian force and orientation torque. It lifts vertically, translates horizontally above the receiving pocket and lowers vertically. Position gain1000N/m, damping25N·s/m, orientation gain5N·m/rad, angular damping0.08N·m·s/rad, force component clamp±25N, torque component clamp±1N·m. Active-only gravity compensation uses0.14kg. The final25mm virtual-target preload is force-driven; it does not teleport or prescribe the motor state. Source identity and exact generated model are stored per run.

- `v1`: initial prototype; excessive gravity feedforward included fixed receiver mass. Preserved as failed history; not the displayed best run.
- `v2`: corrected gravity, weaker1N·m/rad motor attitude hold. Native hinge folds~10° but insertion stalls15.4mm short.
- `v3`: stronger attitude hold plus bounded downward preload. Actual passive hinge~10.10° in transport,9.59° at first receiving contact,1.37° at10s,0.606° at16s. Final motor center0.167mm from canonical seated center. Max own-stop penetration89.8µm; max receiver penetration158.8µm. Initial1s dwell let the bracket rotate into the table transiently (worst overall1.33mm), so this is superseded by immediate-lift v4.
- `no-own-control`: matched v3 controller/path with stop pairs omitted. Hinge falls to~90° and insertion misses by128.8mm, demonstrating that the observed10° stop depends on actual geometry contacts.
- `v4`: removes initial dwell. Motor rises25mm over0.3s, then continues vertical lift, horizontal transport and vertical lowering. Initial table contact penetration is materially reduced. See `v4/result.json` and actual replay.

- `v5` is the displayed corrected run: same immediate pickup as v4, with isotropic own-stop pair friction correctly expanded from source geometry friction to the two sliding directions (`.7 .7 .005 .0001 .0001`). Earlier v1–v4 prototypes accidentally used anisotropic own-stop friction and remain historical. Corrected friction gives a brief sticking phase before the bracket opens under bounded lowering force. See `v5/result.json`, `v5/phase-summary.json`, `v5/simulation.mp4`.

The native own-contact angular screen is clear at0°,5°,8°; starts at10° with47µm overlap. This distinguishes the mechanical stop from accidental overlap already present at the initial angle. Actual motion is not merely a replay of the earlier authored reset angles.

Reproduce:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/probe_wowrobo_passive_final.py --out outputs/passive-final-20261107/new-run --seconds 16 --preload .025
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/render_wowrobo_passive_final.py outputs/passive-final-20261107/new-run
```

macOS rendering needs access to CoreGraphics. Rendering assigns saved qpos only to reproduce the previously integrated trajectory; it never generates the physics. Orange marker identifies the force-held motor point. Replay labels disclose fixed receiver and gravity.

Validation: three native fixture regressions pass (8qpos/7qvel, no actuators/equalities/hands; gravity active; only free root+passive hinge; actual stop contacts appear near10° with original broad hinge limits, zero hinge stiffness and isotropic source friction). Ruff passes for the new module/scripts/test. The main clip ends with the motor force-held. The independent release check below tests bounded retention; no real gripper or hardware calibration is claimed.

## Corrected v5 result and independent release

Final motor position error 0.168mm; maximum own-stop overlap 75.8µm; maximum receiver and overall overlap 167.8µm. The bracket has a sticking phase at about9.86° during descent, then opens to 0.605° when the bounded motor force overcomes it. No angular interpolation/target acts on the hinge. Maximum motor force resultant 25.06N (individual Cartesian components capped25N).

An independent1s release from the actual saved endpoint, with all applied forces/torques zero, gives motor translation drift 0.131mm, motor rotation drift 0.0886°, hinge drift -0.00105°, final maximum coordinate speed 0.001096. This supports only bounded fixed-receiver retention, not a hardware- or bimanual-qualified attachment. See `v5/release.json`/`release.npz`.

## Viewer verification

Final-module video icon opens the Physics simulation tab by default; Real assembly remains available. Served derivative is an exact copy of v5/simulation.mp4. Browser decoded960px/16.033s/readyState4, playback advanced with paused=false, both source tabs switched correctly. Screenshot `viewer-simulation.png` shows the actual final simulated frame. Frontend13tests/build pass.
