# Physical reset admission review — 2026-11-07

No physically qualified reset or full training bank is claimed. The25 approved review poses remain unchanged.

## Implementation and verification

`assembly_policy/so101/wowrobo_physical_resets.py` provides native source-SDF contacts for targets/helpers, gravity, native force limits, episode friction in actual force solving, bounded jaw closing, bilateral finger-pad loads and relative retention, native boundaries before/after settling, target-table material depth, nonterminal feature checks and post-settle tangency/nearest-end checks.

New pin: `assets_local/wowrobo/fit-pin-v2/pin.obj`, shaftØ2.3×14.5mm, gripØ12×28mm. Receiver registration is applied once inside validator;13–15 recompute LEFT IK to follow actual head.

Helper-frame audit:38 geoms/7296 queries, compiled-world/raw-grid maximum error2.948e−16m, independent source-triangle unsigned-distance maximum error0.14969mm within sampled grid-spacing tolerance. This is a sampled audit, not a universal interpolation bound. Receipt: `assets_local/wowrobo/physical-resets-v4/helper-frame-validation.json`.

implicitfast friction split step matches native stepping to1e−12 with unity draws in the contact regression. Changed friction changes actual velocities. Native damping/inertias/free hinges/±3.35Nm force limits are preserved.

## Installed mechanical interfaces and replay

v5+ includes three initially inactive equalities: cover→base(stage7+), connector→servo1 output horns(12+), servo2 case→connector(25+). No helper welds/fixtures. Servo1 output hinge remains free; connector swivel is initialized coherently. Required equalities activate only after initial native fit passes; overlap blocks activation rather than being hidden. Completed-reference anchors remain assumptions pending genuine mating-geometry certification.

Raw input and settled states use `mjSTATE_INTEGRATION`, preserving eq_active and controls; friction draws remain separate metadata. Test verifies equality activation get/set replay, retained hinge and absence of helpers in these constraints. Bank builder retains raw state+spec and copies pose/control into contact-off display. This is NOT a qualified loaded-bank replay test: no pilot currently qualifies.

## Actual bounded results

- v3, dt0.5ms/table2ms: stage08 aborted58.5ms with actual loose upper-part table depth1.1246mm. Hold window was not reached; this does not establish a gripping failure.
- v4, dt0.25ms/table0.5ms/Euler: stage00 at0.65s actual material31.20µm; native bounds clear; rejected not_settled.
- v6, same contact settings/implicitfast/installed interfaces: stage00 after1.55s rejected only not_settled; max velocity0.91847rad/s(free forearm), actual material28.27µm. Tangency/nearest-end all pass. Accessibility IK skipped because candidate already rejected.
- v7 isolated probe, dt0.1ms/table1ms/implicitfast: stage00 at0.55s rejected only not_settled; max velocity0.44747rad/s(free upper module), actual material34.32µm. Tangency/nearest-end all pass. Native passive target hinge damping0.02/frictionloss0 unchanged. No velocities zeroed, no hinges frozen.
- Independent CAD investigation identifies actual connector/horn lip interference: `outputs/native-registration-review-20261107/README.md`. Corrected pin addresses its own blind-bore obstruction, not that separate fit.

Exact model/profile/report paths and metrics: `summary.json`. v7 is an isolated numerical probe, not a calibrated production profile. Builder defaults retain v6 contact settings with implicitfast. Further settling/contact investigation and native CAD fitting remain necessary before a training bank.

## Tests

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python -m pytest -q tests/test_wowrobo_physical_resets.py
../assembly-so101/.venv/bin/ruff check assembly_policy/so101/wowrobo_physical_resets.py tests/test_wowrobo_physical_resets.py
```

6 passed in4.70s; Ruff passed. Tests cover palm exclusion, actual nonterminal features, gravity/contact/force bounds, case-grasp roles, implicit friction force solving and installed-interface hinge/state preservation. Profile-dependent tests skip if generated v6 assets are absent. Mechanism tests do not certify25 states.
