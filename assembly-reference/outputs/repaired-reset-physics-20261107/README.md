# Repaired reset physical integration — 2026-11-07

The reset validator now accepts the repaired candidate explicitly. Existing
profiles and defaults remain separate. This is not a claim that the guided
insertion proof qualifies the full bimanual reset scene.

New keyword arguments on `build_physical_profile` and `WowRoboPhysicalValidator`:

- `completed_anchor_bank`: private completed references derived from the actual
  repaired insertion endpoints; used for installed mechanical-interface poses
  and completion anchors instead of historical photo-v2 references.
- `pin_registration_path`: the registration measured with the centered connector
  and Ø2.3×14.0mm shaft.
- `pin_pose_registered=True`: preserves pin poses/grasp coordinates already
  registered in the new nominal bank; avoids lifting installed pins or replacing
  the canonical32.1mm alignment with the historical32mm overlay.
- `mating_profile='legacy_mapped_all'`: explicit union of the original per-step
  laws. All14 unique pairs numerically match the five successful canonical proof
  XMLs (see `pair-law-comparison.json`). No conflicts, helper welds or force guides
  are introduced. Contact/material laws remain uncalibrated candidates.

Fine target fields are reused only when visual source bytes, source identity,
unit scale and grid file hashes match. An already-matching0.1mm pin field is not
replaced by a coarse rebuilt field. If genuinely changed pin geometry requires
rebuilding, its requested spacing is0.1mm. Cached helper/pin fields with another
source/spacing fail closed. Fine-case field-to-source transforms are applied
exactly once with the existing reframe marker.

Completed-anchor and pin-registration identities, target manifest, contact-law
configuration and compiled model are bound in each new physical profile.
Native profiles must run in fresh processes because grid libraries are immutable.
`check_source_sdf_frames(..., include_targets=True)` supports independent compiled
world-frame verification of both the16targets and38helper mesh geoms.

The bounded reproduction script is `run_probe.py`. It uses real gravity, the
original force-limited native helper actuators, per-data friction in native force
solving, native material guards, bilateral pad loads and relative retention.
Initial geometry/sampler errors remain recorded and do not bypass those guards.
A rejected candidate may be run for diagnosis; rejection flags are preserved.
No artificial state freezing or success labels are introduced.

11 physical mechanism tests pass; Ruff passes. They establish the integration
contracts, not admission of25poses or a production bank. Actual repaired-profile
results are below; the new nominal bank is sealed and preserved.


## Actual repaired full-scene probes

Profile: `assets_local/wowrobo/physical-resets-repaired-v1/profile.json`.
All16 frozen target field bindings matched, including0.1mm case/pin fields.
All14 explicit pair laws retain their canonical numeric values. Genuine installed
interface equalities use the new private completed anchors; there is no force
insertion guide or helper attachment. Grid runtime uses one fresh process.

Compiled SDF verification:54geoms/2592worldqueries, all pass; maximum raw-grid
frame discrepancy2.717e−16m, independent triangle unsigned error≤0.2454mm
(within coarse-grid bound); fine case errors≤50µm. An initial extension of the
checker forgot inverse field-to-source maps for reused cases2/3/4; its false
raw-query flags are preserved in `sdf-frame-check-before-query-frame-fix.json`.
Actual compiled distances already matched source triangles. Corrected checker
output is `sdf-frame-check.json`; runtime/model/fields were not changed by that
checker fix.

Four initial zero-noise probes used10mssettle+20mshold:

| State | Result |
|---|---|
|00loose|Native geometry clear; still moving at30ms.|
|14pinhalf|Native geometry clear; not yet securely acquired/settled at30ms.|
|22upperhalf|Initial cover↔upper printedlink overlap0.5965mm; material guard stops after0.25ms, before hold.|
|30finalhalf|Initial upper printedlink overlaps base2.6626mm,cover2.2255mm,connector1 2.4179mm; servo2case↔forearmprinted1 0.8310mm. Guard stops after0.25ms, before hold.|

`initial-native-audits.json` independently checks pre-dynamics inputs and records
actual pairs. No helper/target initial collision is responsible for these22/30
failures. Unloaded pad values from their incomplete hold window do not establish
a gripping failure.

Longer probes use the actual default150mssettle+150mshold window:

-00loose: geometry clear, maximum native material39.0µm; max velocity0.7556rad/s,
 mainly passive loose forearm motion. Rejected only `not_settled`; this0.3s probe
 does not prove perpetual motion or long-term instability.
-14pinhalf: geometry clear(max native0.2624mm), LEFT pin pads loaded5.49/6.61N
 during hold; RIGHT connector-support pads0/0N. Relative grasp displacement
 reaches7.095mm; max velocity0.7966rad/s mostly another loose forearm. Rejected
 `grasp_slip`, `unloaded_finger_pad`, `not_settled`. Pin remains nonterminal:
 feature position residual0.629mm exceeds0.5mm completion bound.

Results and archived actual input/settled states are in `bounded-probes/` and
`default-window-probes/`; all rejections remain false admission. Short-window
acquisition/settling flags are not substituted for the longer-window evidence.

## Remaining pose inconsistency

The upper force-driven canonical endpoint uses shaft2=-0.081688567rad. The new
photo-support nominal bank instead uses -0.395081336rad for completed24/final30
(17.96degrees different) and -0.479341789rad in22. The servo case stays at the
successful receiving pose, but the printed link rotates to touch a selected
photo far-support patch. It intersects the base/cover/connector. These are new
photo-support variants of the canonical case pose, not exact reuse of the whole
successful upper endpoint and physical hinge. Distinguish that remaining
registration problem from failure of the repaired case/socket CAD. No nominal
pose was changed by this physical probe.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/repaired-reset-physics-20261107/run_probe.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/repaired-reset-physics-20261107/run_probe.py --stages 00_loose 14_pin_half --settle .15 --hold .15 --output outputs/repaired-reset-physics-20261107/default-window-probes
```

All physical contact/material laws and the pin's provisional100gmass remain
uncalibrated assumptions. Nothing here claims a physically admitted training
bank, learned robot insertion, or hardware qualification.
