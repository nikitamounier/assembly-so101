# Repaired CAD reset review — 2026-11-07

The referenced CAD-fix chat was read, then its actual frozen files and successful
guided insertion evidence verified. The updated website at http://localhost:8876/
now serves a new100-entry collection:25references plus75noisyvariations. Old
components-v2, photo-v3 and pilot-v2 remain preserved. No running training job,
cluster resource or monitor was touched.

## Adopted changes

- Exact16target meshes from final-common-v1, including four changed constituents:
  connector printed0/1, upper printed0 (also the final receiving pocket), and pin.
  Every source/grid hash is verified; all38helper/camera meshes and all fixed
  body/joint frames/order/addresses remain exactly unchanged.
- Pin nowØ2.3×14.0mm shaft, Ø12×28mm grip,42.0mm total, matched to the centered
  connector's head stop (~.375mm axial clearance). The earlier14.5mm version
  remains historical and would bottom in this corrected registration.
- Corrected cover, connector, pin and installed upper receiver-relative mating
  frames. All25original IDs/roles/order preserved. Active case poses follow the
  actual successful canonical insertion paths at equivalent stage depths.
- The guide proof locks internal shape. Its whole printed-link pose cannot be
  transplanted blindly into the photos: stale photo witnesses would float up to
  58mm. Real internal hinge articulation restores the selected far-end support
  while keeping each registered case pose exact. All six far-end patch gaps are
  below9.17µm; re-solved helper grip position residual≤31.28µm.
- The repaired-only randomizer similarly keeps case/pocket orientation fixed and
  refits support with the actual hinge after noise, replacing whole-case
  nearest-surface translation. Original amplitudes/friction variation retained;
  legacy distributions remain unchanged. Geometry and actual gap checks still
  reject colliding fits: point support alone is not collision or load proof.
- New physical profile uses the frozen fine target fields, source helper fields,
  repaired completed anchors, matching registered pin, and explicitly selected
  union of all14canonical mating laws. Cases/pin retain.1mm fields and required
  coordinate transforms are applied exactly once. No guide or helper welds.

## Geometry result

16/25reference states pass the unchanged .5mm native boundary screen plus
registration checks. Passing a finite overlap screen is not positive clearance
or physical training admission.

| Reference state | Largest sampled overlap mm | Remaining conflict |
|---|---:|---|
|21_upper_far_side|0.537|base_side_cover_printed_0 ↔ upper_arm_module_printed_0|
|22_upper_half|0.597|base_side_cover_printed_0 ↔ upper_arm_module_printed_0|
|23_upper_near|1.143|base_servo_printed_0 ↔ upper_arm_module_printed_0|
|25_right_support|2.663|base_servo_printed_0 ↔ upper_arm_module_printed_0|
|26_final_floor_grasp|2.663|base_servo_printed_0 ↔ upper_arm_module_printed_0|
|28_final_transport|2.663|base_servo_printed_0 ↔ upper_arm_module_printed_0|
|29_final_far_side|5.150|upper_arm_module_printed_0 ↔ forearm_module_printed_1|
|30_final_half|2.663|base_servo_printed_0 ↔ upper_arm_module_printed_0|
|31_final_near|4.817|upper_arm_module_printed_0 ↔ forearm_module_printed_1|

The important distinction: repaired connector/pin and case/socket paths now
clear the original failure class, but keeping the selected photographic support
articulation can rotate the long printed link into the base/cover/other link.
For installed upper, the photo-support hinge is−.395081rad, compared with the
actual guided endpoint−.081689rad (~18° different), despite matching its case
pose. This is a **photo-support pose variant**, not the exact whole-body guided
endpoint. It does not invalidate the CAD-fix guided insertion proof. It also
cannot be called a physically valid reset merely because a far point touches.
Changing witness/hinge branch must be checked against all native material, not
only pocket alignment. Noise support refitting may select different nearby
surface-contact branches; material checks reject those unsafe choices.

At the same viewer difficulty.5 and original amplitudes, the75variation pilot
improves from51rejected/24pending to29rejected/46pending.27new rejections involve
part-to-part collisions in the9already-invalid late reference families; two
additional pin-stage draws fail helper/target or helper/table checks. Rejection
categories overlap. All statuses remain separate from physical admission.
The viewer shows actual reference failures and the repaired CAD revision rather
than retaining old 'approved/pass' labels.

## Noise retest

396matched static component checks cover12entry/support/partial/near states:
8pairedseeds at difficulty1,.5,.25,.1 plus12zero-noise baselines. These exclude
helperIK/dynamics and separate active from inherited inter-piece overlaps.
At viewer difficulty.5, cover entry04, connector entry09/near10 and pin14/15 have
0/8active overlaps above.5mm; the late upper/final poses retain material failures
with pose noise off and with reduced noise. Full-range cover04 still rejects3/8;
noise should eventually be clearance-bounded. See repaired-ablation.json and
ablation-summary.json. This is a small geometry sweep, not an estimated training
failure rate or force/retention qualification.

## Actual dynamics still fail

Fresh full-scene probes are in ../repaired-reset-physics-20261107/README.md.
Four short probes and two default-window(.15ssettle+.15shold) probes yield zero
admissions:

- **00:** native geometry remains clear (~39µm maximum material depth), but passive
 joints are still moving (~.756rad/s) after.3s.
- **14:** native geometry clear (~.262mm max); LEFT pin pads load5.49/6.61N, but RIGHT
 connector-support pads have0/0N and relative grasp displacement reaches7.10mm.
 It fails support/retention and settling. This supersedes the short-window
 'left unloaded' observation; we do not mislabel the correct left pin pinch.
- **22/30:** actual material guards stop at the first.25ms due to existing long-link
 overlaps, not a newly diagnosed gripper failure.

Independent all54SDFfield frame audit:2592queries, maximum frame agreement error
2.72e−16m; sampled source-triangle distance error≤.2454mm across coarse helpers,
fine cases≤50µm. A diagnostic initially queried transformed fields in the wrong
coordinate frame; corrected independent raw query passes. Runtime fields did
not need changing. All14explicit laws match canonical proof XMLs exactly.

The CAD-fix upper-release result(1.449°tilt versus1°benchmark) also remains an
explicit limitation. Guided completion, static point contact and unit tests do
not establish learned robot assembly or hardware qualification.

## Validation and reproduction

37native regression tests and12frontend tests pass; build and Ruff pass. New
config/asset matching is checked, source and state identities guarded, and old
banks are not silently reinterpreted with new meshes. Native and Three.js viewer
modes were verified; stage10now shows a clear geometry screen, stage23still
shows its real failure. Screenshot viewer.jpg records the latter.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/cad-repair-reset-review-20261107/audit_nominals.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/cad-repair-reset-review-20261107/run_ablation.py --seeds 8 --workers 4 --name repaired-ablation
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_randomized_bank.py --config configs/wowrobo_randomization_repaired.json --validation geometry --difficulty .5 --max-attempts 1 --train-per-stage 2 --eval-per-stage 1 --output assets_local/wowrobo/randomization-bank-repaired-v1
```

The builder is resumable for identical sealed inputs. Changing source/config or
reference metadata requires a new output folder. Old19,200training target remains
unfilled:0physicallyadmitted; the100reviewentries are not training resets.
