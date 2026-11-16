# Repaired CAD nominal reset references

This separate bank applies the frozen `insertion-fix-20261107/final-common-v1`
derived meshes and coherent receiver registrations to the25approved original
state IDs. It does not mutate photo-resets-v3, source vendor CAD, or the frozen
guided-insertion proof. All entries remain unqualified kinematic references.

Installed predecessors are composed receiver-relatively from successful canonical
endpoints. Pin uses the coherentØ2.3×14.0shaft. Upper-case seating matches the
actual successful FULL51 endpoint, while its printed link remains free to
articulate on the genuine shaft. The final path follows that actual articulated
upper receiver. All five private completion anchors are in `completed/`; these
are feature/interface references, not admitted terminal training starts.

Engaged incoming servo-case poses follow the demonstrated canonical axial case
path. The fixed-shape guide alone would float photographed far ends by4–58mm.
Instead, the real internal shaft is optimized within±0.12rad of the original
photo hinge and the free root is compensated to preserve the measured case pose.
This retains both pocket registration and the photographed far support without
adding a fixture or altering receiver/helper poses. Surface witnesses are
refreshed from a15mm far-end printed source patch against upward-facing receiver
surfaces. These are static distances, not loaded contact certificates.

Helper grips are re-solved using their original actual hand/object relation:
servo-case pinch, exterior side walls, upper pin grip and existing support
roles. Open/nonactive helpers retain their original joint pose. Controls match
the new helper coordinates.

## Registration results

-25/25 registration checks pass.
- Maximum helper grip-center residual31.284µm.
- Maximum span-vector residual0.01423; maximum full site orientation residual0.824°.
- Actual fitted case position errors are below1e−10m.
- All six required far-end patch gaps are below9.169µm.
- No dynamic grasp retention, settling, reward or full assembly qualification
  is implied by these kinematic checks. Native material checks are separate.

| State | Far-end source patch gap (µm) |
|---|---:|
|21_upper_far_side|0.003948|
|22_upper_half|0.540329|
|23_upper_near|1.281079|
|29_final_far_side|1.860581|
|30_final_half|9.168646|
|31_final_near|3.675860|

## APIs and reproduction

`RepairedResetSequence` builds references; `fit_case_support_hinge(model,data,
task,evidence,preferred_hinge)` reuses the real-hinge fit without rebuilding
assets. `evidence` requires `receiver_body` and `far_point_piece_local_m`.
The public helper changes only the active free root and internal shaft; it
preserves the current held case pose. Its result remains a candidate requiring
source-material and helper/contact audits.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_repaired_resets.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python -m pytest -q tests/test_wowrobo_repaired_resets.py
```

Three regression tests pass, including actual nonactive-coordinate preservation
and matching the installed upper-case feature frame to the successful endpoint.
Frozen model/mesh identities are in manifest.json and the component manifest.


## Installed-cover support correction

Left supports the installed base side cover throughout states17,18,20,21,22,23,25 and private24. Only the six LEFT coordinates/controls change; every target and RIGHT coordinate is bitwise preserved. See cover-support-change.json and the referenced native finger contact audit. Loaded retention remains separately qualified.

Support correction: actual opposing native finger pads were checked directly. Original template-fit residual is 0.475 mm / span-vector 0.0636; this is not a perfect template IK fit. Precontact gaps are 21.48 / 2.20 micrometers, maximum sampled overlap 20.87 micrometers. Loaded retention remains unqualified.
