# Engaged reset noise ablation — 2026-11-07

## Question and method

Would smaller noise fix the partly inserted / near-seated reset failures?
780 static native-triangle geometry checks: 12 stages, eight paired TRAIN seeds
per stochastic mode, eight modes, plus one zero-noise reference per stage.
Current means difficulty=1 (the full configured envelope), not the website
pilot's difficulty=.5 or the average training draw. Quarter/tenth scale the
configured envelope to25%/10%. This is a small matched stress sweep, not an
estimated failure rate for the training distribution.

All comparisons preserve the authored non-pin reference poses. They use the
repaired Ø2.3×14.5mm pin and receiver-relative registration in every applicable
stage. The sampler uses its original visual model to produce transforms; the
copied qpos is audited against the matching repaired-pin visual XML. Separate
model-layout-check.json proves joint name/order/addresses/axes and fixed body
positions/quaternions agree, and a copied upper-near snapshot has exactly equal
body world poses in both models. Do not describe this as an unchanged frozen
bank state: it includes the separately approved repaired-pin overlay.

Active contacts include every inter-piece target pair involving the current
incoming part. Inherited contacts are all other inter-piece target pairs. Old
pin artifacts are thus absent, and prerequisite failures cannot mask whether
noise adds a new active-part failure. Geometry uses512samples/direction after
AABB filtering; depths are finite sampled penetration, not universal clearance.
No helper IK, loaded grasps, settling, force-driven insertion or reward admission
was tested. Friction does not affect these static geometry measurements.

## Viewer-level follow-up

A further108checks (12×8pairedseeds at difficulty=.5 plus12repeated zero-noise
controls) explicitly match the website pilot's amplitude. All repeated baselines
match the original sweep and tracked source hashes are identical:888totalchecks.
Noise scales below are fractions of configured maxima; .25 is HALF the viewer
pilot's .5 amplitude.

| State | Pose noise off: depth mm | Viewer .5: largest depth mm | Reduced .25: largest depth mm |
|---|---:|---:|---:|
|04_cover_entry|0.156|0.547|0.343|
|10_connector_near|0.873|1.028|0.950|
|22_upper_half|1.972|1.992|1.977|
|23_upper_near|1.940|1.984|1.976|
|30_final_half|2.566|2.586|2.582|
|31_final_near|2.508|2.509|2.509|

At viewer amplitude, cover04 has1/8 active>.5mm rejections; .25 removes that
additional rejection (0/8). Upper22/23 support-projection rejections are2/8 and7/8;
.25 reduces them to1/8 and2/8; .10 gives0/8 for both. However all8upper22/23 and
final30/31 samples still exceed .5mm in actual active-part overlap even at .10,
consistent with their material-overlapping zero-noise reference geometry.

## Results

Largest active-part sampled overlap across eight paired seeds (mm):

| State | All noise off | Full envelope | All pose ranges ×.25 | All pose ranges ×.10 |
|---|---:|---:|---:|---:|
|04_cover_entry|0.156|0.927|0.343|0.223|
|05_cover_near|0.196|0.365|0.243|0.215|
|09_connector_entry|0.000|0.000|0.000|0.000|
|10_connector_near|0.873|1.186|0.950|0.902|
|14_pin_half|0.000|0.000|0.000|0.000|
|15_pin_near|0.000|0.025|0.000|0.000|
|21_upper_far_side|0.465|1.025|0.615|0.526|
|22_upper_half|1.972|2.323|1.977|1.974|
|23_upper_near|1.940|2.061|1.976|1.968|
|29_final_far_side|1.501|1.544|1.500|1.501|
|30_final_half|2.566|2.591|2.582|2.578|
|31_final_near|2.508|2.509|2.509|2.509|

- **Noise does introduce extra collisions:** cover entry04 has3/8 active overlaps
  above the unchanged .5mm screen at the full envelope; quarter/tenth each0/8.
  Its reference already has .156mm sampled overlap; passing .5mm is not a
  certificate of positive clearance. Removing only axial noise still gives3/8,
  pointing to transverse/angular rather than depth variation for this case.
- **Upper support coupling is a separate noise problem:**22/23 have5/8 and8/8
  support-projection rejections at the original caps. Reducing only active-part
  noise leaves those counts unchanged. Scaling all pose ranges toquarter gives
  1/8 and2/8; tenth gives0/8 for both. Removing connector swivel alone gives0/8.
  The connector and incoming upper are rotated together while their far support
  is the fixed base_side_cover; nearest-surface correction then translates the
  incoming upper and changes pocket registration. Reducing swivel helps that
  added error, but a proper joint support/pocket solve is preferable.
- **The main partial/near fit failures remain with zero noise:** connector10
  .873mm, upper22/23 1.972/1.940mm, final30/31 2.566/2.508mm. Their eight full,
  quarter and tenth samples all remain above .5mm for the active part. No claim
  that reducing noise makes these references fit.
- Repaired pin14 has no sampled active overlap in this sweep. Pin15's full
  envelope has one25µm sampled overlap (below .5mm); quarter/tenth yield none.
  This is not proof of dynamic insertion or guaranteed clearance.

## Implementation traps

Supported upper/final stages use axial hinge motion and optional support slide.
Their configured angle_deg and drawn lateral offsets are not actually applied
as independent angular/lateral noise. Merely reducing those fields cannot
change the pose. lateral_radius_m also serves as the projection rejection cap;
shrinking it can create extra rejections without worsening geometry. This
experiment records BOTH the sampler's resulting cap and a fixed original cap;
all projection counts above use the original cap for an apples-to-apples test.
active_zero still includes connector swivel; it is not the all-noise-off control.
There is no demonstrated discontinuity when ALL noise tends to zero.

## Recommendation

Use a tighter initial transverse/tilt envelope for engaged states, beginning
with quarter bounds as a candidate, and validate it against actual available
clearance. Keep axial depth diversity separate, bounded by remaining travel and
nonterminality. For22/23, reduce/constrain connector swivel during supported
insertion or solve the far support and receiving pocket together. Fix reference
registration separately; smaller noise cannot erase baseline material overlap.
Supported noise implementation/caps should be made explicit before production
retuning. No production configs, poses, bank entries, website or physics defaults
were changed by this investigation.

Source hashes are in results.json and exact modules/config in source-snapshot/.
The independent read-only sampler review agrees with the causal interpretation.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/engaged-noise-ablation-20261107/investigate.py --seeds 8 --workers 4 --name results
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/engaged-noise-ablation-20261107/investigate.py --seeds 8 --workers 4 --name half-range --modes all_half
```
