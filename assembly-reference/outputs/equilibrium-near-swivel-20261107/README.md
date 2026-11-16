# Upper-stage swivel branch: no adopted recipe

Diagnostic branch for stages 21–23 on the centered V6 physical model. No CAD,
physics, controller, or canonical reset source was edited by these searches.

The tested coherent base-output rotations were 0, ±5°, ±10°, and ±15°: the
connector, installed pin, and incoming upper module were rotated about the
actual servo-1 output axis, with servo-1 angle changed consistently. The left
base-cover hold stayed fixed. Actual material checks are still required because
the cover receiving edge itself does not rotate with that output.

`search_swivel.py` and `search_flips.py` used the stage-20 motor-local grasp.
Those candidates could not match the required near-stage motor face within the
original five-joint bounds. Equivalent hand rotations did not resolve this.

`search_from23.py` instead transferred the actual captured stage-23 motor-local
grasp from `equilibrium-active-mating-grasps-20261107/original-v2`. Stage 21
remained wrist-limited (best span-vector residual about 0.061 at a −12 mm
longitudinal shift). Stage 22's coarse candidate at −15° yaw replayed with zero
fixed-pad load. Re-fitting stage 23 to its nominal authored case pose likewise
lost the fixed contact; this does not invalidate the original loaded stage-23
snapshot, which this search does not replace.

`search_diagonal.py` explored deeper longitudinal positions with relaxed angular
matching. The −20 mm stage-22 candidate produced severe penetration and moving
pad loads in the 10–30 kN range. It is invalid and must not be adopted.

All successful-looking candidate files in this directory remain diagnostic;
their `local_hold_pass` is false. No recipe here qualifies as a new reset.
Contact margins temporarily enlarged for signed-gap queries were restored to
the native values before the replay. The original loaded stage-23 snapshot is
the useful capture source, rather than the failed re-IK candidates here.
