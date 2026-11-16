# Installed cover support correction — 2026-11-07

The old `hold_base_assembly` anchor targeted the lower `base_servo` housing. LEFT now grips the opposed side walls of the installed `base_side_cover` through public states 17, 18, 20, 21, 22, 23, 25 and private completed anchor 24.

Published review bank: `assets_local/wowrobo/randomization-bank-repaired-cover-support-v2`; reference bank: `assets_local/wowrobo/photo-resets-repaired-cover-support-v2`. Only LEFT six joint coordinates/controls changed. Target pieces and RIGHT coordinates are bitwise preserved. Support metadata now explicitly names the cover, and active-arm grasp noise no longer jitters this supporting grip. Existing random noise distributions and contact laws remain unchanged.

Actual native finger surfaces were checked in all eight affected poses: pad gaps 21.48 / 2.20 micrometers, maximum sampled LEFT-target precontact overlap 20.87 micrometers, no sampled hand-hand overlap, table clearance at least 18.6 mm. Cover contact centroid about 42.9 mm above table. Original grasp-template fit residual 0.475 mm / span vector 0.0636 remains recorded; actual native pad geometry is the supporting geometric evidence, not a claim of perfect template fit.

The 25-reference audit retains the same nine target-target failure states: 21,22,23,25,26,28,29,30,31. New review bank has 25 references +75 variations: 29 geometry rejects /46 pending /0 physically admitted. Do not use as a qualified training bank.

A 0.15 s settle +0.05 s hold probe of state18 still rejects `grasp_slip`, `not_settled`, `unloaded_finger_pad`. Minimum fixed-pad load 0 N, moving-pad load 31.91 N; relative grasp drift 2.884 mm. Correcting the intended target/pose does not establish loaded retention. Evidence: `physical-precontact/18_upper_floor_grasp.json`.

Validation: 3 cover-support regression tests pass, including native IK variation behavior and exact RIGHT/target parity. Prior this-turn physical/bank regression set 16 passes and frontend 12 passes/build. Ruff passes. Viewer serves the v2 bank on localhost8876, preserving the current point-cloud UI source.
