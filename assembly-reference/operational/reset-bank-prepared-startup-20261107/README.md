# Prepared reset sampler startup

Staged patches only; canonical guarded source files were not edited.

`PhotoResetSequence(..., prepared_resting=...)` keeps its native jaw calibration,
source-mesh side-wall measurements, helper/layout model, grip markers, initial
layout, order and previous-joint seeds. Only the legacy offline eight-pose
`piece_mesh`/`supported_pose` construction, geometric refinement object and
future-obstacle lookahead are skipped in prepared sampling mode. Default legacy
authoring initialization remains intact.

The staged randomizer enables this mode only when a grasp library, a nominal
model and repaired CAD are selected, and both transport receipts contain
`held_body_sag.far_point_piece_local_m`. These exact markers are the only resting
fields used by current randomization (`sag_held_link`); all other current sequence
calls were audited: `grip`, `pose`, `solve_arm`, `layout`, `side_grasps`, installed
grip calibration, previous joints, order, origin and rotation. They do not use
`refiner`, `lookahead` or authoring final poses. Prepared sequences are for sampling,
not for calling the authoring `build` or `final_pose` methods.

`patch-proof.json` proves the original initial grasp/layout block is unchanged,
all methods following the constructor are unchanged, and all randomizer restore,
sample, noise and IK method text is unchanged.

Validation (`validation.json` and `validation-hard.json`):

- All25 zero-noise states physically match their references within6.7e-16 after
  recognizing equivalent ±quaternions and the existing hardware-control clipping.
- All25 tiny-noise states remain finite with no geometric sampler rejection.
- Exact qpos/qvel/controls equality against four actual saved native-input states:
 21/29 EVAL seed0 and21/29 TRAIN seed59 (hardest training stratum).
- Both transport far markers are exactly equal to saved approved receipt values.
- Prepared constructor4.73–5.23seconds, validation8.4–8.6seconds; measured peak
 MacRSS2.87–4.59GB including physical-model loading and IK replay comparisons.

No native contact, overlap, retention, dynamics, noise or admission computation
is skipped or cached by this patch. Linux memory still must be measured before
choosing25-worker concurrency.
