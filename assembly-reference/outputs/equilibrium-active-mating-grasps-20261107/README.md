# Active upper-module grasp investigation — 2026-11-07

The final source model is `../equilibrium-model-20261107/profile-centered-v6/profile.json`; coherent initial poses are `../equilibrium-engaged-repair-20261107/coherent-recipes.json`. All actual hold states in `original-v2/*.npz` use registered v6 coordinates. `hold_state` stores exact MuJoCo integration state after 0.1 s preparation, including its actual controls; the NPZ `ctrl` array instead stores pre-preparation controls.

## Key diagnosis

The legacy `_loaded_pads` predicate cuts fixed-finger local X at +4 mm. Actual source finger mesh extends approximately +6.3 mm. Root's raw contact inspection found legitimate fixed-finger contacts around +5.1–5.49 mm for 21/22 and +4.4–5.2 mm for 23, which the old predicate ignored. Root is validating original coherent poses with source-mesh extent counts. Candidate pose searching is paused pending that result.

## Candidate status

- `original-v1`: 50 µm native precontact fit, original hand branch. Do not adopt without new source-extent qualification.
- `original-v2`: 200 µm precontact fit, 0.0005 rad real jaw control preload. Stage 23 had 0.84375 bilateral fraction even using legacy tip cutoff and relative motion 104.7 µm / 0.0626°. Root extracted its hold state for validation. Stages 21/22 were judged false-negative by old pad counts; coherent originals should be tried first because these candidates were fitted against that old cutoff.
- `static_candidates`, `flipped`, `swivel-v1`, `centre-v1`, `from23-v1`, `same23-v1`, `yaw23-v1`: rejected experiments, not production recipes. Some have large intended-finger/case penetration or one-sided loads and must not be adopted just because IK residual or case motion is small.

No production model, source CAD, bank, or viewer was edited by this child agent. Parent owns final source-extent contact predicate and complete-scene validity results.
