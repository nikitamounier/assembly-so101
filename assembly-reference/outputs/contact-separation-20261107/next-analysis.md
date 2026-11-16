# Next targets after general separation and associative caching

Analysis only. Latest benchmark117.473→92.607sPPO and1.378xpacked verifies previous two changes. There is no new detailed pair-time profile after those changes; all pair percentages below are from the earlier hinge profile, not current remaining shares.

## Strongest newly verified structural opportunity

`ASSEMBLY_ATTACHMENTS` in wowrobo_physical_resets.py defines three completed-interface welds: cover/base active fromstage7; connector/outputhorns from12; servo2case/connector from25. Read original benchmark outputs.npz via its NPY schema: eq_active at98:101 is[1,1,0] in upper_far_side21 and[1,1,1] in final_near31, both nominal andvariation worlds. XML equality order agrees. Thus the old speculation that a completed interface might be fixed is now verified for these saved states.

Grouping previous pair timings by actual attached body pairs gives upper13.25% andforearm37.16% (cover4.30%,connector/horns6.36%,connector/servo2case26.50%). Forearm ideal pair-layer ceiling from deleting these costs is1/(1-.3716)=1.59x; not a measured or whole-training forecast.

Candidate: when a specific installed weld is active in that world, omit only its redundant mating pairs. Retain insertion contacts while inactive, exterior contacts always, and articulated servo hinges. Use precomputed pair→equality mapping and mjData.eq_active, not stage labels or shared model mutation. Explicit contact pairs must be handled on their actual dispatch path. Soft native welds allow compliance: removing parallel contact/friction changes physics, so this is NOT an exact optimization. Validate holding/loaded deflection/force distribution/hinge freedom/external impacts and consider attachment compliance calibration. Do not blanket disable all collisions among assembled parts.

## Direct kernel opportunity

sample_surface calls sdf_distance and then sdf_gradient at the same local point for every distance-qualified sample. Both callbacks call grid_eval, repeat voxel lookup/eight half-value loads/conversion and interpolation. Proposed native conditional distance+gradient evaluator computes distance once, rejects if needed, and derives gradient from already loaded corners for passing samples. Preserve arithmetic/clamping/outside-grid semantics; retain generic plugin fallback. Hoist invariant target plugin/grid resolution out of per-sample loop. No speedup measured; likely smaller and easier than changing contact search. The existing get_grid pointer cache already avoids repeated global locking.

## Largest algorithmic opportunity for real contacts

Supplement output is at most12bins (6normal directions×2source directions), but every surviving cached candidate is reevaluated to choose each deepest sample. Current tree discards samples unable to contact; it does not generally discard touching samples unable to win a bin. Seed exact current evaluations of previous winning samples, then use conservative distance and gradient/normal-direction bounds to reject whole patches unable to beat them. Preserve tie-breaking/original sample identity. Unknown bin/near-equal/large-motion cases retain exhaustive fallback. This is branch-and-bound for the existing output, not arbitrary lower point budgets. World-normal bins mean common rotation can change classification even at fixed relative pose.

## Lesser priorities and measurement

- Existing exact-pose memo covers only supplements and uses absolute world poses; four-way swept cache fixes rebuild churn, not repeated positive candidate evaluation.
- Cache rebuild share now2474/(76246+2474)=3.14% of invocations, vs10.02% before. Cost per miss can be high, so count share alone is not time share. Focus on work done on hits before simply expanding cache again.
- Need a short post-change native pass: ordinary/supplement, active-weld pair grouping, candidate samples, distance/gradient calls, cache time, proof success/cost. Existing graph is pre-latest optimization; avoid treating97–98%/37.2% as current measured shares.
- Measure production-weighted task costs and barrier idle time separately from equal-stage/replicated-two-reset panels. Core placement is secondary until measured; current localPPO used2SMTsiblings/onephysicalcore.
- Suggested sequence: conditional fused evaluation as lowest-risk exact win; installed-interface collision ablation as highest-value structural trial; then winner-based branch-and-bound for still-active grasp/insertion contacts. No blanket search-budget cuts, solver focus or productionNIST ratio claim.
