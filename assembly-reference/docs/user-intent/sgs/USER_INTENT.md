# SGS states and the nearby arm

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

**Latest minimum initial-bank requirement (November 7):** All25 starts must have no unintended overlaps and retain held parts in the real grippers. Intended contact/rest/partial insertion is allowed; motion is allowed. This supersedes the earlier strict equilibrium/zero-motion admission requirement. Keep brackets passive, use short native retention and material checks, parallelize, and publish all25 valid starts.

**Implemented:** All25 approved starts passed native minimum checks. Physical bank: `assets_local/wowrobo/initial-valid-resets-v1-physical`; display: `assets_local/wowrobo/initial-valid-resets-v1`. Saved velocities are preserved, allfour target shafts remain unactuated, and display/native body positions match exactly. One prototype per start is complete; variation quotas remain separate.

**Latest WowRobo scope (November 7):** The [photo reset contract](../../WOWROBO_PHOTO_RESETS.md) supersedes the historical unused-arm-open rule below for this new hardware sequence. The other arm holds the connector during pin insertion, LEFT supports base/connector during upper-arm installation, then RIGHT holds the installed upper arm during final-piece installation. Explicit support transitions precede release; both arms remain policy-controlled. Old pilot qualification does not qualify these new states.

Follow [plan.md](../../../plan.md) and the decisions from “sgs states1.” Build physical starting snapshots, not scripted demonstrations. Preserve AssembleNIST’s per-entry success-guided sampler.

Each of the five tasks gets loose, pickup, in-hand, near and partial starts. Prerequisites are already assembled and remain dynamic. Near is outside engagement; partial is engaged but unfinished and must not already earn completion. Screw partials may be supported by their actual threads.

**Unused arm:** Start nearby on its own side, gripper open, holding nothing and outside the insertion corridor. Use several reachable poses and up to ±10 mm variation inside a verified clear region. In subtask-loose starts both arms begin nearby. Both arms remain policy-controlled. Do not prescribe a supporting grasp or silently stabilize the receiver with a scripted arm.

## Historical SO101 pilot (superseded for the current WowRobo initial bank)

The following pilot coverage and stricter gates describe the earlier environment. The minimum requirement above controls the current 25 WowRobo starts.

**Variation:** Use coherent object/assembly variation, initially ±8 mm XY and ±8° yaw for loose placement, with tighter grasp/mating noise and meaningful insertion-depth variation. Record nominal layouts separately from noise. Reject large unplanned assembly rotations/translations used merely to make IK succeed. Keep observation points attached to their real bodies.

**Completion gates:** First 50 accepted states: one TRAIN and one independent EVAL per task/family. Inspect all 25 groups, arm/link/finger clearance, loaded grasp retention, prerequisites, nonterminal reward, 0.5 mm penetration/material limits and bounded-action replay. Replay partials at the finer timestep. Then build **6,400 TRAIN + 1,600 EVAL**, with disjoint seeds, no duplicate padding, resumable progress and complete current-identity publication.

A larger 640-TRAIN-per-group bank remains a later option after measured coverage/learning; it is not the current completion requirement.

**Current state:** Pilot v2 contains all 50 states, with 60 passing current production/fine-step receipts and matching saved-state/control digests. Nearby-arm geometry is checked at every control tick; receiver layout stays fixed. The yoke fix calibrates native pad contacts and preserves the nominal partial mating pose. No physical gates were weakened. Maximum sampled material overlap is 0.212 mm. All 46 earlier accepted starts received fresh replay; four missing yoke starts were generated under the final identity. The complete 8,000-state bank remains pending.

Local inspection: [all 50 states, top and side](../../../artifacts_local/training-ready/nearby-pilot-v2-review/index.html), [qualification audit](../../../artifacts_local/training-ready/nearby-pilot-v2-review/final_audit.json). These are actual accepted starts, not learned motion.

**Method:** [AlphaNIST](https://arxiv.org/html/2610.13354v1#S4.SS2) separates reset-cache coverage from adaptive sampling. Our inherited reference is `assembly_policy/sim/resets.py` and `scripts/build_banks.py`; the SO101 adaptation is `curriculum.py` / `subtask_curriculum.py`.

**For your review:** Pilot visuals, actual pose ranges and family meanings are available above. Only physically validated distributions will be scaled; this creates no new approval hold.

## Scalable state inspection

**User requirement:** The user approved the discussed design and requests images of every SGS state currently available. Future banks containing thousands of states must remain practical to browse. A representative example per group is insufficient.

**Implemented:** The [state explorer](../../../tools/sgs_viewer/README.md) provides task/family coverage, a filtered virtualized thumbnail grid, three native images per entry and one interactive 3D inspector. Shared meshes load once; native poses/details load on selection. It reuses the teleop geometry export and renderer without pose interpolation. This version inspects snapshots; native replay remains optional future work.

Every entry records bank identity, state/control identity, task, family, split, seed and exact archive index. Preview caches bind physical/render settings. Images and detailed receipts load on demand. Qualified SGS and older loose baselines are visibly distinct; historical/stale/unsupported banks are listed separately. Only published banks are supported; draft builds and run-specific sampler overlays are not implemented. Browser-local review flags do not change training eligibility.

**Verified:** 50 qualified pilot states plus 320 current-CAD loose entries are fully indexed, with 1,110 native images and 370 thumbnails. Provenance matches published NPZ rows; loose EVAL aliases are counted once. Exact saved-state parity passes for every render, and every image decodes. Eleven Python tests, three JavaScript tests and browser checks pass, including filters/reviews/3D. A separate 10,000-entry synthetic metadata test opens in 371 ms with 15 active cards and about 15 MB JavaScript heap locally; it does not establish 10,000 physical states or remote latency. [Generated explorer](../../../artifacts_local/sgs-viewer/index.html) · [browser report](../../../artifacts_local/sgs-viewer-browser/report.json). The old representative-only gallery remains historical.


## Scalable initial reset generation (November 7 fork)

Prepare1,800 slots as25×(60TRAIN+12EVAL), with configurable quotas and mathematical object-relative grasps. Keep actual motor-case/printed-body anchors, native SO101 rotational jaw calibration, passive brackets, phase-specific noise and independent split seeds. Use reachable IK followed by the two minimum native admission gates; motion is allowed. Failed draws leave explicit holes, with resumable bounded retries and CPU stage workers. The implemented framework pilot admits7/8 draws; it has not generated the full1,800-state bank. See [framework and evidence](../../WOWROBO_RESET_FRAMEWORK.md).


**Approved full expansion (November7):** The user approved the cover variation preview and asks for1,800 admitted variations across all25 starts,60TRAIN+12EVAL each. This supersedes waiting for review of other components. Preserve the approved noise and two native minimum gates; actual moving reference velocities/contact offsets must be carried coherently. Active versioned generation, counters and publication status: `outputs/reset-bank-1800-20261107-v4/{live-progress.json,supervisor.json}`. No active trainer is hot-loaded.
