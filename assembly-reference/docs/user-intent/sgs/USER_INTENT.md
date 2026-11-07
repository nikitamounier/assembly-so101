# SGS states and the nearby arm

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

**Passive bracket equilibrium requirement (November 7):** All25 current WowRobo starts require unpowered bracket hinges, real own-motor stop contacts, gravity/contact equilibrium with actual helper holds, and an unchanged-control hold replay. Preserve stage/nonterminal meaning; do not freeze brackets or certify poses by setting velocity to zero. Physical generation/admission remains pending until those checks pass.

**Latest WowRobo scope (November 7):** The [photo reset contract](../../WOWROBO_PHOTO_RESETS.md) supersedes the historical unused-arm-open rule below for this new hardware sequence. The other arm holds the connector during pin insertion, LEFT supports base/connector during upper-arm installation, then RIGHT holds the installed upper arm during final-piece installation. Explicit support transitions precede release; both arms remain policy-controlled. Old pilot qualification does not qualify these new states.

Follow [plan.md](../../../plan.md) and the decisions from “sgs states1.” Build physical starting snapshots, not scripted demonstrations. Preserve AssembleNIST’s per-entry success-guided sampler.

Each of the five tasks gets loose, pickup, in-hand, near and partial starts. Prerequisites are already assembled and remain dynamic. Near is outside engagement; partial is engaged but unfinished and must not already earn completion. Screw partials may be supported by their actual threads.

**Unused arm:** Start nearby on its own side, gripper open, holding nothing and outside the insertion corridor. Use several reachable poses and up to ±10 mm variation inside a verified clear region. In subtask-loose starts both arms begin nearby. Both arms remain policy-controlled. Do not prescribe a supporting grasp or silently stabilize the receiver with a scripted arm.

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
