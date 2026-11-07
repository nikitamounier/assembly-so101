# Mating validation and v14 review

**Goal:** Preserve the physical evidence and distinguish current acceptance from historical failures.

## Accepted policy and model

The user reviewed the v14 videos, accepted the fall behavior and explicitly selected **0.5 mm** penetration/material acceptance on November 6 (Pacific). Placement remains 1 mm; release/stability criteria remain unchanged. The shared limit now lives in `assembly_policy/so101/limits.py` and is part of validator/cache identity.

Physical v14 remains `be9e8e90…`, with native adapter `7fa910ed…` and serialized model `79879aef…`. The policy update did not change CAD, contact stiffness, friction, masses, timestep or the compiled model. [Exact identity](CURRENT_ENVIRONMENT.md).

The earlier 0.4 mm gate rejected the coarse yoke path at **0.404546 mm original motor/yoke overlap**. Its final placement/reward passed. The fine path passed at 0.393227 mm. Re-evaluating those same measured trajectories at the user-selected 0.5 mm limit gives **16/16 insertion, wall and thread checks**. Old reports and video labels remain unchanged; this is not a claim of fresh trajectories or zero overlap.

At the worst yoke pose, the native search reported only 0.124518 mm while the original surface audit measured 0.404546 mm. A private supplemental-contact prototype detected 0.407560 mm, but its coarse insertion stopped 14.397 mm off target and failed release. It was not adopted. Independent original-material checking remains necessary.

## Evidence

Local receipts are under `artifacts_local/` and are not included in source Git.

| Check | Result | Receipt |
| --- | --- | --- |
| Compiled physical equivalence after limit update | Exact match | `drop-repair/v14-limit05-identity.json` |
| New acceptance boundary/guards | 8 passed, including 0.45 mm accepted and 0.55 mm rejected | `drop-repair/limit05-tests-final.log` |
| Fresh guarded native preflight | 31 passed | `drop-repair/preflight-v14-limit05.json` |
| Fresh RTX 5080 CUDA learner/update/restore | 13 passed; all-loose software smoke | `drop-repair/cuda-v14-limit05/report.json` |
| V14 software baseline before limit change | 82 core / 84 teleop passed | `drop-repair/pytest-{core,teleop}-v14.log` |
| V14 reward baseline before limit change | 34/34 passed | `drop-repair/rewards-v14.json` |
| Recorded motion at both timesteps | 16/16 under 0.5 mm; original coarse 7/8 under 0.4 mm retained | `drop-repair/motion-v14-limit05-reevaluation.json` |
| Five prepared targets | Native reward qualified | `mating-v14-targets/report.json` |
| Full-sequence loose starts | 320 accepted / zero rejected | `drop-repair/prepare-v14.log` |

## User-requested media

[Prepared mating close-ups](../artifacts_local/mating-review-v14/matings.jpg), [26.9 s insertion/thread reel](../artifacts_local/mating-review-v14/v14-force-insertions-and-thread.mp4), [35 s gravity-drop reel](../artifacts_local/mating-review-v14/v14-gravity-drops.mp4).

These are independent native tests. Regular-part insertion uses bounded force/torque alignment and receiver support, followed by active-part release and a separate fully unsupported native reward check. No part pose is assigned after initialization. The screw comparison is an explicitly guided analytic thread bench, not screwdriver execution. Videos replay saved states at 1× simulation time.

The drops are vertical gravity releases at nominal XY/rotation with all bodies dynamic. All five pass their recorded physics checks; none reaches the 1 mm placement tolerance. Final errors are motor 11.806 mm, horn 2.061 mm, screw 3.393 mm, retainer 27.089 mm and yoke 120.179 mm. Yoke normally enters laterally, so its vertical release is a drop stress test. A physics pass is not assembly success. The user accepts these fall outcomes.

Media inspection, trace/model bindings, hashes and scope are in `mating-review-v14/media-review.json` and the per-clip receipts. The original 0.4 mm yoke video correctly retains its FAIL label. Held-out audits have not been represented as complete merely because primary checks or visuals pass.

## Remaining work

Implement and validate nearby open unused-arm starts, inspect the 50-state pilot, complete the 6,400 TRAIN / 1,600 EVAL bank and run checks against that exact bank. Preserve historical proposals and independent evaluation. No long learner, final SGS completion handoff or new source push has occurred. [Readiness audit](RUN_READINESS.md), [SGS plan](../plan.md).

Actual hardware calibration, robot grasp/insertion, tool pickup/button driving and learned full assembly remain unproven. Original vendor geometry and NIST SDF/driver sources are unchanged. Completed pre-v11 private diagnostics were archived with verified hashes; restoration instructions are in `artifacts_local/mating-repair/ARCHIVED.json`.
