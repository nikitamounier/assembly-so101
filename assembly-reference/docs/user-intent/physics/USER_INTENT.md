# CAD, contacts and shared environment

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

**November 7 hardware supersession:** The intended assembly now uses WowRobo leader modules and two WowRobo follower helpers with wrist cameras, with five structural pieces plus the dimensioned pin. The new [CAD component specification](../../WOWROBO_COMPONENTS.md) and independent review scene preserve this grouping and internal servo pivots. The review has gravity/collisions disabled and is not a replacement training factory. The Hiwonder environment and validation described below remain historical implemented behavior; its reset banks, rewards and fit evidence do not qualify the new target. Camera mounting transforms/intrinsics and the pin receiving hole still require hardware-specific information.

Training and Quest must use the same physical factory inside this AssembleNIST fork: `assembly_policy.so101.scene.model`, `workcell.py` and `configs/so101_scene.json`. Controller/render code belongs in `teleop/`.

The base begins on its side, opening upward for the motor. A completion preview keeps it sideways for the motor, then shows the assembly upright from the horn onward. Keep the accepted helper mounts and cameras. Predecessor parts remain dynamic.

**Latest explicit decision:** You accepted the v14 falls and changed the penetration/material-overlap limit to **0.5 mm**. This supersedes the former 0.4 mm rule. Placement remains 1 mm; stable retention remains 0.4 s. Sleeping, snapping, pose projection and runtime grasp welds remain excluded.

**Implemented:** One shared `PENETRATION_LIMIT_M` now feeds qualification, reward/material guards, replay diagnostics and launch defaults. Original CAD, contact stiffness, friction, masses and timestep were not changed by this acceptance update. The v14 compiled model is byte-identical to the reviewed model; eight focused boundary/guard tests pass. The recorded 0.404546 mm yoke overlap fits the new limit. Historical 0.4 mm reports/videos retain their original outcomes.

**Evidence:** [physical identity/limit receipt](../../../artifacts_local/drop-repair/v14-limit05-identity.json), [motion re-evaluation](../../../artifacts_local/drop-repair/motion-v14-limit05-reevaluation.json), [review videos](../../../artifacts_local/mating-review-v14/media-review.json). These generated receipts are local, not source-Git assets.

**Main-plan gap:** Your [original main plan](https://docs.google.com/document/d/1EHdy_YtqkQP9oIOBMTXQc2aok_c8v6XVW6KqGrLY-wo/edit?tab=t.0) requests friction variation between simulations and calibration of the press fits. Runtime friction is currently fixed. The current implementation sequence is a tested nominal-friction training baseline first, followed by validated variation; no randomized range has been silently selected. Hardware calibration remains outstanding.

**Still needed:** Current-validator SGS qualification, nominal-friction run checks followed by the planned variation, and run-specific checks. Sim-to-real accuracy is not established.
