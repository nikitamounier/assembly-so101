# Overnight training readiness

**Goal:** Distinguish the active bounded pilot from the full production run and historical snapshot.

**The 50-state nearby-arm pilot is qualified and its bounded GPU experiment is active. The intended 8,000-state production run remains pending.** All 60 production/fine-step state receipts, 31 strict pilot preflight checks and 13 CUDA update/restore checks pass. Accepted v14 physics and the 0.5 mm limit remain unchanged.

[RUN_READINESS.md](RUN_READINESS.md) records evidence, live run links and remaining gates. [Sectional user intent](user-intent/README.md) explains each part of the plan.

`robotics-assembly-pilot-v3.service` uses eight environments/eight native CPU physics threads with CUDA PPO, capped at 20 updates and two hours including independent initial/final evaluation. It ends by November 6, 2026, 11:54 p.m. Pacific. The immutable baseline uses one normalization-calibration rollout without optimizer updates. Read live learner logs for actual updates. The staged software smoke measured about 2–3 transitions/second; native physics remains the bottleneck.

The supervisor requires at least 2 GiB disk/available RAM and monitors numerical/resource health. The production profile still requires 256 TRAIN/64 EVAL per task/family. Only this labeled pilot uses one of each. Full-sequence learning and hardware transfer have not been demonstrated.

The fixed `training-v11-20261106` / `ff16c45` experiment remains available separately under [START_RUN.md](START_RUN.md). Preserve that tag and keep its banks/checkpoints separate. The user explicitly requests publishing the current source for friends. This collaboration snapshot retains the pending full-bank and learning limitations.
