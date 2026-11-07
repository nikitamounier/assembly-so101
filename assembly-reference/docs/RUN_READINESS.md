# Training run readiness

**Goal:** Separate a working training integration from readiness for the full production run.

**Verdict: the complete 50-state pilot is qualified and its bounded GPU experiment is active. The 8,000-state production run is not ready.** Reviewed November 6, 2026 (Pacific). The live run receipts establish actual update counts and outcomes.

| Area | Current evidence | Readiness |
| --- | --- | --- |
| Shared physics | Accepted v14 model; user-approved 0.5 mm penetration/material limit | Numerical baseline; hardware uncalibrated |
| Insertion / wall / thread fixtures | Existing trajectories meet 16/16 checks under the approved limit | Re-evaluated evidence, not newly simulated trajectories |
| Nearby-arm SGS pilot | All 25 task/family groups have one TRAIN and one independent EVAL start | 50/50 states; 60/60 production/fine-step receipts pass |
| Geometry / grasp / prerequisites | Sampled material overlap max 0.212 mm; held-grasp slip 0.256 mm; predecessor error 0.191 mm | Pass under unchanged additional gates |
| Pilot native preflight | Exact pilot bank, safety penalty 1, pose stability | 31/31 pass |
| Pilot CUDA learner / restore | Actual RTX 5080 PPO updates and checkpoint restore | 13/13 software checks pass; learned assembly unproven |
| Inputs / reward | Selected 448 points, 265 history values, 12 actions; sparse native completion | Implemented; learning remains to measure |
| Friction variation | Tested nominal friction for this baseline | Requested variation remains follow-up |
| Complete staged bank | No current 6,400 TRAIN / 1,600 EVAL bank | Blocks production completion |
| Resources | 16 logical CPUs, about 16 GiB RAM, RTX 5080/16 GiB VRAM; disk minimum clears | Staged smoke about 2–3 transitions/s; native physics dominates |
| Teleop | 84-test v14 baseline passes; live Quest service stopped | Wearer session separate from learner readiness |
| Publication | Latest collaboration snapshot requested for friends | Publication authorized; full-bank completion remains pending |

## Bounded experiment

The active user service `robotics-assembly-pilot-v3.service` owns `runs/so101-nearby-pilot-v3`. It uses a byte-identical private copy of the qualified pilot, frozen runtime source, eight environments/eight native physics threads and CUDA PPO. The bound is 20 optimizer updates or two hours including evaluation. It ends no later than **November 6, 11:54 p.m. Pacific**; training ends by 11:24 p.m., reserving final evaluation time. The first attempt stopped cleanly before PPO after about 15 minutes in serial assessment. This fresh attempt evaluates independent episodes in parallel, retaining all starts, seeds, horizons and physics gates. The loose report reuses all-family episodes only after proving exact state/index/seed equivalence. This is a pilot integration experiment, not the full production run.

One calibration rollout fitted input normalization with 256 transitions, zero optimizer updates, unchanged weights and zero unstable states. Initial evaluation uses immutable `baseline_policy.pt`. Before/after comparisons use the same independent subtask-loose/all-family starts and per-episode seeds with stochastic sampled actions. Calibration transitions are reported separately. Falling loss alone is not success.

The supervisor checks finite metrics, unstable states, resources and stalled updates; preserves checkpoints; and bounds restarts/evaluations. Restarts restore model, optimizer, SGS and counters, not live environment/RNG state. `runs/so101-nearby-pilot-v3/STOP` requests a graceful stop. Runtime and validator source remain frozen. Actual iteration count, rather than a clean exit alone, determines whether all 20 updates ran. Five evaluator tests and 23 launcher tests pass. Four short native motor/yoke probes produced identical serial/parallel states, controls and histories; two workers reduced their time from 52.86 to 27.47 seconds.

## Remaining gates

1. Complete the bounded experiment and compare initial/final task outcomes, safety failures, update counts and throughput. Diagnose failures before scaling.
2. Generate all 6,400 TRAIN / 1,600 EVAL states under the current identity, with independent seeds and no duplicate padding. Pilot family coverage is not production diversity.
3. Validate the complete bank, its own strict preflight/device checks and storage headroom. The current source is being published for collaboration at the user’s request; the final qualified source/data handoff and intended larger run remain separate.
4. Validate requested friction variation and hardware calibration separately. Simulation-speed, AGX and Isaac comparisons remain subsequent work.

The production command requires the complete bank:

```bash
.venv/bin/python scripts/cloud/run.py --profile configs/so101_local.json --bank-dir data/so101/banks_shared_v14_subtask_256 --run-dir runs/so101-local-baseline
```

This pilot uses an explicit one-TRAIN/one-EVAL-per-family override; production requirements were not lowered. Full-sequence assembly remains a separate evaluation.

## Review and evidence

[Sectional intent](user-intent/README.md), [50-state gallery](../artifacts_local/training-ready/nearby-pilot-v2-review/index.html), [bank audit](../artifacts_local/training-ready/nearby-pilot-v2-review/final_audit.json), [pilot preflight](../artifacts_local/training-ready/preflight-pilot-v2.json), [CUDA check](../artifacts_local/training-ready/cuda-pilot-v2/report.json), [live supervisor](../runs/so101-nearby-pilot-v3/supervisor.json), [run provenance](../runs/so101-nearby-pilot-v3/provenance.json).

Generated receipts/runs are local and excluded from source Git. Historical v10/v11 proposals, the separate 320 all-loose starts and the fixed v11 experimental tag do not certify this staged bank.
