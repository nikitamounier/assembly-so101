# Actual assembly on H100: experimental MuJoCo Warp port

The optional custom SDF/contact backend passes 12 bounded, final-source H100 checks using the actual accepted-v6 assembly model and saved starts. Production training continues with native MuJoCo. No production backend switch or PPO qualification was performed.

## Port and physical identity

[Backend](../../assembly_policy/so101/wowrobo_warp.py), [runner](runner.py), [results](results.json), [offline validation](validation.json).

- Model signature: `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`; 130 geometries, 17 explicit pairs, 54 SDF plugin instances. Accepted-v5 and v6 have the same physical model; final results use v6.
- Backend SHA256: `794b1a90168542cfc062f8efd9e7e6af4c21a27aeae43cd80f24b045e3cc5f51`. Runner SHA256: `f2c15ad85a0c8a78054553051f4d548af5f5dde29dfa547c0cbf93b1baa07c7c`. Saved source copies and native build receipt are in [results/source](results/source).
- Ported original grid interpolation/gradient to FP64 GPU evaluation with lossless decoded grid samples. Ordinary MJWarp hook outputs and contact buffers remain float32. Packed attributes retain original double headers exactly.
- Ported the full original vertex/triangle-centroid sampling, deterministic deepest sample selection, 12 signed-normal bins, original pair-capacity/replacement policy and contact supplement insertion before constraint construction. No simplified collision geometry.
- Per-world native friction roles and three episode multipliers are applied before constraints. Minimum-contact sensor, finite checks and buffer-overflow guards are included.
- Original per-pair SDF budgets (4/10 or 40/40), 0.00025 s timestep and 1e-12 solver tolerance retained. No relaxation of native contact laws, iteration limits or the 0.5 mm acceptance guard. Private patched MJWarp SDF module; no installed package edits. Optional dense Jacobian storage uses a private model copy; native storage remains the default.
- Production imports are unchanged. The experimental module requires explicit loading and original-source paths; global hooks are confined to its separate process.

## Final qualification panel

All 12 records completed, matched the final source hashes/model identity, remained finite, and passed the saved-evidence validator.

| Check | Result |
|---|---|
| SDF distance/gradient, 3,726 samples over 54 instances | Exactly zero sampled error against native plugin |
| Custom supplemental contact selection on tested poses | Counts match; exactly zero depth error; positions within 1 µm and normals within 1e-5 |
| Independent effective native friction comparison | 2,935 matched contact rows; maximum coefficient error 3.652e-8 |
| Peak penetration over all final GPU/native tests | 0.173549 mm; guard 0.5 mm |
| Maximum free-body position difference from native | 25.856 µm |
| Maximum free-body orientation difference | 0.050435 degrees |
| Native numerical warnings | Zero |

Query sampling includes 64 interior points plus bounds, outside bounds and center per instance. Supplemental contact comparison uses matched float32 geometry transforms, independently calling the original native supplement. It establishes local contact-port behavior at those poses, not all possible configurations.

Actual saved task starts: cover `05_cover_near`, connector `10_connector_near`, pin `15_pin_near`, upper `21_upper_far_side`, final `29_final_far_side`. Each received 160 substeps (40 ms). Additional checks: accepted friction variation on upper; 800-substep (200 ms) passive upper and final; 800-substep cover with a small commanded arm action (actuator 1 increment 0.00175 rad per control interval, five controls); and two 32-world cover batches. Batch worlds replicate the same saved start.

## Measured physics throughput

Single bounded measurements after compilation/warmup, replaying reset integration state including warmstart. A control interval is **160 physics substeps**. Native timings use one simulation thread, on the same allocated test host. These numbers exclude observations, policy, reset scheduling, data transfer and PPO.

| Workload | GPU wall seconds | Native one-world seconds | Aggregate gain vs one native thread |
|---|---:|---:|---:|
| Cover, 160 substeps | 2.293 | 3.571 | 1.56× |
| Connector, 160 | 2.873 | 4.388 | 1.53× |
| Pin, 160 | 2.948 | 5.814 | 1.97× |
| Upper, 160 | 3.910 | 9.198 | 2.35× |
| Final, 160 | 4.676 | 11.843 | 2.53× |
| Upper varied friction, 160 | 3.816 | 9.074 | 2.38× |
| Upper passive, 800 | 19.192 | 45.130 | 2.35× |
| Final passive, 800 | 23.050 | 58.716 | 2.55× |
| Cover commanded arm, 800 | 11.624 | 15.250 | 1.31× |
| Cover, 32 worlds × 160, native sparse storage | 12.022 | 3.231 | 8.60× |
| Cover, 32 worlds × 160, private dense storage | 9.900 | 3.168 | 10.24× |

The batches deliver 2.66 and 3.23 aggregate physics control intervals/second respectively. Gains use `worlds * native_one_world_seconds / gpu_seconds`. They are not gains against 32 CPU workers or the existing fleet, and do not establish a 10–20× complete-PPO speedup. Dense storage was tested only on this cover batch. The solver remains the dominant GPU cost with the original tolerance.

## Isolation and main-run receipt

Remote test root: `/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-port-20261107`.
Frozen source root: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4`.
Isolated dependencies/cache: `/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/{.venv,warp-cache}`.

Tests used explicit bounded overlapping steps in existing idle retained allocations. No allocation submission, cancellation, manual requeue, main policy/checkpoint/controller/monitor/source mutation or driver/base-environment change. Jobs independently preempted/self-requeued during the session: the initial spare 3606264 later became the main owner's learner on GPU1112; our guard refused that busy device. Final tests ran on idle 3598717/GPU105E. GPU-process and controller-ownership guards yield if another GPU consumer appears or a fresh controller selects our allocation. No test remained running after the panel.

The final [read-only receipt](main-status-receipt.json), at approximately 17:28 Pacific Nov 7, shows learner 3606264/GPU1112 running, controller age 6.8 s and committed iteration 19. Both retained jobs were running. This establishes continued main-run progress; it is not a formal measurement of zero shared-host contention.

An extra CPU-only metadata audit using a 4 GB relay step was OOM-killed; its logs remain `native-and-main-receipt.stderr`. This was an isolated inspection step, not a physics result or the learner. It was not repeated because the completed model/signature/option evidence already exists in the final test artifacts.

## Numerical fixes and limitations

Exploratory failures are retained in logs/remote `initial-round`, excluded from `results.json`. Fixed optional rigid-only contact arrays of size zero; corrected native centroid arithmetic (float32 vertex division before double accumulation); disabled fused multiply-add in the custom kernels/private SDF module; matched native reciprocal-times-gradient normalization. Without the last two changes, tiny arithmetic differences flipped deepest-sample ties and contact bins on flat surfaces. Final-source query and depth errors are zero on the qualification panel.

Ordinary MJWarp SDF contact manifolds still differ from native MuJoCo (for example, different total contact counts). Matching the custom supplement does not make the full engines equivalent. Initial constraint-force diagnostic differences around 1–4% are not force-parity qualification: GPU warmup versus native cached/warmstart state was not controlled for an exact-force test. No long episode, full learned assembly, observations/rewards/PPO integration or training-regression qualification was performed. A separate long-horizon and policy/control panel is required before considering activation. There is no automatic activation path in this experiment.

## Reproduce or inspect

Dependencies are pinned in [requirements.txt](requirements.txt): MuJoCo 4.0.0, mujoco-warp 4.0.0, Warp 1.19.0+cu12. H100 driver 560.35.05 supports the CUDA12 wheel; the default CUDA13 wheel did not work here. The official wheel used was `https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl`.

Offline artifact validation from this repository (any Python with NumPy):

```sh
python outputs/mjwarp-port-20261107/verify_results.py
```

The preserved remote snapshot and isolated venv are prerequisites for rerunning. `launch.py` deploys only this experimental module/runner into that test root. Select an actually idle retained allocation; the historical job ID is not a perpetual reservation. Example original panel commands:

```sh
WARP_JOB=3598717 python outputs/mjwarp-port-20261107/launch.py '--mode query --bank-version 6'
WARP_JOB=3598717 python outputs/mjwarp-port-20261107/launch.py '--mode assembly --bank-version 6 --case 21_upper_far_side --steps 800'
WARP_JOB=3598717 python outputs/mjwarp-port-20261107/launch.py '--mode benchmark --bank-version 6 --case 05_cover_near --worlds 32 --steps 160 --jacobian dense'
```

Reruns replace that case's result files in the remote experiment root; archive old results first when retaining a comparison. `remote.py` accepts allocated-step Python through stdin and propagates the SSH/step exit code. Source snapshots, query/contact/trajectory NPZ files and all 12 final JSON records are preserved under `results/`. The earlier source snapshot manifest names the initial v5 bank; each final record explicitly identifies v6 and the shared physical signature. Local Ruff and saved-artifact validation pass.
