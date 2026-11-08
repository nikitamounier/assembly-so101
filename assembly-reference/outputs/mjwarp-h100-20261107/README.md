# MuJoCo Warp on Chimera H100 — 2026-11-07

The bounded probe completed successfully on retained allocation3606264/GPU71E4. GPU execution works with a CUDA12 wheel, and the frozen WowRobo model imports with its exact native identity. **There is no valid assembly GPU throughput or learning result yet:** the task's custom contact/friction implementations require an explicit port.

## Environment

- H10080GB HBM3, driver560.35.05; MuJoCo4.0.0/MJWarp4.0.0/Warp1.19.0+cu12.
- Isolated environment/cache/results: `/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107`.
- Initial PyPI Warp1.19.0 wheel initialized CPU only because it required CUDA13. Replaced only the isolated wheel with NVIDIA's official CUDA12 build. Shared drivers/training dependencies unchanged.
- Original observation-wrapper attempt lacked `igl`; final probe uses the canonical native physics loader directly. It does not require observation generation for this import check.
- Both partitions checked with1GPU/2CPU/16GiB/30min test-only: ordinary projected next day; quick listed preemption victims. No new submissions or preemptions. The retained3606264GPU was idle; a one-second watcher would exit the probe if another GPU process appeared. Final receipt reports no GPU processes.

## Actual assembly compatibility

Frozen accepted-v5 model SHA `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879` matches the bank manifest.51positions/46velocities,130geoms,17explicit contact pairs,54SDF plugin instances,0.25ms timestep. Stock `put_model` succeeded in0.185s. This snapshot predates any separately owned v6 continuation.

Installed `collision_sdf.py` defaults for `user_sdf` and `user_sdf_grad` print an error and return zero distance/gradient. Importing a model does not compile/transfer its native C++ SDF implementations. Our supplemental surface contacts and per-world friction hook also remain native-only in this probe. Therefore no assembly GPU stepping or training was performed with these missing implementations. The importer also clamps native solver tolerance to at least1e-6; a future port must record/qualify that difference.

## Synthetic GPU execution and scaling

Sphere falling onto a plane,0.25ms timestep,160substeps per40ms interval. CUDA graph contains one physics step and is launched160times; three consecutive intervals timed after compilation. CPU reference advances480steps from the same initial state. This includes Python graph dispatch and GPU synchronization, but excludes task resets, observations, rewards, policy and PPO. GPU state resets after the warmup step; only position/velocity/time are explicitly reset, so this is a smoke/scaling test, not rigorous solver-state equivalence.

| Parallel worlds | Median wall ms /40ms simulated | Aggregate world intervals/s |
|---:|---:|---:|
| 1 | 30.40 | 33 |
| 64 | 30.68 | 2086 |
| 256 | 35.09 | 7295 |

All three cases finite; max final position difference versus native6.385e-9m. First GPU step/compilation48.43s. A single native world needed about0.80–0.84ms per160steps, so GPU loses badly at one world and gains aggregate throughput by batching. No assembly speedup can be inferred from this simple scene.

## Next experiment

1. Implement current54grid distance/gradient hooks on GPU; compare pointwise values and gradients with native, including surface/grid boundaries.
2. Port supplemental sample/tree contact generation and per-world friction at the corresponding solve stage. Preserve contact laws, passive stops and full reset/control/velocity/warmstart semantics.
3. Qualify grasp retention, blocked insertion, penetration, passive motion, observations/rewards and termination before timing real assembly rollouts.
4. Benchmark full valid transitions/PPO iteration timing on64/256/1024worlds, with memory/overflow checks. Historical SO101 Warp prototypes are a different model and do not qualify this one.

Reproduce the bounded probe in an idle authorized H100 allocation (script contains a GPU-conflict guard):

```sh
srun --jobid=<authorized-id> --exact -n1 -c2 --mem=16G --time=00:15:00 \
  /large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/.venv/bin/python -u \
  /large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/probe.py
```

Evidence: [result.json](result.json), [probe.py](probe.py), [probe.log](probe.log), [requirements.txt](requirements.txt), [receipt.json](receipt.json), [hook audit](hook-audit.stdout). No learner restart, policy update, backend migration or monitor change.

Official references: [Warp CUDA12 wheels](https://nvidia.github.io/warp/v1.19/user_guide/installation.html), [MJWarp API/SDF hooks](https://mujoco.readthedocs.io/en/latest/mjwarp/).
