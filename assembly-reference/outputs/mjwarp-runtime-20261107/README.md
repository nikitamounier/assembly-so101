# Current WowRobo MuJoCo Warp runtime

Follow-up: [deep profiling and selected faster runtime](../mjwarp-profile-20261107/README.md) now measures7.41× versus one frozen native thread and scaling through2048worlds. The results below describe the original runtime baseline.

Optional batched runtime for the current native-v13 physical model. This extends the earlier custom SDF/contact prototype into the native trainer's rollout interface. Live training was not switched.

## Implementation

- `assembly_policy/so101/wowrobo_warp_runtime.py`: native `Rollout.rollout` interface; FULLPHYSICS, per-world controls/weld activation/userdata, explicit reset warmstarts, three friction draws and four measured-hinge endpoint offsets.
- Reusable CUDA graphs integrate and record every substep. CPU observation/reward code and endpoint native forwards stay with the existing environment. Native solver tolerance, iteration budget, geometry, timestep and search budgets are retained. Dense Jacobian storage is an explicit private model setting.
- GPU batched joint ranges carry each world's hinge noise without touching shared native arrays. Per-substep capacity/overflow flags are latched; input shape/finiteness/equality/friction/noise and raw float32 integration clocks are checked. The returned API clock uses exact float64 integer-step time after validation.
- Custom contacts retain FP64 SDF interpolation, native sample ordering/normal bins and native friction roles. Engine state and ordinary contacts remain float32. Engines differ numerically and in ordinary manifolds.
- Process-global custom search/context hooks are restored by `close()`. One runtime per isolated process. The port also stops a pair immediately when supplemental allocation overflows, avoiding access to an uninitialized replacement slot.
- `scripts/train_wowrobo_warp.py`: explicitly selected current-trainer entry point; backend/source/version/options signatures in config/checkpoints; separate experimental status; original PPO, observations, reward, reset sampler and checkpoint guards. Existing native checkpoints cannot be silently resumed under Warp. The native evaluator is excluded from this entry point.
- Optional dependency extra: `wowrobo-gpu` (MuJoCo/MJWarp4.0.0, Warp1.19.0). H100 driver560 uses the official CUDA12 Warp wheel from the previous port's pinned environment; the default CUDA13 wheel is incompatible with that driver.

## Validation

Isolated retained H100 allocation3606264, GPU720E; native-v13 model signature `13ac7a8d07d4d63e7de21bd426319fd32e60acfd33e9fd681a43a2d1f8cd959e`, diagnostic1143-row bank. Existing live learner is in a different retained allocation. An idle-device guard and competing-GPU-process watcher protect test ownership. All test outputs are under `/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-runtime-20261107`.

Two-world/four-substep smoke passed graph reuse, finite observations/state, partial reset and hook restoration. Tiny-batch warmed GPU execution was slower than serial native; no universal speedup claim.

Five distinct tasks/stages,80substeps ×4controls (160ms), varied saved states: rewards equal for all20world-control results; no safety/unstable flags. Largest endpoint body-position difference from native44.1µm; qpos maximum1.774e-3 (mixed position/quaternion/joint coordinates), state maximum mixes coordinates/velocities and is not a distance metric. Warm intervals after the first call: serial native23.8427s vs GPU14.0064s, **1.702×**. This is one matched sequence against one native simulation thread, includes environment endpoint refresh, and does not compare to five parallel CPU workers or establish a full-PPO speedup. Graph construction/compile is excluded from that warm comparison. The comparator is frozen installed native-v13; the later promoted general-separation/four-way-cache candidate is not included.

One actual PPO update through the new entry point completed:2worlds ×2controls,4valid transitions, finite policy and45Adam states, durable checkpoint SHA256 `78709cfb11a4a1ec73268dd401c0f6f0ea53fa0a8d6b3047ae03ed09fb3fd84e`; collection7.0s/update0.2s. Fresh diagnostic policy, not production continuation. [Checkpoint verification](results/ppo-w2/checkpoint-proof.json). The checkpoint predates only the defensive supplemental-overflow early return; its exact source is preserved in `ppo-tested-wowrobo_warp.py`. Runtime adapter bytes match the final source.

The first16-world attempt was interrupted by the allocation's scheduled self-requeue; rerunning after the retained lease returned passed with the final source.16worlds spanning all5stages,80substeps ×3controls:48world-control rewards match, zero safety/unstable flags, maximum body-position difference24.6µm over120ms. Warm intervals excluding the first call: serial native48.8371s vsGPU18.1208s, **2.695×**. Partial reset, two malformed-input rejections and hook restoration pass. [Final-source evidence](results/w16-s80-c3/result.json). This remains one matched short sequence against one simulation thread; no16-parallel-CPU or full-PPO performance claim. No scheduler job was submitted/canceled/requeued manually; no live source/checkpoint/controller was changed. The supplemental-overflow early return compiles and the final source passes this panel; deliberate overflow injection remains untested.

No full episode, learned insertion success, held-out admission panel, full19200-bank qualification or production activation is established. Full physical behavior and fleet economics remain to be qualified. Do not infer engine equivalence from the bounded tests.

## Use

The physics stays in this repository. Supply the current reset/warmstart/friction-aware trainer explicitly (the historical default local bridge may precede this schema). Use a separate process and run directory. For example, with a compatible current trainer and valid bank:

```sh
python scripts/train_wowrobo_warp.py \
  --trainer-root /path/to/current/assembly-so101 \
  --scratch /path/to/private/warp-scratch -- \
  --suite wowrobo --remote-workers 0 --run-dir /path/to/new/warp-run \
  --bank-dir /path/to/current/bank --envs 16 --nthread 1 --no-eval
```

For diagnostic banks, explicitly add `--wowrobo-debug-resets`. Preserve `WOWROBO_CANONICAL_ROOT` if the versioned trainer's bridge requires it. Set `WOWROBO_WARP_GRID_SOURCE` and `WOWROBO_WARP_FRICTION_SOURCE` when the matching native sources live elsewhere.

To attach to an already constructed current environment in an isolated process, call `install_on_environment(env, scratch)`. Persist `env.warp_identity` with the run metadata. Native safety checks require the full per-substep trajectories returned by the engine.

Reproduction on an actually idle retained GPU (historical IDs are not permanent reservations):

```sh
python outputs/mjwarp-runtime-20261107/launch.py --job ID --worlds 5 --steps 80 --controls 4
python outputs/mjwarp-runtime-20261107/launch.py --job ID --worlds 2 --ppo
```

The launcher writes only to the private experiment directory, reuses pinned installed dependencies and refuses a busy GPU. Qualification selects cover/connector/pin/upper/final stages with distinct saved rows, exercises the unchanged environment step/observation interface, malformed-input rejection and a partial-world reset. Initial integration attempts exposed stale dependency/source paths and the installed one-dimensional overflow-array ABI; these were corrected before the successful panel.
