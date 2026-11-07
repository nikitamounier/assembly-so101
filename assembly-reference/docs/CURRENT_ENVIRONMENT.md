# Current shared environment

**Goal:** Identify the physical model, the current acceptance policy and what is ready to use.

## Current source

Local development uses **shared-side-v14-horn-support**, MuJoCo 3.17.0. Training and the repository-owned Quest adapter use the same scene factory. It has 27 bodies, 65 positions, 58 velocities, 12 actuators, three cameras, no attachment equalities and a 0.5 ms timestep.

| Identity | SHA256 |
| --- | --- |
| Physical model signature | `be9e8e9082d7f1b94de9b2ea2efaf450d97f2b18f5d17bb82ad8ced7fedcb72b` |
| Configuration | `f4d1002adfb316f8611d464e88026d23c6629b0d7be4af8af8b29f2d667a80de` |
| Serialized monitored model | `79879aefb70ecda54f599e3801c7a59f8f147a7e57c83dbed190f94564a5ca10` |
| Native contact adapter | `7fa910ed64699face3ca62a09f5c076a5ef2f3d5f7d78bae6c0cb56da939115b` |

**Latest user decision:** Accept the v14 falls and use **0.5 mm** native penetration/original-material acceptance, superseding 0.4 mm. The shared constant is `assembly_policy/so101/limits.py`; validator and target-cache identities include it. The physical model is byte-identical before/after this policy change. Contact impedance widths and geometry/noise dimensions are not acceptance thresholds and were preserved. Historical reports keep their original limits and outcomes.

## Verification and readiness

The unchanged physical v14 baseline passed 82 core and 84 teleop tests. Under the new policy, eight focused acceptance/guard tests, a fresh 31-check guarded preflight and all 13 actual CUDA update/restore checks pass. The existing 16 insertion/wall/thread checks meet the new limit on re-evaluation; their trajectories were not rerun or relabeled. The earlier coarse yoke peak is 0.404546 mm; fine-step peak is 0.393227 mm. [Detailed evidence](ASSEMBLY_MATING_REPAIR.md).

The **50-state nearby-arm pilot is qualified and its bounded training experiment is active**: all 60 production/fine-step receipts, 31 strict pilot preflight checks and 13 CUDA checks pass. The user freed storage, clearing the launch minimum. The full 8,000-state production bank remains pending. [Run-readiness audit](RUN_READINESS.md), [user-intent sections](user-intent/README.md).

The 320 full-sequence all-loose v14 starts and reference remain under `data/so101/banks_shared_v14` and `shared_v14_reference`. They are not the staged bank. Historical SGS proposals need fresh validation. The default staged training path remains `banks_shared_v14_subtask_256` and is absent; the loader fails closed. Selected observations remain 448 points and 265 history values, with 12 actions at 25 Hz.

## Runtime and publication

The local Quest service is stopped; no live wearer session was checked. The supervised pilot uses eight environments/eight native physics threads and CUDA PPO; it is bounded to 20 updates/two hours. Native physics is CPU-based. Hardware geometry/friction/compliance, robot grasp/tool execution and sim-to-real transfer remain unqualified.

Remote main was last verified at experimental v13 `85c006a`. The fixed v11 tag `training-v11-20261106` remains at `ff16c45`; [its instructions](START_RUN.md) describe that separate experiment. The user now requests publishing current source for collaboration; this is separate from the pending final SGS completion handoff.

Original CAD/grids, upstream PPO/SimBa/SGS, helper mounts/cameras and the physical-button HOTO driver are preserved. Numerical contact approximations include continuous original-surface motor/base and motor/horn supplements, source-contained contact patches and a 0.5 mm yoke/table contact skin. There is no sleeping or pose projection.
