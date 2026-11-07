# Hybrid smoke and current-CAD readiness — November 7

**Infrastructure smoke PASS; latest-CAD main training BLOCKED.**

The reference chats `states & variations` (01a1b5de-9df9-7f83-a66f-b5933514cdcb) and `cad fix` (01a1b69f-b569-7a80-af79-4cc72efb78e9) were read. The candidate is final-common-v1 with the 14 mm pin, components-repaired-v1, photo-resets-repaired-v1, randomization-bank-repaired-v1 and physical-resets-repaired-v1. Guided insertion is not free bimanual training qualification.

## Mixed-provider smoke

Receipt: `../../../assembly-so101/operational/hybrid-smoke-20261107/20261107T185515Z/result.json`. App ap-I4CwYH51EvcC24k4WmRylk; real worker hosts GPU1298 and modal, matching model/bundle signatures, MuJoCo 4.0.0. Eight environments, four parity steps, eight exercised resets, zero observation error. Two real PPO updates with 256/256 valid transitions each, finite losses, parameters changed; exact policy/optimizer/sampler checkpoint restore. Tiny-fixture throughput ~160 transitions/s, per-step median remote latency 40.8–41.4 ms. These are infrastructure checks, not learning success, current-CAD qualification or large-fleet scaling evidence.

Modal used one 4-core learner and one 1-core worker; Chimera used one CPU in existing allocation3077705. No cube application, allocation ownership or source changed. Temporary endpoint carried pinned TLS and per-run HMAC. After completion, both this app and the interrupted first image-build app ap-L16fvFc26JhRItazftf4qS were verified stopped with zero tasks. No main run launched.

## Current blockers

- Repaired review summary reports **0 physically qualified resets**, 16/25 nominal geometry passes and failures21,22,23,25,26,28,29,30,31; 75 variations have29reject/46pending. Preserve rejection labels.
- Reduced-noise trials do not repair zero-noise pose/support overlaps. State23 still has ~1.94 mm overlap in the later ablation.
- Default-window physical probes fail loose settling and loaded right support retention; pin-half right pads0/0N and relative drift7.10mm. See `../repaired-reset-physics-20261107/`.
- Trainer's SO101 suite still imports the historical environment; the candidate needs an actual compatible observation/reward/reset/backend integration. Never route it through the old model as a fallback.
- Need current-CAD mixed-provider parity, full PPO/resume, and measured full-batch throughput/cost acceptance after physical admission.

`qualification.json` records these failures and hashes exact inspected inputs. The disabled main template references it. The cloud supervisor refuses a live launch unless all required checks are explicitly qualified and listed artifact hashes still match. This is an operational interlock, not a physical validator; a future qualifying receipt must include the complete frozen source, model/grid/mesh/reset/observation identities and genuine validation evidence.

## Performance audit

`probe_performance.py` and `performance.json` use current native CAD in an isolated plugin build, sparse Jacobian and accelerated surface tree mode1, unchanged solver/contact parameters. Two rejected diagnostic states, three repeats of eight steps each; not a long rollout benchmark. Physical model SHA25689b7a9b3f7473bd083713ec4264fbfc610ea2422f35f52acc62a97d61da5c4a0.

Current timestep250µs =>160 substeps/40ms action, versus cube2ms =>20. Do not assume cube throughput, or an exactly8x runtime ratio. Measured split stepping took45.8ms per8 physics steps for loose and120.0ms for pin-half on this Mac, about0.92s and2.40s per40ms control interval if those short-window costs persisted. Contact-state and hardware dependent; not a cloud CPU sizing result.

Split stepping vs plain mj_step with unit friction:0.986x/1.043x runtime, final integration states exactly equal. Passing cached geometry roles under nonunit friction:1.000x/1.027x speedup, also exactly equal states. No material metadata bottleneck established; no speculative physics/performance patch applied. Preserve native contacts, source boundary guards, solver tolerances and timestep until physical qualification supports any change. Profile native contact search/solver, reset amortization, batched policy/communication and slow-worker barriers at the intended fleet size before calling the production throughput acceptable.
