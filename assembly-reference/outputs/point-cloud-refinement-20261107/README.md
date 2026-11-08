# Task-specific point-cloud refinement — 2026-11-07

All five current-assembly tasks retain 448 points: 128 active, 256 receiving and 64 actual fingers. The shared current-model encoder and viewer use the same tables. No policy is attached; this is a measured geometry-coverage revision, not a learning-performance result.

| Task | Receiver mating feature | Nearby surfaces | Coarse installed context | Active mating / rest |
|---|---:|---:|---:|---:|
| base_side_cover | 64 | 128 | 64 | 32 / 96 |
| shoulder_connector | 48 | 144 | 64 | 32 / 96 |
| assembly_pin | 48 | 144 | 64 | 32 / 96 |
| upper_arm_module | 64 | 128 | 64 | 32 / 96 |
| forearm_module | 64 | 128 | 64 | 32 / 96 |

Mating regions use actual source triangle surfaces close to the incoming mating geometry at its measured seat. Cover: base rails/case interface; connector: servo horns and case; upper: connector receiving pocket; final: upper-arm receiving pocket. The reference poses select triangles once and do not generate target-position points or change live object poses. Nearby points cover source surfaces within 12 mm of the seated mating geometry. The remaining 64 receiver points retain every installed predecessor with small-part presence floors. Distal fingers remain 16 points per finger, from the current model.

The pin has explicit 16-rim / 24-bore-wall / 8-blind-stop source samples. Its active 128 points reserve 8 tip and 24 shaft/shoulder samples; the remaining 96 cover the physical part. Hole samples cover all four quadrants. The nearest receiver sample to the mouth changes from 4.415 mm to 1.735 mm, with 8 now inside a 2 mm mouth neighborhood (previously zero). The center of the empty bore is not populated with fictitious geometry.

## Iterations

Compared 32, 48, 64 and 96 dedicated feature points for every task while retaining 256 total receiver points. More feature points reduce geometric probe distance but remove nearby context. Chosen 48 for connector/pin and 64 for the wider cover/upper/final interfaces; this is an engineering balance pending policy-learning ablation, not an established optimal budget.

P95 nearest dedicated-feature-sample distance over finite source-surface candidate probes (mm):

| Task | 32 | 48 | 64 | 96 |
|---|---:|---:|---:|---:|
| base_side_cover | 8.944 | 6.746 | 5.870 | 4.595 |
| shoulder_connector | 3.323 | 2.776 | 2.319 | 1.971 |
| assembly_pin | 1.722 | 1.372 | 1.194 | 0.963 |
| upper_arm_module | 6.393 | 5.255 | 4.559 | 3.673 |
| forearm_module | 6.129 | 4.914 | 4.154 | 3.445 |

These probes assess spacing over selected CAD regions. They are not a submillimetre clearance certificate or independent unseen-mesh benchmark. Full receiver points also cover some of these regions; final-audit.json records both distances.

## Verification

- All 100 native saved states checked: 96 task observations finite, exact-repeat and 448 × 3; four setup entries correctly have no task cloud. No saved state or cached table mutates.
- Each task has 384 unique active/receiver samples. Native source-mesh nearest-surface residual is below 8 nm after float32 storage. Points remain attached to their own source body, including articulated cases.
- 17 Python viewer/encoder tests and 14 frontend tests pass, including frame inversion, source-surface provenance, changed-frame rejection and state preservation.
- New observation signature and model/mesh/topology fingerprints distinguish the changed input contract. It is not silently compatible with previous policy weights.
- First construction ~6.6 s, cached thereafter. Training jobs, physics, CAD and banks are unchanged.

`final-audit.json`, `selected-point-tables.npz`, `selected-feature-config.json` and the frozen builder are the selected result. `budget-iterations.json` retains the earlier uniform-budget sweep; its `None` row was the initial 64/64/48/64/64 proposal before connector refinement to48. `previous-wowrobo-obs.py` preserves the broad predecessor sampler. Reproducer scripts are saved beside the evidence and run from the workspace root with `PYTHONPATH=assembly-so101 assembly-so101/.venv/bin/python`.
