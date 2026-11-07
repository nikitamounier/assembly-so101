# Noisy SGS starting states

**Goal:** Supply physically checked loose, pickup, in-hand, partial and near starts for all five assembly tasks. Prepared starts are not insertion demonstrations or a learned policy.

This is the historical v9 workflow and evidence. Current approved training uses [balanced subtask starts](SUBTASK_CURRICULUM.md); v9 receipts do not qualify later physical profiles.

## Verified local bank, November 6, 2026

`data/so101/banks_shared_v9_sgs_noise_v1` contains **514 states**: 287 TRAIN and 227 family EVAL, including the original 320 loose states. It is qualified against profile `shared-side-v9-hoto-s-a01`, model signature `99f8a29b33345a667b64a861ae38e9e4822f07ed13c37c6e92dc50605b9f8793`.

Counts below are TRAIN / EVAL; every task also retains 32 / 32 loose states.

| Task | Pickup | In hand | Partial | Near |
| --- | --- | --- | --- | --- |
| Motor | 10 / 5 | 10 / 5 | 6 / 3 | 10 / 5 |
| Horn | 8 / 4 | 8 / 4 | 8 / 4 | 8 / 4 |
| Screw | 8 / 4 | 3 / 2 | 3 / 2 | 3 / 2 |
| Retainer | 8 / 4 | 8 / 4 | 6 / 3 | 3 / 2 |
| Yoke | 8 / 4 | 3 / 2 | 3 / 2 | 3 / 2 |

All 194 added states passed fresh canonical replay; the original 320 loose states passed a separate 0.32 s replay. Maximum sampled material overlap was 0.395 mm, sampled native depth 0.146 mm, predecessor error 0.499 mm and held translation slip 0.490 mm. All nine retained retainer partial starts also passed at 0.25 ms; two otherwise accepted starts failed that extra check and were excluded. Representative partial starts for the other four tasks and screw/yoke in-hand starts passed at 0.25 ms; this is not a second-timestep audit of the entire bank.

The training loader preserves task/family/state mapping; all 25 combinations produce finite point-cloud and proprioceptive observations. The 50-entry, two-per-family selection also passes. Evidence: [physics and coverage](../artifacts_local/curriculum-final-audit/summary.json), [training integration](../artifacts_local/curriculum-final-audit/integration.json). No policy training or source publication was performed by this curriculum task.

## Build and select

Run from the repository root after native setup:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-teleop/bin/python -m assembly_policy.so101.curriculum \
  --out artifacts_local/curriculum_proposals \
  --train-count 8 --eval-count 4 --attempts 128
```

Counts are targets, not guaranteed coverage. The generator rejects invalid candidates and records failures. `--tasks`, `--families`, `--seed-offset` and repeated `--grasp-source` permit additional independent batches using calibrated measured grasps. Outputs are built in a hidden sibling directory; existing outputs are never overwritten.

Freshly replay saved proposals before merging. For example:

```bash
PYTHONPATH=. .venv-teleop/bin/python scripts/check_so101_curriculum_bank.py \
  artifacts_local/curriculum_proposals --out artifacts_local/curriculum_replayed
PYTHONPATH=. .venv-teleop/bin/python scripts/merge_so101_curriculum.py \
  --out data/so101/banks_shared_v9_sgs_noise_v1 artifacts_local/curriculum_replayed
```

The merger requires every family in both splits for all five tasks, current validator/native-binary hashes, canonical 0.5 ms timestep, matching saved arrays and independent seeds. An optional `--additional-check DIRECTORY` restricts each audited task/family to states that also passed that extra replay (for example, at 0.25 ms); missing or failed states in those groups are excluded, not silently grandfathered in. It rejects stale receipts, duplicate states and material/grasp-limit violations. Inspect actual coverage before use. Pass the new path explicitly as `--bank-dir`; default training selection remains unchanged. The loader rejects a prepared bank when the shared environment or validator changes. Replay after any mating repair; do not merely rewrite its manifest.

## Noise and preparation

- Workcell and independently loose parts: table XY up to ±3 mm; yaw ±2°.
- Generic grasps: translation ±0.15 mm (screw ±0.01 mm); rotation up to ±0.3°, reduced for partial starts. Approach offsets vary by up to ±0.5 mm laterally and ±0.3 mm axially, with tighter screw/retainer bounds.
- Measured grasps: fresh background and receiver scenes, with arm-joint noise ±0.003 rad. TRAIN and EVAL use separate seeds and states but share calibrated grasp parameters; they are not independent grasp-design tests.
- The five-axis arms sometimes need a different nominal receiver pose. Coupled preparation can rotate the free assembly by up to π and translate it up to 60 mm X / 80 mm Y. This is recorded preparation, separate from small random noise. Clamped arms stay fixed; reaching those prepared states from loose parts is unverified.
- Yoke predecessors use a small retainer-height search followed by 1.04 s of free settling. Every retained predecessor must pass unchanged placement and material checks.

These are initialization bounds. Native settling changes the saved poses. Masses, friction, damping, CAD, contact response, rewards and action limits are not randomized or changed by this curriculum.

## Acceptance and family meaning

Qualification restores through `BatchedAssemblyEnv.set_states` and executes bounded production actions with active finger squeeze and arm-target tracking. Those reference controls are receipts, not extra policy inputs. Zero incremental action need not maintain grip force.

Native fine-step penetration, velocity, thrown-part and clock guards remain enabled. Independent regular-part material checks use both meshes' vertices and triangle centroids and the table plane. Native and sampled material penetration must be ≤0.4 mm; predecessor keypoint errors must stay <1 mm. Held starts need loaded bilateral finger contact in at least four of the last five samples, translation slip ≤2 mm and rotation slip ≤0.1 rad.

Pickup begins on the table and replays for 1.28 s. In-hand starts must be airborne without external support. Near starts must remain clear of the receiving fixture. Held replay lasts 1 s. Partial starts additionally require task-specific engagement and alignment:

- Motor, horn and retainer: actual receiving contact and an unseated, aligned pose.
- Screw: geometrically engaged analytic threads, including pitch overlap, axis, radial clearance and helical phase. The selected partial subtype is **thread-supported with both hands free**, checked for ≤0.25 mm / 0.01 rad drift. It does not qualify screwdriver pickup or tightening.
- Yoke: **shallow fork entry**, with at least 1 mm of motor-envelope capture, lateral error ≤1.5 mm and angle ≤0.14 rad. A clearance-fit entry need not carry receiver force. Deeper insertion candidates remain unqualified.

CAD checks cover saved, midpoint and endpoint poses; native guards cover each physics step. The screw uses the native analytic geometry rather than the incomplete vendor mesh. These gates do not certify continuous mesh clearance, realistic hardware forces, real-kit calibration or a complete insertion path.

## Data and evaluation

`*_train.npz` adds qualified prepared states to the original loose TRAIN. Original `*_eval.npz` files remain byte-identical for full-tabletop evaluation. `family_eval/*_eval.npz` adds separately seeded prepared holdouts; report these separately from loose performance. The existing policy evaluator still selects the original loose files by default.

`*_provenance.json` records actual saved states, controls, noise, preparation and replay receipts. `qualification.json` records coverage and failures. Source copies and hashes bind the results to their implementation. A checkpoint with another pooled training-bank identity cannot resume against this bank.

Current local qualification results and remaining limits are recorded in `MEMORY.md`; generated datasets and private CAD remain outside source commits.
