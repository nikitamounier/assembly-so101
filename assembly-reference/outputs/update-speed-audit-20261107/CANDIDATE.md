# Surface-tree candidate, November 7

Authorized first physics-preserving optimization from the update-speed audit.

The candidate changes `assembly-so101/plugin/so101_surface_tree.h` only: compose the visual-to-target transform once per pair; calculate transformed node bounds using center/extent and the absolute relative rotation; retain conservative roundoff padding; reuse per-thread traversal and candidate buffers. Exact leaf arithmetic, ascending sample order, contact bins, SDF values, solver, timestep, controls and geometry are unchanged.

The relative rotation is rigid. Its absolute matrix times each node's half-extents bounds every transformed corner. The padding is `1e-12 + 256*epsilon*scale`, where scale includes absolute node endpoints and both world positions; it covers rounding in composition and the original uncomposed sample transforms. This bounds-only arithmetic does not determine contact depth or normals. Thread-local scratch assumes the current grid-distance callbacks do not recursively invoke collision detection.

## Frozen identity and isolation

Model: `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`.

The six native build inputs were copied from the active frozen source into separate baseline and candidate folders. Build receipts verify that only the surface-tree header differs. Baseline header SHA256: `3df103be0e3b8cfde509a00da787ad753fc40cd4686db9a4b45163be70e88d8f`. Candidate and library hashes are in `candidate-build-receipt.json`.

Tests run in fresh processes with distinct native build directories under the frozen root's `tree-optimization-fork-20261107`. The harness preloads the canonical builder with only SOURCE/build-directory overrides. Both sides consume the same frozen profile, model, grid and reset bank. CPU-only Slurm steps use retained lease3598717, 2 CPUs, 12G memory, single-world rollout on the caller thread. No live source, job, policy, monitor, model, hyperparameter, or cloud app changed. These are isolated diagnostic probes, not a production iteration benchmark or proof that the reset bank is physically qualified.

## Initial three-interval replay

Eight task-conditioned cases, three40ms intervals each, 160 native substeps/interval. First action is zero, subsequent deterministic actions move both arms and grippers. All432 captured arrays are exactly identical: every substep's integration trajectory and sensors; final qpos/qvel/warmstart; contacts including depth, position, frame, geometry and friction; contact forces; observations, reward, done and every info field. Passive coordinates are included in full-state trajectories.

Mean measured times (seconds, profiling counters enabled on both sides):

| Case | Baseline | Candidate |
|---|---:|---:|
| Loose, across five tasks | 1.355 | 1.269 |
| Cover near | 3.136 | 2.966 |
| Upper far-side prototype | 8.432 | 7.813 |
| Upper far-side friction variation | 8.381 | 7.978 |

Aggregate paired speedup1.06457x (6.065% less wall time). Supplement timing improves about7–10%; ordinary SDF search is unchanged. Node/sample/rejection counts match in this panel. Full captured results are `*-quick-mode1.{json,npz}` and `quick-comparison.json`.

All seven existing `assembly-so101/tests/test_surface_acceleration.py` tests passed in18.49s, including exhaustive legacy parity, contact perturbations, shared-model parallelism and grid-bound rejection checks. C++ libraries compile with the existing `-O2 -Wall -Werror` build. Probe helpers pass Ruff.

## Broader qualification

Uninstrumented full-panel replay passed:30 task-conditioned cases (all25 unique prototypes, duplicated loose poses across tasks, and one held-state friction variation), one40ms/160-substep interval each. All540 arrays are byte-identical. Baseline166.890151s; candidate158.556128s; speedup1.052562×, or4.9949% less wall time. Every measured case is faster, ranging1.0143–1.0741×. This is one paired full-panel pass, not a statistical confidence interval. See `all-comparison.json` and `*-all-mode1-unprofiled.{json,npz}`.

Across this panel and the three-interval replay,972 arrays are byte-identical over54 control intervals/8640 physics substeps. These checks preserve the same admitted-bank behavior; they do not independently validate whether those initial states satisfy every physical training requirement.

The exhaustive oracle also passed:87 cases,435 byte-identical arrays, up to243 contacts/pose. It compares exhaustive mode2 on one thread against candidate mode1 on two simultaneous threads sharing an immutable model. Cases include29 task-conditioned prototype rows, each unmodified and with two deterministic free-body translation/quaternion perturbations. Comparisons cover contact depths/positions/frames/geometry/friction, contact forces, acceleration, constraint forces and sensors. See `oracle_tree.py` and `oracle-parity.json`.

The candidate passes these isolated checks. Production worker throughput and a whole-iteration benchmark remain prerequisites for claiming a live-training benefit. The parent run remains frozen; deployment must be a separately identified release/continuation, not hot-loading this header into running workers.

## Reproduction

The launchers use the recorded target.json and retained allocation3598717; they must be reviewed if either has changed. All native compilation/probing is inside allocated Slurm steps. No new lease is submitted.

```sh
python3 outputs/update-speed-audit-20261107/run_tree_probe.py quick
python3 outputs/update-speed-audit-20261107/run_tree_checks.py
cd ../assembly-so101
.venv/bin/python -m pytest tests/test_surface_acceleration.py -q
```

Start the commands above in the canonical robotics-assembly-3.0 repository; the regression command then switches to its sibling assembly-so101 checkout.

