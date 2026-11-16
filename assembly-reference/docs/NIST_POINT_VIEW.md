# Standard AssembleNIST point view

**Goal:** Visualize the original NIST board and Franka Panda inputs without the SO101 adapter or proposed sampling changes.

The conversation view covers all 20 original tasks. It uses `assembly_policy.rl.obs.PointTables` with its default seed 0 and 128 part, 128 mating-geometry and 64 finger points. The sampler matches the upstream source previously downloaded from `suniique/assembly-nist`. Each task uses the original `ResetGenerator.loose()` at seed 0 and the original `build_xml(..., visual=True)` geometry. The standard setup has one Franka Panda arm, all static board fixtures and one selected movable part.

Reference CAD is simplified for display. Colored marks retain the actual sampled coordinates, quantized for the conversation payload with maximum measured rounding below 0.000006 dm. Numeric XYZ is in the original TCP frame, scaled from metres to decimetres. The full arm and board are reference geometry, not extra policy point channels. The finger close-up also draws the native finger collision shapes used by the sampler. These are reset snapshots, not policy executions or successful assembly demonstrations.

The view provides task selection, orbit/zoom, workcell/board/pair/finger views, point-group visibility, optional reference CAD and selected-point XYZ inspection. It preserves the standard point counts and sampling locations. No SO101 or Quest runtime, physics, weights, reset bank or sampling configuration is modified.

## Verification

- Exported 6,400 points across 20 tasks using the production upstream sampler.
- Independently transformed active-part and mating samples agree with native world geometry within 0.0002 mm; finger samples lie on native collision surfaces within 0.000007 mm.
- Native TCP and the upstream production `FrankaFK` agree across all 20 poses: maximum position difference below 0.000000000001 m and rotation-entry difference below 0.000000000001.
- Browser checks pass all 20 task-coordinate selections, four views, finite projections, point inspection, orbit, group visibility and reference toggling.
- Light/dark layouts pass at 736/360/320 px without horizontal overflow or JavaScript exceptions. Workcell, finger close-up and narrow/theme screenshots were visually inspected.
- Source hashes, native checks and browser receipts are retained under ignored `artifacts_local/nist-point-viz/`. The final fragment is below 1 MB.

## Files changed and follow-up

- Conversation `nist-franka-points.html`: interactive standard Franka/NIST scene and actual point samples.
- `docs/NIST_POINT_VIEW.md`: sources, interpretation and verification.
- `docs/POLICY_INPUT_VIEW.md`: link the separate original-setup comparison.
- `User Intent.md`: record the request for an unchanged standard setup.
- `TASK.md`: record completed visualization work while preserving the existing handoff.
- `MEMORY.md`: session findings and handoff.
- Ignored `artifacts_local/nist-point-viz/`: exporter, embedded dataset, native material reference, preview checks and receipts.

Follow-up: none required for this view. Proposed SO101 sampling changes remain separate and unimplemented.
