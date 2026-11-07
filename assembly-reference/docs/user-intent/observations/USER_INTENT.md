# Policy inputs and actions

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Keep the selected **448-point** input: 128 active-part points, 256 receiving-assembly points and 64 distal-finger points for the current WowRobo assembly. The older screw/tool observation path is historical and is not the current viewer input. The optional 128-point mating budget remains a comparison, not the selected default.

History is five frames of 53 values: **265 proprioceptive/history values**. Actions are **12 continuous relative targets**: five joints and one gripper for each arm, at 25 Hz. Native physics currently steps at 0.5 ms.

Points move coherently with physical objects. The current path has no independent point jitter, missing-point simulation, CAD-shape randomization or mass/friction randomization. Do not confuse pose variation with sensor noise. Geometry is simulator-derived; a deployed perception pipeline is not implemented.

The local reset explorer at localhost:8876 now has a Point cloud section for the selected reference pose or variation. Its current-assembly shared builder is `assembly-so101/assembly_policy/so101/wowrobo_obs.py`; viewer export calls that exact encoder. It uses current source visual meshes and the current four finger bodies, preserves articulated-case sample attachment and ordering, and displays all float32 values in the left fixed-finger body frame scaled by ten. JSON includes the flat tensor, per-point body identity and hashes. No WowRobo policy is attached yet; the section states that explicitly. Setup has no invented active-task cloud. See [implementation and verification](../../WOWROBO_POINT_CLOUD.md).

**Current state:** The 448-point selection and compatibility signatures are implemented and preserved. Existing core/preflight checks cover shape, finite values and task/geometry wiring. Changed layouts must not silently reuse incompatible checkpoints.

**Open coverage issue (November 7):** The current pin receiver budget is157base/33cover/66connector. At15_pin_near, no receiver points fall within a2mm-radius,±2mm-axial neighborhood of the registered hole mouth; nearest sample4.415mm. User asks whether task-local geometry should replace broad predecessor coverage. Recommended for review: fixed task-specific dense mating-feature surfaces plus nearby interaction/collision context and sparse assembly context. Selecting only currently touching surfaces would omit the destination before contact. No observation-contract change has been made from this discussion. Evidence: `outputs/point-cloud-viewer-20261107/pin-coverage-review.json`.

**For your review:** Confirm these inputs and action semantics. Zero incremental action does not maintain gripper squeeze; state qualification must use the actual closing-action contract.

See [input specification](../../SO101_MATING_INPUTS.md) and [visualization](../../visualizations/so101-mating-budget.html).
