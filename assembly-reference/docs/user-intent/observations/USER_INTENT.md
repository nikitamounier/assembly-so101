# Policy inputs and actions

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Keep the selected **448-point** input: 128 active-part points, 256 receiving-assembly points and 64 distal-finger points for the current WowRobo assembly. The older screw/tool observation path is historical and is not the current viewer input. The optional 128-point mating budget remains a comparison, not the selected default.

History is five frames of 53 values: **265 proprioceptive/history values**. Actions are **12 continuous relative targets**: five joints and one gripper for each arm, at 25 Hz. Native physics currently steps at 0.5 ms.

Points move coherently with physical objects. The current path has no independent point jitter, missing-point simulation, CAD-shape randomization or mass/friction randomization. Do not confuse pose variation with sensor noise. Geometry is simulator-derived; a deployed perception pipeline is not implemented.

The local reset explorer at localhost:8876 now has a Point cloud section for the selected reference pose or variation. Its current-assembly shared builder is `assembly-so101/assembly_policy/so101/wowrobo_obs.py`; viewer export calls that exact encoder. It uses current source visual meshes and the current four finger bodies, preserves articulated-case sample attachment and ordering, and displays all float32 values in the left fixed-finger body frame scaled by ten. JSON includes the flat tensor, per-point body identity and hashes. No WowRobo policy is attached yet; the section states that explicitly. Setup has no invented active-task cloud. See [implementation and verification](../../WOWROBO_POINT_CLOUD.md).

**Current state:** The 448-point selection and compatibility signatures are implemented and preserved. Existing core/preflight checks cover shape, finite values and task/geometry wiring. Changed layouts must not silently reuse incompatible checkpoints.

**Task-specific refinement (November 7):** User authorized implementing and iterating coverage. After32/48/64/96-point feature sweeps, selected48receiver-feature points for pin/connector,64for cover/upper/final;64coarse installed-assembly context, remainder nearby. Active128 reserves32mating-surface points; pin8tip+24shaft/shoulder. Pin receiving feature48=16rim+24bore+8blindstop. Source points cover all4rim quadrants,8within2mm of mouth versus0before; nearest1.735mm versus4.415mm. All100states checked (96task/4setup),384unique active/receiver points per task, native source-surface error<8nm after float32 storage. Source/model/frame bindings and observation signature updated. These are CAD-coverage checks; learning-performance comparisons remain future work. Evidence: `outputs/point-cloud-refinement-20261107/README.md`.

**For your review:** Confirm these inputs and action semantics. Zero incremental action does not maintain gripper squeeze; state qualification must use the actual closing-action contract.

See [input specification](../../SO101_MATING_INPUTS.md) and [visualization](../../visualizations/so101-mating-budget.html).
