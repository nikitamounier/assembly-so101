# Understanding the policy input

**Goal:** Show what the dual SO101 policy receives using an actual frozen production snapshot.

**Current implementation:** the user requested and received [part/mating observations](SO101_MATING_INPUTS.md). The new SO101 view uses the five-task production sampler with 128 active, 256 combined mating/tool and 64 distal-finger points, with a128-combined-mating option. No extra or padded tool slots. The older v6 visualization described below is historical and does not represent current sampling.

The November 5 visualization uses a frozen, read-only v6 snapshot of the Mac simulator. The later v8 scene removes the unused arm link and returns to five tasks; this historical visualization still shows its original six-task data. Changing its selected task changes the sampling table in that frozen scene. It does not execute a task or advance Quest.

The November 6 [paper-based sampling review](OBSERVATION_SAMPLING_REVIEW.md) verifies current v8 base/gripper transforms, proposes a 512-point interface-aware comparison and explains state-preserving runtime task switching. Those sampling and sequencing changes are proposed, not implemented in the historical view or production policy.

A separate [standard NIST/Franka view](NIST_POINT_VIEW.md) now shows all 20 upstream tasks with the original 128/128/64 samples, standard single Franka arm, board geometry and original seeded loose resets. It is independent of the SO101 view and the proposed sampling changes.

## What the points mean

| Samples | Meaning |
| --- | --- |
| 128 | Surface locations on the active task part, such as the motor. |
| 128 | Context locations shared across the other observed parts and screwdriver bodies. |
| 64 | 16 locations on each left/right palm/fixed-finger and moving-finger body. |

Each sample becomes three coordinates, so geometry supplies 960 numbers. Samples are area-weighted from modeled surfaces, generated once with the production default seed 0 and reused in a fixed order as bodies move. The sampler includes hidden/back surfaces. This is simulator geometry, not RGB pixels or a depth-camera point cloud. It excludes the table, clamps and full arm links.

In the frozen motor-task snapshot, the base gets only 12 of the 128 context samples. Other context bodies each receive 11 or 12, including separate tool sub-bodies. The later legacy five-task layout gives the base 13 samples. That historical layout does not guarantee useful cavity/rim coverage. The selected combined mating layout below supersedes uniform context allocation; the frozen visualization continues to show its original data.

The active part is a task definition selected by software. During training, the reset sampler chooses a reset-bank entry whose task ID is passed to the environment; `PointTables` maps the corresponding task name to the active body. The policy does not independently choose which object gets 128 points. The visualization's dropdown chooses that same table for inspection without controlling an episode or advancing the physical scene. See [task/reset selection](../assembly_policy/train.py) and [environment reset](../assembly_policy/rl/vecenv.py).

The input coordinates use the **left gripper body frame**, with metres multiplied by 10, so the displayed input values are decimetres. This body origin differs from the `left_gripperframe` tool site used for proprioception. Colors, marker shapes, axes and simplified gray reference meshes help the viewer; none are extra point-input channels.

The historical 320-point cloud is flattened in its fixed order into a 960-value MLP input. It is not an unordered PointNet encoder. The task identity chooses the point table; it is not separately concatenated as a task one-hot vector.

## Current receiver-focused layout

The merged training snapshot uses the selected **448-point** observation: 128 active, 256 combined predecessor assembly/tool, and 64 distal-finger points. Screw splits its mating budget equally between assembly and tool. Other tasks use the full mating budget for the base and all predecessors. The earlier `mating_v1` weighted-context proposal is superseded. See [allocation and checkpoint compatibility](SO101_MATING_INPUTS.md).

## Other inputs

The policy also receives five frames of 53 robot-state values, totaling 265. Each frame contains two tool poses/twists (26), both arms' joint positions (12), previous action (12) and driver button/spindle state (3). The visualization shows this schema, not an invented action history from teleoperation. Only the current geometry cloud is supplied, rather than five clouds.

Sources: [production sampler](../assembly_policy/so101/obs.py), [state construction](../assembly_policy/so101/env.py), [network](../assembly_policy/rl/nets.py).

## How placement is learned

For the motor task, the correct motor position and orientation relative to the base are defined by the assembled CAD offsets. `goals.placement_errors` transforms that reference with the current base pose and compares actual versus target keypoints. The reference target is used by the training judge; no explicit goal pose or ghost target point cloud is currently fed into the policy.

During training, the policy observes current points and robot-state history, samples 12 arm/gripper commands, and the simulator executes them. The environment scores the resulting state. PPO uses observed returns and a learned value estimate to update the network toward actions associated with better returns. Thus the intended target relationship must be learned in the weights from experience for each task; it is not deduced automatically from the CAD or selected by the robot.

The positive task signal is sparse: valid placement below 1 mm keypoint error, release and about 0.4 seconds of stable retention, with additional screw contact checks. Motion/effort costs and configured safety handling also affect returns. Placement distance is measured but is not presently a continuous closer-is-better reward. Discovering valid success from all-loose starts is consequently difficult; the learning mechanism does not establish successful assembly by the current policy. See [placement target](../assembly_policy/so101/goals.py), [reward](../assembly_policy/so101/env.py) and [rollout/PPO update](../assembly_policy/train.py).

## Evidence and boundaries

- Snapshot captured November 5, 2026 at 22:36:53 America/New_York, profile `shared-side-v6-arm-link`.
- Model signature `642239a3bf93d4b1e7056d3ab1283d2817e518ac7422039c7fdb0dd141d15dc6`; configuration SHA256 `ebfca66ca900c636a50487fea6dbe2d6fb02851115ac1b232986a6d93a96217b`.
- All six tasks use the actual `PointTables` output. Independently reconstructed coordinates match all six production tensors exactly before presentation rounding. Maximum coordinate rounding difference is below 0.000001 dm.
- The actor and critic are instantiated with 320 points, 265 proprioception values and 12 actions. Actor including exploration parameters: 1,435,544; critic: 1,432,705; total: 2,868,249. Adding the sixth task does not change those network dimensions or parameter counts.
- CPU export uses isolated PyTorch 2.14.1 and MuJoCo 3.17.0. It does not qualify a training run or modify the teleoperation environment's dependencies.
- Gray reference geometry is simplified to fit the response. All 320 sampled coordinates per task are preserved apart from six-decimal presentation rounding.

Local receipts and reproducible export/preview scripts are under ignored `artifacts_local/observation-viz/`. `browser-verification.json` records successful task selection across all six tasks, exact displayed-coordinate checks, group filtering, frame selection, orbit and mesh toggling. Light/dark layouts pass at 736, 360 and 320 px without horizontal overflow or JavaScript exceptions; native screenshots were inspected. Physics, controller mappings, running services and training jobs are unchanged by this explanation.

## Files changed and follow-up

- Conversation visualization `policy-input-points.html`: actual six-task sample data, orbit view, group visibility, reference meshes and selected XYZ inspection.
- `docs/POLICY_INPUT_VIEW.md`: input interpretation, source links and verification.
- `User Intent.md`: explicit input-interpretability requirement and remeasured six-task architecture counts.
- `MEMORY.md`: dated explanation/verification decision and session status.
- `TASK.md`: observation-visualization progress, preserving the separate task-target work.
- Ignored `artifacts_local/observation-viz/`: export, mesh reduction, preview checks and receipts.

Follow-up: no environment change is needed for the visualization. Real camera perception and physical motor seating remain separate unresolved work.
