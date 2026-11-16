# Assembly design audit

**Historical failed-design audit.** The current repair, successful guided checks and remaining free-drop failures are tracked in [mating repair](ASSEMBLY_MATING_REPAIR.md).
**Goal:** Check the original task, target and contact assumptions before treating this scene as a training-ready assembly problem.

Reviewed November 5, 2026 (America/Los_Angeles), against source `60aeb33` plus the diagnostic changes listed below. Physical profile remains **shared-side-v9-hoto-s-a01**, signature `99f8a29b33345a667b64a861ae38e9e4822f07ed13c37c6e92dc50605b9f8793`. Running Quest, canonical geometry/physics, rewards and reset banks are unchanged.

**Verdict: there are real design and validation errors. Full assembly is unqualified.** Correcting the screw benchmark removes one false failure; it does not repair the mating environment. Neither reward-valid initialized poses nor a strict software preflight demonstrate a reachable, realistic assembly path.

## What was checked

| Area | Evidence | Implication |
| --- | --- | --- |
| Original Hiwonder CAD → runtime | All six part origins and imported vertices match the pinned vendor `.3dm` exactly; object IDs and face counts agree. Millimetres convert to metres. | No detected import/scale error. Base and motor render meshes are not watertight; these are tessellations, not physical metrology. |
| Render transforms | Original part vertices and compiled visual mesh positions agree within 10.7 nm. Current regular SDF/SDF contact frames reconstruct within 5.7e−17 m. | No detected visual/collision frame mismatch in these checks. |
| Base/motor fit | Original source has 31.8 mm cavity versus approximately 32.0 mm motor thickness; sampled target interference 0.205 mm. The printable source has the same cavity. | A rigid, frictional model of interfering CAD requires justified compliance/tolerance. Real printing dimensions and force are unknown. |
| Horn/yoke target edits | `fastener.origins` raises both by 1.1 mm. Horn/motor overlap falls from 0.750 to 0.177 mm; motor/yoke overlap increases from zero to 0.651 mm. | Applying the horn correction to the yoke is a design error. Restoring yoke height alone fails the dynamic/whole-assembly checks; it is not adopted as a complete fix. |
| Mating friction/contact response | Motor seating succeeds only in the tested zero-friction counterfactual; coefficient 0.2 and current 0.7 cases fail. Single-interface and NoSlip-off ablations also fail. | Neither global zero friction nor deleting one contact/solver feature is a qualified repair. Material/contact calibration is missing. |
| Completion validation | All five initialized targets earn reward, but the retainer and final yoke scenes exceed the independent 0.4 mm sampled-material gate. | Native contact depth is not a bound on whole-mesh penetration. Existing reward tests prove their programmed gates, not physical fit. |
| Screw benchmark | Inherited motor rotation with an unrotated screw start caused 36.586 mm misalignment. Guides also selected 4/10 search instead of workcell free-body 16/20. | Both fixture errors are corrected. Original near-zero-turn results are invalid evidence of a bad workcell thread. |
| Screw geometry | Nominal M3×5 screw with 0.5 mm pitch and 60 µm radial simulation clearance supplements incomplete vendor threads. | Pitch, clearance and engagement are engineering assumptions; bench consistency does not identify the actual kit screw. |
| Tool/hardware | HOTO envelope/mass use published data; details/dynamics are estimated. Native button tests exist; robot pickup, OFF carrying, engagement, reaction torque and actual driving remain unqualified. | Guided thread success does not qualify the tool operation. Exact kit masses, friction, compliance and servo calibration remain unavailable. |
| Robot layout/control | Previous centered-layout checks reach 12/12 position-only endpoints. Each arm has five positioning joints plus a gripper. | This is not a collision-free grasp/path or arbitrary wrist-orientation guarantee. No new layout is adopted by this audit. |
| Learning | PPO/update/restore checks and initialized rewards exist. Current bank is 320 all-loose starts, no staged families; motor receives only 13 receiving-base context points. Later tasks require all predecessor placements. | These are full-prefix assembly episodes, not independent one-part lessons. Exploration and receiving-geometry coverage remain unqualified. SGS reweights existing entries; it does not invent missing stages. Separate pose previews do not change this bank. |

Import and printable evidence: [source fit audit](SO101_FIT.md). Hardware identity comes from the [official Hiwonder SO101 manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO_ARM101_Open_Source_6_Axis_Robotic_Arm_User_Manual.html); intended assembly forces/tolerances cannot be inferred from it. [MuJoCo contact guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip) recommends checking geometry/contact wrench before friction tuning. Its soft-contact/NoSlip behavior does not calibrate this kit.

## Completed poses: independent material check

These states already passed native reward/release/stability validation. Independent comparisons use both meshes' original vertices and face centroids, not generated contacts. The analytic nominal screw supplement is excluded from material comparisons; its contact/pitch tests are separate.

| Completed task | Native deepest contact | Maximum sampled material overlap | Material gate |
| --- | ---: | ---: | --- |
| Motor |0.060 mm |0.162 mm |Pass |
| Horn |0.056 mm |0.158 mm |Pass |
| Screw |0.061 mm |0.163 mm |Pass for regular predecessors; screw material not compared |
| Retainer |0.069 mm |0.568 mm, base/retainer |Fail |
| Yoke |0.129 mm |0.582 mm, base/retainer |Fail |

The deepest retainer/base samples are genuinely inside: exact winding is approximately 1 in both directions. At the retainer target, the native base field reports −0.562 mm at the sample where exact triangle distance is −0.568 mm. This is not a large grid interpolation or frame error. SDF native contact distance uses the maximum of the two fields at its optimizer point; it is not the deepest penetration of either surface. Do not convert the native 0.4 mm gate into a whole-material guarantee.

Restoring only the original yoke height still earns all five initialized rewards, but retainer/yoke completed scenes have maximum sampled material overlap 0.568/0.619 mm. Its tested horizontal approach also loses alignment and fails. A one-line target adjustment is insufficient.

## Actual motion, separate from target initialization

The fixture initializes the active part once outside its predecessors, then uses finite external forces/torques and native steps. The base is supported by a finite-force controller; predecessors remain free. The active part is released for 0.6 s. This is a contact-mechanics test, not robot execution or learned assembly.

All eight original pressed-path cases start clear and pass a free-space controller check. Their numerical states remain finite with zero warnings. At both 0.5 and 0.25 ms, **0/8 meet all gates**: cumulative keypoint error <1 mm, rotation <0.03 rad, release speed <0.01 m/s, native and sampled-material overlap <0.4 mm. Peak material is sampled every 100 ms, plus endpoint/native-peak poses; this is not continuous collision certification.

| Part | Force cap / terminal feed | Released active keypoint error, 0.5 / 0.25 ms | Peak sampled material, 0.5 / 0.25 ms |
| --- | ---: | ---: | ---: |
| Motor |20 N |12.417 / 12.528 mm |0.281 / 0.224 mm |
| Horn |2 N |2.183 / 2.191 mm |0.060 / 0.053 mm |
| Retainer |5 N |0.446 / 0.571 mm |0.558 / 0.609 mm |
| Yoke |5 N |13.326 / 13.367 mm |0.205 / 0.201 mm |

These are keypoint errors, not purely axial distances. A failed nominal path is not an impossibility proof.

Counterfactuals retain the original CAD:

- Motor, moving-part friction zero, 20 N: at 0.5 ms, released error **0.117 mm** and peak sampled material **0.209 mm**, passing this fixture's gates. At 0.25 ms it reaches **0.123 mm** error/ **0.349 mm** peak material but fails release speed (**10.614 mm/s**, limit 10 mm/s). Predecessor material settings stay unchanged. Zero friction is not a validated production repair. An earlier all-regular-part friction ablation gives the same 0.5 ms result.
- Motor, moving-part friction 0.2: released errors **12.386/12.371 mm** at 0.5/0.25 ms; both fail.
- Motor, volume-only interface: **12.534 mm** error; surface-only: **7.539 mm** and **0.524 mm** peak material. Deleting either interface is insufficient.
- Motor, NoSlip disabled: **12.416/12.401 mm** errors; both fail. NoSlip alone does not explain the blockage.
- Horn, moving-part friction zero while preserving motor/base friction: released error **0.585 mm**, but rotation **0.0549 rad** and peak material **0.613 mm**; fails.
- Retainer at only 1 N still reaches **0.504 mm** peak material; simply pushing less does not qualify that tested path.
- Horn at 5 N still has **2.200 mm** placement error. A vertical yoke candidate loses alignment; it is not evidence that all vertical approaches are impossible.

Removing terminal feed does not qualify the nominal paths: **0/8 position-guided cases pass**, for **0/16 total** pressed/position-guided cases. Released active errors at 0.5/0.25 ms are motor 12.447/12.526 mm, horn 2.187/2.152 mm, retainer 0.421/0.688 mm and yoke 13.385/13.495 mm. Retainer peak sampled material is 0.635/1.067 mm. All remain numerically finite with zero warnings. Receipts under `no-preload-small` and `no-preload-large` distinguish guidance from continued pressing; the retainer's material behavior is not timestep-qualified.

## Corrected thread fixture and software checks

The fixture is now motor-local, with aligned motor/screw axes regardless of workcell rotation, and uses the shared 16-start/20-iteration free-body search budget. At its original 0.002 Nm cap/60 rpm bench setting it turns **1.4665 revolutions**, advances **0.7220 mm** versus **0.7333 mm** expected: pitch error **0.0112 mm**, below 0.05 mm. A 1 N axial-only push advances **0.00108 mm**, below 0.1 mm. This is guided motion, not robot/tool driving.

Nine targeted regressions pass, including two rotated-workcell fixtures. **31/31 strict preflight checks pass**, with no task-mechanics exception. `configs/cloud_stress.json` now disables the obsolete exception. That preflight checks thread mechanics and software invariants; it does not run the complete material/path audit above. Historical reward/tool reports retain their original results; their screw-failure statements are superseded here.

## Required correction sequence

1. Repair and independently validate every mating target, including base/retainer overlap and the unjustified yoke lift. Check the full set of predecessors, not motor/base alone.
2. Qualify insertion/release with bounded forces, original geometry and appropriate contact/compliance. Keep timestep, material-overlap and grasp retention checks together; do not select frictionless mating because it happens to seat.
3. Demonstrate feasible two-arm grasps, assembly reorientation, actual screwdriver/button operation and retained predecessors.
4. Qualify independently trainable stages and receiving-geometry observations before scaling learning. Existing pose-review candidates remain separate from production SGS.

Hardware dimensions/force/compliance measurements are needed to establish real-kit fidelity. There is no verified universal friction value available from the video/CAD. No overnight job is launched: mating remains unqualified and the local disk is below the supervisor's 2 GiB floor.

## Reproduction and receipts

```bash
PYTHONPATH="$PWD" MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_so101_sliding --thread
PYTHONPATH="$PWD" MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_so101_sliding --targets-only \
  --out artifacts_local/design-audit/canonical-targets
```

The auditor reports failed cases rather than changing the scene to make them pass. Diagnostic flags `--part-friction`, `--motor-interface`, `--noslip`, `--target-correction`, `--feed-force`, `--extraction` and `--force-cap` are explicit overrides; reports bind base identity, script hash and override values. Do not label such a variant as the unmodified v9 environment.

Ignored receipts: `artifacts_local/sliding-check/report.json` (eight pressed paths and corrected thread), `strict-preflight.json`, archived `default-checker-used.py`; `artifacts_local/design-audit/geometry.json`, `grid_sweeps.json`, `contact_frames.json`, `canonical-targets/report.json`, individual ablation reports and `regressions.log`. Source/import and grid sweeps are read-only local prototypes; original vendor CAD stays outside source control. Dense nominal-path sweeps found at most 0.0251 mm false grid-inside depth on the motor path and 0.00155 mm on the horn path. They do not establish continuous clearance.

## Files changed

| Source file | Modification |
| --- | --- |
|`scripts/check_so101_sliding.py` |Force-driven/release and independent all-predecessor target diagnostics, explicit counterfactuals and failure receipts. |
|`scripts/so101_thread_check.py` |Correct motor-local alignment and free-body collision-search budget. |
|`tests/test_thread_bench_frame.py` |Reject rotation-dependent misalignment and downgraded bench search. |
|`configs/cloud_stress.json` |Require strict preflight instead of the obsolete screw exception. |
|`docs/ASSEMBLY_DESIGN_AUDIT.md` |Record design errors, controls, counterfactuals and limits. |
|`docs/CURRENT_ENVIRONMENT.md` |Point current qualification status to this audit. |
|`docs/OVERNIGHT_READINESS.md` |Supersede the erroneous screw-bench conclusion; retain assembly/resource blockers. |
|`User Intent.md` |Record the broader design audit and corrected qualification boundary, preserving concurrent pose-preview intent. |
|`MEMORY.md` |Record fixture corrections and rejected physical shortcuts, preserving concurrent entries. |

Intentionally unchanged: canonical `scene.py`, `workcell.py`, `fastener.py`, physical JSON, visual/source CAD, rewards, observations, learner, production banks, active Quest service and other chat's preview code. Diagnostic receipts/prototype utilities are ignored, not published runtime changes.
