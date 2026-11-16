# WowRobo photo-directed reset sequence

This specification records the user’s photo-based reset requirements. The 17 assembly references are IMG_1624, IMG_1623, IMG_1622, IMG_1621, IMG_1620, IMG_1618, IMG_1617, IMG_1616, IMG_1615, IMG_1614, IMG_1613, IMG_1612, IMG_1611, IMG_1610, IMG_1609, IMG_1608 and IMG_1607. They establish qualitative poses, visible grip locations and the role of both arms. Exact depths, insertion fractions, force, loads and completion are not measured from them.

## Scope and current status

The active collection contains25 review entries with their original IDs. Per user correction,2,6,11,16,19,24,27and32 are removed. This includes all five visible seated completion references. Their assembled poses remain internal prerequisites for later steps, and frozen v2 remains available as provenance. Every active entry is an unqualified candidate, not evidence of physical training admission.

`assembly_policy/so101/wowrobo_photo_resets.py` authors native snapshots for the existing Three.js/MuJoCo viewer. The ongoing implementation registers the actual finger frame and asymmetric gap, refines grasps against native meshes, and derives supported insertion poses from receiver contact. The common positioned CAD assembly is a source reference; it does not by itself define the final folded/resting poses.

**The review model remains unqualified for training.** Gravity and collision stepping remain disabled in this CAD-review scene. Geometric checks and two-finger proximity are distinct from loaded grasp retention, frictional contact, valid insertion dynamics or executed rewards. The actual pin hole and camera calibration remain unmeasured. Each current export needs its own model/state/control identities and qualification receipts.

## Required grasps and arm roles

| Step | Active arm and actual grip | Other arm | Primary references |
|---|---|---|---|
| Side cover | LEFT closes on the two opposed exterior side walls near the flange end | RIGHT waits open nearby | IMG_1622, IMG_1621, IMG_1620 |
| Shoulder connector | RIGHT closes on the opposed socket side faces with fingers parallel to the table and horizontal push | LEFT stays open in a natural retreat from cover placement | IMG_1618, IMG_1617, IMG_1616 |
| Pin | LEFT pinches the larger upper cylinder; the thin shaft stays clear and points down for insertion | RIGHT keeps holding the connector | IMG_1615, IMG_1614, IMG_1613 |
| Upper arm | RIGHT pinches the exposed black servo 2 case | LEFT holds the base/connector assembly throughout | IMG_1612, IMG_1611, IMG_1610 |
| Final module | LEFT pinches the exposed black servo 3 case with the corrected wrist orientation | RIGHT holds the installed upper-arm piece throughout | IMG_1609, IMG_1608, IMG_1607 |

In IMG_1612 one finger passes into the link’s open window and the other reaches the opposite exposed black case face. States 18 and 19 must retain the approved component orientation and use that case grasp. Moving the grip onto the printed yoke is not an acceptable substitute. Both fingers must surround the case; a pose with one finger inside the solid and the other away from it is rejected.

The final-module wrist must match IMG_1609; the user requested a 90° correction from the rejected wrist pose. The reference specifies the physical black case faces to pinch. The approved loose-part orientation and its placement reference remain separate from the actual finger-closure normal.

Frozen v3 review snapshots retain the original two-cylinder pin: a28mm ×Ø12mm upper grip and a22mm ×Ø2.7mm shaft. The user subsequently authorized a fit-first physical pin: unchangedØ12×28mm grip, Ø2.3×14.5mm shaft, total42.5mm; see [pin fit evidence](WOWROBO_PIN_FIT.md). The new physical overlay uses this version without rewriting frozen v3. IMG_1615/1614 constrain the pickup/insertion action and hand roles; they do not measure or certify that custom geometry. Grip only the larger cylinder.

## Supported insertion geometry

**States21–23:** The opposite non-servo end of the upper-arm module rests on the actual edge of the base/connector receiver, forming the triangle visible in IMG_1611. The held servo case points straight down into its receiving pocket. As the servo end descends through partial, near and seated references, the opposite end remains supported. The part is not simply suspended above the receiver.

**States29–31:** Apply the same two-end relationship to the final module. The held servo aligns with the pocket in the installed upper arm, and the other end rests on that upper-arm receiver as in IMG_1608. Because the installed upper arm is tilted, correct alignment is relative to that receiver frame; it need not be world vertical. Include the entire far end, including its hanging servo case and brackets, when determining first contact and checking interference.

Fold angles are consequences of the mating and resting contacts. The earlier fixed 25°/9°/1.5° rocking angles and a universal common-frame seated pose are superseded. Exact fractions such as “halfway” remain authored labels until the slot and insertion travel are registered.

## Support continuity

The v3 connector correction keeps the finger-length axis within0.03–0.50° of horizontal in07–11 and retains the seated grasp through12–17. The small planar yaw (about4–5.4° relative to the component axis) centers the asymmetric native jaw gap with five arm DOFs. It does not reintroduce the previous60–75° downward pitch. Both fingers remain within1mm of the actual opposed outer socket walls, including their rounded surfaces. Additional opening margins from the previous angled grip are removed.

The v3 RIGHT fork support is slightly tighter: native pad gaps are about0.150/0.160mm, compared with0.697/0.665mm in v2. The jaw closes0.01491rad and the hand shifts0.553mm while preserving its wrist branch. This one reviewed pose is retained through the remaining states25–31. Native contact authoring is preserved directly because the coarse voxel clearance objective would reopen the visible gaps; a receiver-pose identity guard rejects use after the receiver moves. Contact forces remain unqualified.

- RIGHT maintains its connector grasp through the pin sequence.
- After pin insertion, LEFT acquires the base/connector hold shown in IMG_1612 before RIGHT releases the connector to fetch the upper arm.
- LEFT keeps supporting that assembly throughout upper pickup, transport, edge placement and insertion.
- RIGHT supports the installed upper-arm piece before LEFT releases its prior hold to fetch the final module, as in IMG_1609.
- RIGHT maintains that support through final pickup, transport and insertion.

Both arms remain policy-controlled after reset. Supporting grasps are part of the initial snapshots to validate, not scripted constraints during rollout. Any required grip relocation must use an explicit supported transition; a silently teleported or unsupported handoff is insufficient. The user’s latest requirement supersedes the historical unused-arm-open rule for these WowRobo phases.

States20and28 include an authored8° downward link sag about the held servo shaft. The servo case and both helpers retain their prior world poses; the far end drops about16.72/16.27mm. This is a visual gravity-like reset pose, not a simulated equilibrium. State27 is removed.

## Stage inventory

| State | Phase | Left arm | Right arm | Photo reference |
|---|---|---|---|---|
| 00_loose | loose | nearby_open | nearby_open | IMG_1624, IMG_1623 |
| 01_cover_pickup | pickup | grasp_active | nearby_open | IMG_1622 |
| 03_cover_aligned | aligned | grasp_active | nearby_open | IMG_1622 |
| 04_cover_entry | entry | grasp_active | nearby_open | IMG_1621 |
| 05_cover_near | near_seated | grasp_active | nearby_open | Inferred intermediate |
| 07_connector_pickup | pickup | nearby_open | grasp_active | IMG_1618 |
| 08_connector_aligned | aligned | nearby_open | grasp_active | IMG_1617 |
| 09_connector_entry | entry | nearby_open | grasp_active | IMG_1616 |
| 10_connector_near | near_seated | nearby_open | grasp_active | Inferred intermediate |
| 12_pin_pickup | pickup | grasp_active | hold_connector | IMG_1615 |
| 13_pin_aligned | aligned | grasp_active | hold_connector | IMG_1614 |
| 14_pin_half | half_inserted | grasp_active | hold_connector | IMG_1613 |
| 15_pin_near | near_seated | grasp_active | hold_connector | Inferred intermediate |
| 17_left_support | support_transition | hold_base_assembly | hold_connector | IMG_1613, IMG_1612 |
| 18_upper_floor_grasp | floor_grasp | hold_base_assembly | grasp_active | IMG_1612 |
| 20_upper_transport | transport | hold_base_assembly | grasp_active | Inferred intermediate |
| 21_upper_far_side | far_side_supported | hold_base_assembly | grasp_active | IMG_1611 |
| 22_upper_half | half_inserted | hold_base_assembly | grasp_active | IMG_1610 |
| 23_upper_near | near_seated | hold_base_assembly | grasp_active | Inferred intermediate |
| 25_right_support | support_transition | hold_base_assembly | hold_upper_arm | IMG_1610, IMG_1609 |
| 26_final_floor_grasp | floor_grasp | grasp_active | hold_upper_arm | IMG_1609 |
| 28_final_transport | transport | grasp_active | hold_upper_arm | Inferred intermediate |
| 29_final_far_side | far_side_supported | grasp_active | hold_upper_arm | IMG_1608 |
| 30_final_half | half_inserted | grasp_active | hold_upper_arm | IMG_1607 |
| 31_final_near | near_seated | grasp_active | hold_upper_arm | Inferred intermediate |

Photo references are qualitative: IMG_1613 shows pin insertion; IMG_1610 and IMG_1607 show engaged modules. They do not establish an exact half-depth or prove the final reward. The two support-transition snapshots also represent an authored continuity requirement rather than a calibrated reconstruction of a single exposure.

## Authored offsets and completion

The current cover, connector and pin stand-offs are authoring parameters: cover alignment 55 mm, entry 20 mm and near 2 mm; connector alignment 65 mm, entry 25 mm and near 2 mm; pin alignment 32 mm, halfway 11 mm of shaft travel remaining and near 2 mm. These are not image measurements and do not certify engagement. Upper/final tilt and seating must follow their registered support and receiver frames.

Entry, partial and near-seated candidates must not already earn completion. Fully seated poses remain internal completion anchors; those entries are removed from the visible bank. The generated review snapshots do not execute or validate a new reward function.

## Geometric review and physical admission

For every changed state, inspect the relevant photo, both opposed finger contacts, palm and wrist-camera clearance, and interference with the rest of the assembly. State 25 requires particular attention: holding the installed piece must leave the final insertion corridor clear. Check the whole incoming far end rather than only its printed bracket. A small IK center error alone does not establish a correct grip.

Coarse SDF refinement and sampled mesh-boundary checks are geometric authoring tools. They need explicit coverage/resolution and rejected-state receipts; they do not prove continuous collision freedom. Earlier IK residuals and support-joint drift figures came from the superseded grasp model and must not be reused as validation of this revision.

The revised photo audit still identifies differences requiring review: state18's printed link sits about23mm above the table at its current internal servo angle; the LEFT base support is lower than the visible cover-wall hold in IMG1612/1611; the clear RIGHT fork support uses an inverted wrist, unlike the pictured roll. These differences are shown in the viewer's PHOTO REVIEW field. The default protruding servo4 orientation is retained because IMG1609/1608 support it. Sampled checks also reject several pin/receiver and near-seated CAD fits, independently of finger clearance. No reset is admitted for training.

The RIGHT support's coarse refiner can report a false contact failure because its80-point pad subset misses the closest patch. An independent check of all native inner-face vertices/triangle centers found opposed gaps about0.697/0.665mm in the exported authoring pose. Preserve both reports; do not silently relabel the coarse check as passing. Supporting poses stay continuous across their phases even when a check rejects them.

Before training admission:

1. Register the receiving pockets, slide axes, resting features and actual pin hole.
2. Verify opposed contacts on the intended case, cover sides, connector sides or large pin cylinder, and reject crossings of fingers, palm or camera through solids. Preserve the approved object orientation in states 18/19.
3. Verify the far-end resting contact and receiver-relative servo alignment throughout21–23 and29–31; preserve assembly support during the two handoffs.
4. Validate loaded grasp retention, receiver stability and current per-state geometry identities under the intended physical engine.
5. Check the actual nonterminal/terminal reward gates, bounded-action rollouts and fine-step insertion dynamics.
6. Only then randomize coherently around qualified families and publish disjoint training/evaluation reset caches.

## Rebuild and inspect

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_photo_resets.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/render_wowrobo_photo_resets.py
```

The canonical contract is `configs/wowrobo_reset_sequence.json`. Generated local review exports contain native `scene.xml`, stage archives, `stage-receipts.json`, `reset-contract.json` and PNG previews. Inspect the export’s own manifest and receipts rather than treating an older render or bank identity as the current result.
