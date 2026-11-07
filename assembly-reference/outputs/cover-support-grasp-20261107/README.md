# Installed cover support grasp — 2026-11-07

Candidate only. No existing source, CAD, bank or configuration was modified by this search. Parent integrates the versioned state/receipt changes.

The previous support role used `base_servo` and a low25mm housing grip. IMG1611 shows LEFT holding the installed base side cover while RIGHT handles the upper arm. The corrected pinch contacts the cover's opposed exterior side walls, with actual contact centroid42.91mm above the table.

## Final candidate

- `best.json`: six native LEFT coordinates, explicit `support_grasp` override, geometric calibration, actual contact evidence and per-stage native audits.
- `calibration-details.json`: optional measured contact-centroid calibration and proof that cover/finger world poses are exactly identical in all8affected states.
- `corrected-<stage>.npz`: full qpos/controls with only LEFT changed.
- `verification.json`: full sampled native audits for17,18,20,21,22,23,private24,25.

LEFT q = `[-0.062401900890025355,0.5324429639402127,1.3114642050648837,-1.6580627494556925,-1.5485379508666797,0.3998793946252]`.

Initial valid cover section is near localY58mm/Z41mm, across the wide U-shaped cover rather than a narrow rim/hole. Holding the five arm joints fixed and closing only the jaw by0.0115rad gives:

| Metric | Fixed pad | Moving pad |
|---|---:|---:|
| Sampled native gap |0.150212mm|0.149831mm|
| Pad normal dot cover normal |−0.99798|−0.90222|
| Contact site-localX |−0.208mm|−3.619mm|
| Contact site-localY |−0.683mm|+4.946mm|

Both contact locations are inside the physical validator's finger region: −35mm<X<+4mm, abs(Y)<10mm. Source fixed-finger geometry is19 (`left_wrist_roll_follower_so101_v1`), not geometry18 (servo case on the same `left_gripper` body); moving finger is20. Cover target normals oppose each other. The fixed pad is on its actual flat inner surface, not a nearby palm.

All8states have **zero sampled LEFT-target penetration**, zero hand-hand penetration and minimum LEFT moving-helper/table clearance18.6mm. Existing target-target conflicts are retained and do not count as a grasp repair. Target and RIGHT joint coordinates remain bitwise unchanged.

The actual support approach pitch is−10.65°, limited by native wrist range. The old requested-axis IK residuals remain0.475mm position and0.06363 span-vector error. We do not relabel that as a perfect axis fit: the actual native pad distances, normals and material clearances establish a slightly skewed geometric side pinch. Contact-centroid recalibration is explicitly labeled as measurement at this pose, not a newly solved IK claim.

The positive-roll alternative puts the wrist camera under the hand and was independently rejected for table penetration. Higher negative-roll cover grasps can intersect the upper link by4mm during insertion, despite looking clear in pickup. Those failed candidates remain in the search logs; they were not selected.

## Scope and limits

Geometry uses reciprocal native-triangle finite surface sampling at512points/direction after AABB filtering, including helper/camera meshes. Zero sampled overlap is not universal clearance. The0.15mm pad gaps are static registration margins; contact loads, friction, dynamic retention and task admission are unqualified.

The stable support should not inherit the active upper-part's independently sampled grasp-slide/angle noise. Parent is handling that explicit receipt-driven sampler behavior while retaining active-hand draws.
