# Real final-module reference versus current simulation

## 2026-11-07 [codex] Motion audit

Source: `/Users/nikitamounier/Downloads/IMG_1634.MOV`, 14.77 seconds. Reviewed extracted frames at 0, 4, 7, 10 and 13 seconds. The clip shows manual articulation and lowering of the final link onto the receiving arm. At the sampled elevated frames the human holds the printed free end; it does not itself establish the proposed robot motor-case pickup or a gravity-driven mechanical stop. Grip-by-motor, passive folding to the motor, then contact-driven opening/sliding is the user's explicit intended sequence.

**Conclusion: present references approximate several poses, but current successful insertion test does not reproduce or validate that dynamic sequence.**

- Current reset references are separate authored states. Shaft3 angles read from the new `photo-resets-repaired-cover-support-v1` NPZs (`states` starts with time, then 51 qpos): 26 floor 0 degrees, 28 transport 8 degrees, 29 far-side 17.5429 degrees, 30 half 7.0196 degrees, 31 near 5.4191 degrees. Their hinge changes are authored, not a trajectory or proof of continuous contact. Transport sag is prescribed as 8 degrees by `configs/wowrobo_reset_sequence.json`, through `PhotoResetSequence.sag_held_link`, which preserves case pose while setting the relative hinge/root geometry.
- The final CAD fit success uses `outputs/insertion-fix-20261107/final/canonical/guide-scene.guide.json`: nq=nv=1. All internal hinges including `target_servo_3_shaft` are removed and baked at aligned start. `candidate-trajectory.npz` has qpos shape `(6204, 1)`, only axial travel. It reaches 25.002862mm at 20N, but cannot fold or open the bracket. This is an explicitly disclosed one-axis fit fixture.
- The independent physical reset model does preserve a free shaft3 hinge (range -1.7..1.7rad, damping .02) and gravity. However its `servo_3_case__contact` is in direct child body `target_servo_3_pivot` of the printed-link body `forearm_module`; `forearm_module_printed_0/1__contact` live on the parent. Parent filtering remains enabled (only sleep is disabled). The explicit contact list has case3 versus the RECEIVING upper printed fork, but no case3 versus its OWN printed bracket. Thus ordinary automatic contacts do not currently supply the requested bracket-against-motor physical stop. A broad numerical hinge range is not a calibrated physical stop.

Recommended next physical implementation, separate from this read-only audit:

1. Add a narrowly scoped explicit own-bracket/case contact pair for the intended stop, preserving all other parent filters, and validate against source CAD plus real stop-angle measurements. Do not blanket-enable every articulated parent contact.
2. Test motor-case held with a physically supported gripper, shaft3 passive and gravity active. Verify the bracket folds until actual stop contact, rather than prescribed sag.
3. Lower with the hinge free and actual receiving-arm contact so the far end can slide and the angle can open. Log shaft angle, both relevant contact pairs, case path and final mating depth. Maintain the non-active supporting arm and verify its loaded hold.
4. Keep the existing guided fit clip labeled as a fixed-shape fit diagnostic; only call the new sequence matched after this trajectory succeeds.

No simulation, contact masks, bank poses or qualification status were changed by this audit.
