# Engaged reset repair — 2026-11-07

## Measured registration defect

The output-shaft coordinate is positive toward the front. Both servo2 and servo3 printed cages have inside faces at -34.1475 and -0.7475 mm, a 33.400 mm gap. Their case housing caps are at -33.5975 and -0.4975 mm, width33.100 mm. The housing center is 0.400 mm too far forward: front interference0.250 mm, rear clearance0.550 mm. Centering the housing backward0.400 mm leaves0.150 mm on both sides. Rawsource face-plane and area evidence is in axial-registration.json.

This is a registration correction, with no mesh cuts or contact filtering. New reusable module `assembly_policy/so101/wowrobo_housing_registration.py` applies an idempotent XML marker and shifts only the two case-body placements along their original output-axis lines. Joint positions/axes, meshes, masses, inertias, gains and force caps remain unchanged. Its migration shifts the corresponding free printed roots so actual world case poses and xanchor points are preserved. All25 source references pass the exact migration checks in registration-migration-proof.json. For final insertion, incoming case and LEFT hand must follow the changed upper receiver frame coherently.

Without centering, a nominal own-stop contact near the shaft front face causes interference at all original photo upper angles. Simply raising the hinge to -8.6deg removes this interference but lifts the whole far-end patch26–49 mm off the photographed support. That is not an acceptable pose fix. After centering, original photo-angle own contacts vanish while real outerhousing collisions remain at larger fold angles. See centered-true-stops.json and far-native.json.

## Static pose candidates

`coherent-recipes.json` and `*-coherent-initializer.npz` contain qpos51/control12 candidates on the centered model. These are not dynamic admission certificates.

- Upper21/22/23 need small actual hinge initialization corrections (~+0.32,+0.32,+1.74deg) to remove printed/base interference while preserving the held case pose.
- From support25 onward, the installed upper link is raised through its free hinge to -15.1976deg (~+7.44deg from the old intersecting pose). RIGHT's goal follows that upper frame; new RIGHT IK position residual30um, span error0.020.
- Final29/30/31 cases and LEFT hand follow that raised receiver rigidly before the final passive hinge is refit. The case remains pocket-aligned; final relative hinge candidates are about13.07/6.79/1.41deg. RIGHT support shares the same pose across all late starts.
- All9 static NativeBoundaryAudit checks pass with unchanged0.5mm material limit, and helper floor minima are zero at fixed feet. Maximum sampled overlap0.266mm is inherited fitted-interface preload; the former0.5–5mm printed-link penetrations are removed. Actual proofs: coherent-native-audits.json.

The raw coherent controls target the new helper joint postures; they do not yet include new clamp/gravity preloads. Full native short grip retention and any capture correction remain the parent's responsibility. An additional RIGHT support capture test is recorded separately under preload/ if completed; it uses the same geometry/contact laws and real bounded jaw controls only.

## RIGHT support capture

`preload/25_right_support.json/.npz` is a real short native replay on the centered derived model. Initial capture shifts RIGHT fixedhand0.19075mm towardthebody, closesjaw0.01465094rad and uses actual jawcommand preload0.0012rad within originalactuatorcaps. After0.05s settling, a0.2s unchanged-control hold retains minimumfixed8.43655N/moving7.05881N; upperbodymotion23.27um/0.0118deg. No new hinge target, holdingforce or helper weld. Initial and end NativeBoundaryAudit bothpass (support-capture-native-audit.json).

`coherent-captured-recipes.json` and `*-coherent-captured.npz` share that actual RIGHTq/control across25,26,28,29,30,31, whose upperreceiverpose is identical. RIGHT controls already include gravity and clamp preload; do not add it twice. LEFTcontrols remain the ordinary retargetedposture for these static candidates; LEFTactivegrasp capture/retention is still the parent's separate check. This does not claim every late stage dynamically passes merely because the sharedRIGHTsupport does.

The older force-held final-module diagnostic folded to10deg while using the original off-centerhousing contact. Centering may change the passivefold angle until true outerhousingcontact; that old replay remains historical and does not qualifythecenteredmodel. Trueouterstopsremainaftercentering, e.g. source21theta−.9707rad retainsabout2.97mm outerhousingoverlap andsource29theta+.7062rad retainsabout1.01mm overlap. These out-of-range collisionprobes are geometry tests, not acceptedstates.
