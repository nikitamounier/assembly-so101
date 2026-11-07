# Bounded equilibrium method diagnostics — 2026-11-07

These are diagnostic candidates only, not published reset banks or physical admission proofs. No core source, banks, or viewer files changed here.

## Natural full-arm settle, previous no-own-stop profile

`probe.py` retains original servo commands and passive hinge dynamics. It activates only already-installed hardware attachments. There is no early collision abort, no applied force/torque, no bracket-angle target, no velocity reset during integration.

- `00_loose.json`: 1.0 s integrated in28.67 s wall (initialization excluded). Upper/final module root drift19.41/24.21mm; motor drift14.40/16.18mm. Final maxqvel0.19555rad/s (shaft1), next coverangular0.07123rad/s. Both modules' original table poses are not equilibria.
- `18_upper_floor_grasp.json`: 1.0 s integrated in77.87 s. Final maxqvel0.91455rad/s; leftfixedpad0N/moving38.22N; rightboth0N. Upperroot drift23.92mm, motordrift7.159mm. Simply saving a settled frame would lose the intended grasp.

## Bounded static native-force least-squares, new own-stop full model

`refine.py` uses the new own-stop model (`outputs/equilibrium-model-20261107/profile`), all46 generalized coordinates and12 real motor commands. Search uses zero velocity only to evaluate a mathematical candidate; a separate untouched-control forward hold falsifies it. No external forces/welds were introduced.

19 iterations reduced scaled acceleration cost69870→698.9 but ended at xtol with optimality1.12e15 and maxqacc411.14rad/s². Initial bilateral LEFT34.26/58.00N is lost by.025s; .1s hold maxqvel0.7886rad/s, leftfixed0N/moving48.95N. Contact-set changes make finite-difference residuals nonsmooth. This does not establish equilibrium.

## Real motor gravity compensation and jaw closure

`gravity.py` adds qfrcbias/kp to actual helper position-servo commands (clipped to native ctrlrange), closes jaws at0.4rad/s only while a pad is unloaded, and integrates. No external forces or velocity reset. Full .25 s interval, new own-stop profile.

At.05s RIGHT is bilateral6.758/28.913N. LEFT is0/35.866N. Raw contact trace proves the fixed cover contact disappears physically; it is not merely rejected by the pad mask. At.25s RIGHT6.601/28.644N; LEFT0/39.990N; maxqvel2.393rad/s. LEFT gripper body displacement is[-.300,+1.364,+.044]mm, away from the cover along the pinch normal. Initially a valid fixed-pad rawcontact had21um penetration, sitepoint[-.669,.460,-.010]mm andnormaldotZ=1.

Thus gravity compensation helps active servo grasp, but support-grip servo deflection under jaw load needs a real inward hand-command adjustment and independent forward hold check. The fixed pad is not being hidden by the classifier. Do not silence admission checks or save zero velocity to claim stationarity.

## One-millimeter inward real-command test

`inward.py`: initialq unchanged; gravity compensation plus bounded arm-Jacobian least squares adds a desired1mm site-Z translation to LEFT real motor commands (achievedlinear0.957mm, with0.228degrotation). Adaptivejaw closure is capped when moving-pad force exceeds5N. .4ssettle then.2sunchangedcontrolhold, no angle/velocity intervention.

Fails: LEFTfixed0/moving48.02N at.6s, no rawfixedcovercontacts. Handdisplacement[-.411,+1.356,+.070]mm awayfromcover. RIGHTfixed0/moving7.36N: unlike uncapped previousgravitytest, the5Ncap also prevents RIGHT reaching its fixed pad; any support-specific strategy must not be imposed on active RIGHT. Fullhold minimumforces ininward18-1mm.json; ctrlbitwisefixed throughoutfinal.2s. No successful equilibrium candidate. A real jaw-open→fixed-pad-preposition→gentleclose sequence may be needed to acquire both pads before unilateral jaw reaction drives the compliant helper away; this is a hypothesis, not tested here.
