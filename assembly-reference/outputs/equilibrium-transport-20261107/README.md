# Actual servo transport captures — 2026-11-07

Reusable recipes: `recipes.json`. Apply only each recipe's `affected_joints`; all other initial qpos are bitwise unchanged. A parent's corrected prerequisite assembly must not be overwritten with the historical original prerequisite qpos included in the diagnostic arrays.

## Adopted 20 upper transport

`transfer-v2/recipe.json` / `20_upper_transport.npz` transfer the *actual loaded* state18 RIGHT servo2 grip into a reachable transport posture, relax wrist flex 0.06rad from its limit and deepen the servo in the fingers by8mm. This corrects the original floating transport grip, which fell out before jaw capture. The incoming case moves about36mm and rotates15.31deg relative to the old inferred transport reference; assembly/LEFT and every unrelated initial coordinate are preserved exactly.

Actual full-v5 native scene:0.8s settle +0.2s unchanged-control hold, minimum RIGHT fixed12.445N / moving38.953N; case hold motion3.964um /0.001908deg. Both initial and hold NativeBoundaryAudit pass; maximum sampled overlap0.266/0.254mm includes existing base fit, within the unchanged0.5mm guard. All helper source vertices clear the table (base foot roundoff~−3e−15m). No hinge targets, holding forces, gains or force limits were changed. Actual measured grip calibration is in the recipe. Training admission remains separate.

## Adopted 28 final transport

`canonical28-v1/recipe.json` / `28_final_transport.npz` use the exact identity of servo2 and servo3 source case fields (identical grid SHA, existing field-to-source rigid matches~16nm) to transfer the successful actual inner-face servo2 grasp to the LEFT servo3 grip. This avoids the old final-case corner/back-face grip, which held on the floor but fell out when airborne. Wrist flex is0.06rad below its limit. Incoming case moves about59mm and rotates26.78deg from the old inferred transport reference; every unrelated initial coordinate is bitwise preserved.

Local physics scope: unchanged LEFT arm, free final-module root and both passive shafts, and the original native source-mesh floor; unrelated geometry contacts and completed-assembly welds are disabled for this local capture proof. No active component DOF is fixed and no force/torque guide is applied. Parent must verify its corrected prerequisites and the full combined scene.

0.4s settle +0.2s unchanged-control hold: minimum LEFT fixed27.373N / moving40.292N; servo3 hold motion5.461um /0.001804deg. Native source-triangle audit finds no incoming forearm-vs-assembly or RIGHT collisions. The only incoming helper contacts are intended LEFT fingers against servo3 case: maximum initial99.0um / hold104.6um overlap. All helper source vertices clear the floor. Measured actual grip calibration and local span/approach axes are in the recipe.

## Rejected diagnostics preserved

- `preload/`: geometric-only5um capture with original transport pose fails both20 and28.
- `transfer-v1/`: loaded18 transfer without deeper fingers holds20 almost still but fixed-pad load intermittently vanishes, so rejected.
- `floor26-isolated/`: actual floor26 LEFT/forearm/table acquisition passes locally (3.334/5.541N;18.47um/.01023deg), but its original corner grip fails after direct airborne transfer.
- `transfer-final-v1`, `transfer-final-v2`, `transfer-final-v3`: direct floor26-grip transfers lose the held case. Stronger jaw commands and a different candidate-only initial hinge do not fix the wrong contact geometry. These are NOT adopted.

No production source, CAD, model contact laws, banks or viewer were changed by this subtask. Every saved accepted hold retains its real integration state and control vector. The 28 local proof does not establish global prerequisite retention, reward admission or training eligibility.
