# Cover-support capture and equilibrium diagnostics — 2026-11-07

The old model's LEFT hand moved about2.3mm away from the cover even when its jaw was deliberately OPEN and both cover-pad loads were zero. Requested inward control displacements0.5mm and1mm gave almost identical outward motion. This falsifies the earlier hypothesis that unilateral jaw pressure alone caused the support failure.

The diagnostic led to inspection of helper fixedbase/firstshoulder bearings. These had the same native world-parent collision exception as targetservo1. Their physically internal mesh interfaces generated false contacts and constrained shoulder yaw. The model agent added explicit helper base/shoulder body exclusions in `outputs/equilibrium-model-20261107/profile-bearing-v4`, preserving actual bracket-stop pairs.

Historical v3 acquisition trials are retained in `acquire-{halfmm,1mm}-t15.*`. They opened the real LEFT jaw0.045rad for0.15s, then used a small actual jaw torque through its position servo, with a final0.2s unchanged-control hold. Neither established a two-sided grasp. These trials used the defective helper bearing contacts and are invalid as evidence against the corrected cover grasp.

The v4 baselines leave LEFT's authored cover grasp and its jaw control unchanged. RIGHT may close its actual jaw during acquisition. The final0.2s uses exactly unchanged controls. No helper weld, virtual motor force, bracket-angle target or velocity reset is used. A comparison trial adds small gravity feedforward only through the existing bounded position-servo controls.

Commands:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-grasps-20261107/baseline.py --profile outputs/equilibrium-model-20261107/profile-bearing-v4 --seconds .8 --name baseline-v4
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-grasps-20261107/baseline.py --profile outputs/equilibrium-model-20261107/profile-bearing-v4 --seconds .8 --gravity --name baseline-gravity-v4
```

Current partial evidence at0.4s: original LEFT control has two-sided loads2.76/2.91N, hand displacement only32um, rather than the old2.3mm drift and unloaded fixed pad. Full unchanged-control hold results are saved in the matching JSON files on completion. This is a local grip diagnostic, not complete scene/reward admission; unrelated loose target settling still dominates global velocity.


## Corrected-model local grip result

After0.6s settling and0.2s unchanged-control hold, originalLEFTcontrols on profile-bearing-v4 retain opposed cover contact: minimum fixed2.72686N/moving2.67676N, maximum hand-site hold motion0.856um. RIGHTservo minimum8.83885/26.17066N. Cover-relative-hand endpoint motion0.971um/0.000182deg. No new LEFT acquisition pose or jaw-force controller is required. Gravity feedforward comparison also passes the local cover hold but is unnecessary. Exact local evidence baseline-v4.json/.npz and hold-endpoints.json. Whole scene remains unqualified due future looseforearm rolling/rocking.

## Loose forearm preparation

`loose-forearm-v1` isolates only the physically independent unheld forearm and the original table, preserving its free root and both passive shafts, all source geometry/contact settings, originalinertia and gravity. It starts from the actualbaseline-v4 endq/qvel and continues5seconds with no actuation, externalforces, velocity resets or hinge targets. It does not pass: last0.5s rootmotion0.442mm/0.2059deg, peakvelocity0.3658rad/s. Only two SDF-plane point contacts support the whole module (printed1 and servo3case); COMprojection lies4.4mm off their contactline, explaining continued rocking. Exactcontactreview and allwindows are retained. Native source-mesh/plane manifolds are a proposed exact support remedy, not yet applied here.


The old-floor gravity-potential candidate (`loose-forearm-energy-v1`) lowers COMheight20.363→19.879mm with worldroll+.220deg/pitch−.500deg and unpowered hinge3 at9.943deg. Independent2s originalSDF-floor replay improves the final0.5s motion to51um/0.0237deg, but peakqvel0.0899rad/s still fails the strict0.05 gate. This is retained as a rejected candidate, not a qualified reset.


## Verified loose forearm library on exact source-mesh floor manifolds

`loose-forearm-v2-native-floor` uses corrected profile-bearing-v5 source-mesh/plane contacts, with the same sourcegeometry, masses, inertia, friction, gravity and passivehinges. It transfers actualend q/qvel from baseline-v4 (not a zero-velocity restpicture) and continues integrating with no actuation, equalities or appliedforces. First0.5s settles. The following0.5s passes: rootmotion3.76e-13m, case3motion1.75e-12m, allrotation changes at numericalroundoff(~4.5e-6deg), peakqvel2.56e-11. Three native plane contacts form the support manifold; minimumdistance−0.159um. Freely settled shaft3 is2.795160685deg, shaft4≈0.

`latest.npz` contains the actual isolated qpos9/qvel8 and jointnames (`forearm_module_free`, `target_servo_3_shaft`, `target_servo_4_shaft`). Map each joint to the matching full-model addresses; preserve qvel rather than clearing it. Its source diagnostic has no actuators/equalities, so this is physically independent loose-module equilibrium evidence, not full-scene/task/reward admission. Parent all-state checks remain the final shared-model proof. Exact sceneSHA/profileidentity and both0.5s windows are retained in result.json.
