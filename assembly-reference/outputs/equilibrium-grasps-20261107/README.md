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
