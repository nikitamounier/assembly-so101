# Contact-kernel and installed-interface trials — November7,2026

## Accepted result

Promoted only the exact grid optimization: resolve the immutable target grid once per surface-tree query; evaluate distance and a qualifying gradient from one interpolation. Generic non-grid plugin callbacks retain their original path. Grid clamping, half precision values, arithmetic and contact sample order remain unchanged. Both native libraries must be rebuilt together for the new exported grid helpers.

Three new canonical plugin changes: `assemblenist_sdf.cc`, `so101_contact_search.c`, `so101_surface_tree.h`. They preserve the previously qualified conservative separation and four-way cache. `qualified-source/` and `qualified-source-hashes.json` contain the complete five-file bundle, including the previous contact backend/cache header; use allfive if the target release predates the last handoff. `source-before-hoisted/` and `promote_hoisted.py` preserve the source delta with concurrent-edit guards.

| Test | Baseline | Hoisted/fused grid | Result |
|---|---:|---:|---:|
| Actual32step/64transition PPO |99.6285s|86.5508s|1.1511xthroughput|
| PPO collection |99.0625s|86.0237s|1.1516x|
| ProcessCPU |160.7054s|137.4466s|14.47%less|
| Four-stage warm interval sum |5.24901s|4.65518s|1.12756x|

One matched diagnostic update per version, same archived native32policy+Adam, seeded resets, .5ms/80substeps, two local native worlds/two threads onSMT siblings[2,98]/onephysicalcore ofGPU105E. Initialization/build excluded. This is not a fleet estimate, productionSGS-weighted claim or NIST ratio. Bank remains original1143rows; expanded19200bank not testedhere.

Exact audit: model tensors, Adam, learning rate, iteration/transitions AND sampler checkpoint state all equal; every scientific metric equal. Both64transitions, numerical instability0, same1safety termination/3episodes.120replayNPYarrays byte-identical. Independent C++reference compares separate plugin distance/gradient callbacks with the hoisted conditional evaluator:162000queries/54grids/thresholds[-100um,0,+50um], finite inside/outside-grid positions; distances and qualifying gradients bitwise equal. Native builds passedWall/Werror. See exact-ppo-audit.json, newhost-exact-audit.json, hoisted/grid-probe.json.

## Rejected or unpromoted candidates

- Fused interpolation without hoisting repeated invariant lookups:120arrays exact but negligible fixed-stage improvement. The successful version also resolves grid/plugin/hash metadata once per pair query.
- Simple per-sample winning-bin cutoff:120arrays exact, roughly13%slower on the old-host four-stage sum. Rejected.
- Seeded chunk branch-and-bound: reevaluate previous winning samples to establish bounds, then conservatively reject32-sample chunks; retain original emission order/ties.120arrays byte-identical, but new-host sum5.33964s vsbaseline5.24901s (~1.7%slower). Rejected. The all-bin conservative cutoff is weak when seed bins are missing; no normal-direction-cone pruning was implemented or claimed.
- Active installed-interface contact omission: old-host forearm warm2.68477→1.29593s, upper1.94313→1.54912s; these are genuine fixed-stage gains. Original softwelds fail the predeclared short-load relative tolerance:forearm rotation~1.25degrees. Pose calibration of onlyservo2, then allthree references, did not resolve it.
- Calibrated references plus angular torque scale0.05→0.2: passed short64msand500msstatic force/hold comparisons (tolerances: baseline+0.2mm andbaseline+0.5degrees). Articulated shaft joints retained and still moved. However actual-policy PPO produced2unstable transitions and5/5safety-terminated episodes, versusbaseline0unstable and1/3safety. NativeQACCwarning atDOF22/time.2635s. REJECTED for current training/reset setup. Its47.36siteration timing is excluded: earlier terminations and instability make it a different workload. No learning-quality gain claimed.

The attachment branch changes physics, unlike the accepted kernel. Its opt-in header/factory/fit prototypes in `integration/` are marked REJECTED.json and are NOT promoted or deployment candidates. Only the exact kernel is eligible for main-run release. Failure might involve reset/model incompatibility and dynamic constraint conditioning; this run does not establish one unique cause or prove all attachment-only models impossible. Such a model would need independently admitted resets and dynamic validation.

## Load-test details and provenance

Shortcases128steps/64ms, two saved worlds in upper_far_side21 andforearm_final_near31:hold,externalforce[1,-.5,.5]N,then .002Nmservo2torque. Applied forces are not cleared between shortcases; later cases combine earlier loads. Identical sequence used for reference/candidates. Longcases clear applied forces eachreset and use1000steps/.5s externalforce, bothstages/twoworlds. Static success did not predict actual-policy safety.

Fixed references measured from one saved forearm-final-near state; eq_data3:10 layout validated against original XMLrelpose before modifying private model. Torquescale.2applied to allthree completed-interface welds. These mutations occur only in diagnostic processes. Sourceprofiles/currentbank/livelearner untouched. CandidatePPO uses an explicitly unqualified diagnostic bank linked to originalNPZs, new model identity bound to changed equality data/native source, allow_debug flag required. This is not new bank admission.

Original allocation3598717requeued fromGPU720EtoGPU105E; first all-reference calibration attempt was explicitly rejected bySlurmbeforeexecution. Baseline repeated onGPU105E for exact/pruning comparisons. Old-host ablations remain inablation.json; do not compare oldhost timings directly withnewhost.

Two setup failures beforepolicycollection preserved:bankidentity guard correctly rejected changedphysics againstoldmanifest, then wrapper assertion usedLocalVecEnv.m instead of.env.m. Corrected via separate unqualified bank and correct inner environment. Failedlogs retained. No live allocation/checkpoint/source/CPUrole changes bythischat. All compute uses isolated ROOT/native-contact-optimization-20261107/contact-patches-20261107 insideexisting3598717.

## Reproduction and deployment

patch.py/hoist.py/patch_winners.py preserve source variants; experiment.py/run.py panels; load.py/calibrate*_load.py/long_load.py physicaltests; calibrated_ppo.py explicitunqualifiedpolicytrial; grid_probe.cc/grid_probe.py independentkerneloracle; final_exact.py acceptedupdate/probe; logs and JSONreceipts retained. Mainowner must copy the qualifiedfive-file bundle into a new versioned release, rebuild bothlibraries, run normalcurrent-bank/worker gates and continue from CURRENTcheckpoint under existing authorization. Physics/state format/attachment behavior of the accepted version is unchanged. This chat has not activated training.
