# Active faster MuJoCo training — November 7

`run-v8` is live on retained Chimera workers, resumed from native checkpoint25 /966743transitions. No paid CPU expansion is active while useful compute is reassessed under the faster physics. The native Modal and SAIL pools were retired for the explicit physics release, not a time cutoff; SAIL cleanup reports no errors.

This is **approximate assembly physics**, not an accuracy-preserving native optimization. Same five tasks, sparse stable-seating reward, action scales,1143accepted TRAIN rows and detailed CAD point inputs. Primitive boxes/distal finger pads, stock MuJoCo,2ms timestep/20substeps,10solveriterations; penetration/velocity terminations removed, numerical checks retained. Robot self/part-part contacts disabled. Coarse grasp drift and0/40held-out success in the earlier small pilot remain known. Speed alone does not demonstrate learned assembly.

## Verified activation

- Isolated adapter at remote ROOT/bare-v8/assembly-so101; original native files untouched.
- Bare physics signature `f881fdddcaaee07cd546f9ef11f14a2fd1c929f516f74a2d219026a110b2aa5e`; original CAD/parent model `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`.
- Exact match to pilot's executed environment class. Explicit native-reset import only accepts those exact parent/new identities; ordinary identity checks otherwise remain.
- Two-worker/20env distributed smoke:3actualPPOupdates/1920transitions, finite, checkpointdurable,32step rollout~0.46s.
- Native checkpoint archived `ROOT/run-v6/checkpoint.before-bare-v8.pt`, SHA `df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b`. Controller/learner/nativeCPUleaves positively closed; all40CPUallocations retained.
- Explicit cross-physics migration preserves policy/Adam/lr/counters/all1143SGShistories+RNG exactly. New initialcheckpoint SHA `603d0aae9448cf167ebdb01afed847b2553d5b3ba3a8886c8551b13595b0d34c`.
- Independent durable checkpoint203:178newupdates /2260310newtransitions, total3227053. Latestmatching~11719TPS, t_sim0.274s, t_update0.396s, recent30updates unstable0. Active390–400worlds as one retained lease requeues; configured9600capacity/960slots/307200targetbatch, actual12480–12800validtransitions/update. Training success still0 in that snapshot.
- One retainedGPUlearner starts3598717, idle-only3606264fallback. No GPU/CPUallocation cancelled. No newCPUjobs submitted.
- Passive observer29groups active; dashboard automatically followsv8 and displays new physics signature/approximate label. Four dashboard export regressions PASS.

RemoteROOT: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4`.
Operators `ROOT/ops-v8`, source manifest `ROOT/active-inputs-v8.json`, dashboard alias `ROOT/v8`. Local `../target-v8.json`, `../start_v8.py`, `../monitor_entry_v8.py`. Same sharedheartbeat ACTIVE/30s; PRIMARYintentionalpause/CUBEretirement preserved. `autoscale-v8` disabled pending CPU sizing. The existing useful400CPUbaseline already produces~12kTPS; network/learner overhead now matters more, so old4092+4092paid-fleet sizing is not assumed optimal.

Evidence: [live proof](live-proof.json), [distributed smoke](distributed-smoke.json), [migration](migration.json), [source release](release.json). [Pilot details](../../../../outputs/bare-assembly-20261107/README.md). Frozen cloud overlay must match final SHA `6b4c3119f2f19aac258b8a8f97461c3f1f2ef7af54c8443d587285630bb4700c` before any future cloud release.
