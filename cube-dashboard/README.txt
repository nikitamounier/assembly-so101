Cube handoff live dashboard

Open http://localhost:8880 (local only).
The original reset viewer remains at http://localhost:8877.

Start from workspace root:
PYTHONPATH="$PWD/assembly-so101" assembly-so101/.venv/bin/python cube-dashboard/server.py

Rebuild frontend:
cd cube-dashboard
../assembly-so101/web/node_modules/.bin/vite build --config vite.config.mjs

The server makes one read-only SSH/srun export every10seconds through protected CPU3077705. No Python or model inspection runs on login. Browser checks the local cache every3seconds. Failed reads retain last good data with an unavailable status. Raw run auth keys/server.json are never read or exposed.

Source run:
/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/runs/cube-v6-1024-20261107

Training reward logging started at checkpoint3663→iteration3664. Earlier reward history is unavailable, never filled with zeros or inferred from success rates. The last1200updates are shown. Episode return is undiscounted raw environment reward (success minus2e-5controlcost), excluding timeout value bootstrapping. Complete-episode counts/card means refer to the current recorder generation. Startup fragments and worker-loss partial episodes are excluded from these statistics. Per-step reward uses all valid environment transitions.

Latest successful episode is an ACTUAL complete training episode saved by EpisodeRecorder, with observed12armjointpositions plus cube and target poses. Kinematics reconstructs the local Three.js replay; it does not resimulate physics. When no actual training success has been captured, show explicit waiting state. Never insert scripted/evaluation/test successes into this live run.

Isolated replay verification:
PYTHONPATH=assembly-so101 assembly-so101/.venv/bin/python cube-dashboard/verify_replay.py
This generates one real qualified native MuJoCo placement in a temporary test directory, verifies the14frame/190visualgeom reconstruction, and never changes training artifacts.

Cube now uses35retainedCPUjobs; donated5benchmarkjobs remain excluded. Current cubecontroller612941 uses generation3lock, adopted existingworkers; oldcontrollers503517 AND520464 deliberately remainSIGSTOP so their containing Slurmsteps and adoptedlaunchers stayalive. Never resume/kill them or their containingsteps. Originalcheckpoint64 untouched. Before restarting any controller, use exactPID/start/argv reconciliation and current nativejobepochs; do not blindly duplicate workers/learners.

Canonical local proofs:
assembly-so101/artifacts_local/cube_trial/chimera/dashboard/{production-proof,automation-before,automation-update,automation-readback}.json
Shared monitor continues unchangedcadence; source manifest includes telemetry.py, modified train.py and allocated dashboard_export.py. Physics/reward/PPO unchanged.

Reward contract placement_v2 activated aftercheckpoint5861:wholecube+2mmboundary;actualtablecontact;righttwo-padgraspwithin100msbeforedown;rightonly>=.275;all4pads<=.01Nnormal;linear<.02m/s,angular<.5rad/s;3frames+controlcostunchanged. Controller696771/gen5;prior648184alsodeliberatelySIGSTOP. Seeplacement-v2proofs.

CURRENT run changed by explicithumanfreshrestart toROOT/runs/cube-placement-v2-fresh-1024-20261107;freshpolicy/Adam/SGS,previousrun6091preserved. Controller708043/gen6;prior696771alsoSIGSTOP. Same35CPUallocationsretained. Actualrecordedplaced_heldsuccess observediteration56 (.28s/return.9958/8frames),notfullhandoff. Dashboardfollowsnewcluster.json.

Familycardselectionnowselects latestsuccessfromTHATfamily (orwaitingifnosavedclip). API/episode?family=KNOWN_FAMILY;perfamilyrecordingretainsall9independently. ShowAllrestoresglobal. Samefreshcurrentruncheckpoint406resumed,notreset;controller729445/gen7;prior708043alsoSIGSTOP. Localserverexecsession45237. Browser+all9APIchecksPASS;see family-replaysproofs.

Currentrewardplacement_v3_continuous:initialdragallowed;afterleftliftoff,nozero-padcontactuntilrighttargettouchdownandNOtabletouchoutsidecircleatanytime;bothfaultspermanentperepisode. Samefreshcurrentrunresumed1578;oldv2successesarchivedinsuccesses-before-continuity-v3,newclipsversion3only. Controller791734/gen8,prior729445alsoSIGSTOP. 48local/LinuxgatesPASS.

2026-11-07 USER REQUESTED TABLE-ONLY REWARD + PRE-RULE CHECKPOINT RESTORE
Reward placement_v4_table_only removes zero-pad-touch disqualification; keeps permanent off-target table contact after pickup/liftoff. Initial pickup dragging allowed. All other placement-v2 conditions unchanged/native MuJoCo. 48 tests pass local + Linux.
Restored checkpoint1578/51291430 from checkpoint.before-continuity-v3.pt, source SHAe7b6cee7615134d25a3906fb6b5d3ed68d40a90541d04537bf5467d102a4a732. Exact model/opt/sampler/lr/it/transitions equality verified after reward-only migration. Superseded v3 continuation checkpoint2044/66485947, metrics/clips preserved in superseded-continuity-v3. Visible log restored to1578 then newv4 branch.
35cubeCPU+GPU3598717 retained, original holds/donated5 untouched/no allocations canceled/submitted. Controller817289 generation9; old791734 also intentionally SIGSTOP/neverresume. Source manifest19f36d812c7eec4e4a5146c3c72f7ff47e3df53aef6335d7ebd472afb5f5d312; observer unchanged. SAME ACTIVE30s shared heartbeat fullpreserved/readback promptSHA54aef7a371b41fa31c433df44af21560237dcc018e7fc18100cc69866f16ea7f. Coordination message to initial chat rejected by auto-review: no explicit cross-chat follow-up authorization; not retried. Automation updated directly using supported tool; no delivery history reset.
Local dashboard annotation updated; screenshot cube-dashboard/table-only-restored.png. New successful v4 replay observed from right_transport, not full loose-start handoff.

Policy-matched right_transport bank v7 active from6328; otherfamilies/rewardunchanged. Samepolicy/optimizerresume6327;50local/Linuxgates. Viewer8877 andactiveclusteruseexactv7bank. Contextsidecarpreserves5framepolicyhistory/action/graspclock/carryprefix.

Acquisitionbridgev8:11families, newpartial_right_contact/light_right_grasp cards betweenpre/dual;128TRAIN+32EVALeach. SAMEreward/latestcheckpoint8756continue;all35CPUs+GPUretained. Newresetsboundary8757.
