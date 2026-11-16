# Active reset update: 24 minimum-valid starts

The user corrected the reference to **states & variations**, chat01a1b5de-9df9-7f83-a66f-b5933514cdcb, and explicitly requested adding its repaired states to the live Chimera run. Source reports allow motion and require no unintended overlaps plus retention of held parts. This is initial-reset admission under those two gates, not stationary equilibrium or complete task/hardware qualification.

Active remote root:
`/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24`

Active run: `run-v3`. Parent `.../wowrobo-diagnostic-20261107T205745Z/run-v2` was stopped gracefully after iteration3 /38,243validtransitions. All40CPUleases/400schedulerCPUs retained, same GPU3606264 with approved guarded fallback3598717, noModal. No allocation cancellation/release/reacquisition.

## Reset and model update

-24passed source stages,28task rows across all5tasks. Shared00_loose is associated with each task, not counted as five distinct physical starts. Counts: cover5, connector5, pin5, upper6, final7. One nominal witness per passed source stage; no invented variations or duplicate-padding claims.
-Exclude21_upper_far_side, which still fails bilateral capture. Source selection is frozen in selected.json and inputs.json, sourced from minimum-v6, minimum-v6-source-fingers and minimum-v6-source-fingers-refined. The older published physical viewer bank had22entries and was not substituted for the later passed receipts.
-Matching centered-housing-v6 physical profile originalSHA909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef. Housing registration corrected0.4mm; no mesh/mass/inertia/actuator weakening. Training model identity8ca5214599b52bd60ccef6ac4c1f1919bc542e7519bcb253c5abe2a77fac6f78. Point-frame fingerprints and seated target references regenerated for the centered model, preserving448points/12actions and the0.5mm/1degree/0.4s stable-seating reward.
-Reset conversion uses the witnessed native INTEGRATION state, preserving qpos/qvel, equality activation, actual helper controls and initial qacc warmstart. No external forces are allowed. Fullphysics+equalities+controls+warmstart payload has159values. The old rendered/FULLPHYSICS-only conversion would lose grip preload and warmstart.
-For this new payload only, zero actions hold the witnessed servo command and bounded policy deltas update that command. Legacy payload behavior stays unchanged. This is an explicit control-state migration to retain physical grasp preload; it is not silently zeroed or reconstructed from qpos.
-Current bank remains labeled diagnostic/incomplete globally, while all24included source states carry the passed minimum-validity label. No full production/evaluation qualification claimed.

## Checkpoint-preserving continuation

Migration-v3.json proves exact policy and optimizer preservation, same actual learning rate, counters3/38243. Sampler histories are reset for28new reset entries; old47entry outcomes do not map to these new states. Original checkpoint is saved as run-v3/parent-checkpoint.pt and its SHA is recorded in the migrated checkpoint. Subsequent trainer saves retain parent_checkpoint provenance.

Parent model7b159e7d... is not relabeled as the new model. New observation/reward/bank/native identities are recorded explicitly. Only the next actual log row, iteration4or later, demonstrates SGD using the new bank; migrated checkpoint3 alone does not.

## Verification

-Source-state and control hashes match all24receipts;28task rows have finite observations and exact reset controls/warmstarts.
-Zero-action native rollout matches the trainer exactly for loose starts and all five HELD near-insertion starts05/10/15/23/31. All ten checks had no safety termination. First comparison initially differed only because its clock was not reset; corrected test uses the same episode clock.
-Native model structural audit: exact geometry/body frames/mass/inertia, contact pairs/laws, actuator gains/limits, solver and timestep match the passed-state physical model. Added only the min-contact sensor and256MiB arena capacity. Source/native physics are not weakened.
-Old learner closed positively after checkpoint3; old worker processes closed by exactrun/job/start identity, retaining original holding batches. Old controller retired. New learner log confirms5tasks/28entries and resumediteration3. Private snapshot186inputhashes guarded by the active monitor.
-Release/migration/model-match/operator receipts are in release-receipts.json and the remote root's reset-update-v3.json, reset-update-validation-v3.json, model-match-validation-v3.json, migration-v3.json and ops-v3.json.

## Monitoring and compute

Same shared heartbeat staysACTIVE; PRIMARY intentional pause and CUBE/benchmark retirement preserved. The attempted global heartbeat pause was rejected by automatic approval review, so it remained active throughout the handoff. Its latest contract now targets only this successor for diagnostic monitoring; no duplicate monitor/chat.

Pinned local entry `v3/monitor_entry.py` has read/scale/failover modes. Scaler stateautoscale-v3 retains100slots/1024envs/32steps/32768targetsamples,1000rolloutCPUceiling, pending counts,10CPUs/worker,5minute forecastcache, own-host exclusion, near-start<=180s and uncertain-submit fence. New submissions require a REAL current-run log row>=4 and fresh SGD. Old scaler is disabled. Retained400CPUbaseline is not a claim that1000CPUs are currently available.

```sh
python3 robotics-assembly-3.0/operational/wowrobo-chimera-20261107/v3/status.py
python3 robotics-assembly-3.0/operational/wowrobo-chimera-20261107/v3/monitor_entry.py read
```

Later repairs do not hot-load. New source states/model changes require a new frozen update and recorded continuation.
