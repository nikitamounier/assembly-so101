# Qualified faster native runtime — ready, not active

Newer human direction: measured hinge-stop limits plus tiny reset noise are being implemented in the source optimization chat. This release remains a qualified fallback/reference; do not blindly activate it as the final answer to that newer scope. Main parent must inspect the hinge-limit qualification before choosing the next training version.

The main training run remains intentionally paused at native-v10 checkpoint **32 / 1,046,323 transitions** (original31 archive preserved; current learner and controller both paused after a lease renewal) while all40 retained CPU leases serve the user-prioritized19,200-state build. This release is installed on Chimera, qualified and source-pinned, but has **no active main learner/controller/workers**. Its distributed smoke and migration preview are isolated tests.

The existing monitor recognizes this resource pause, suppresses stale-progress repairs, and notifies the main parent once the complete19,200-state return receipt and all40 restored CPU leases are verified. The parent then performs the versioned activation. Do not reclaim CPUs from the state build.

## Installed release

Remote experiment root: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4`.

- Isolated source/runtime `native-v12/`, operators `ops-v12/`, future main run `run-v12/`.
- Physical signature `f33363e7d1ce26294fe0438000dd0137c7963febb3c812988440501d824a4b48`.
- Retains native CAD/SDF/supplements, friction, safety guards,0.5ms timestep/80substeps per40ms action,40x40 search budgets, observations and rewards. No bare-v8 policy or lower-budget candidate.
- Four coupled source files frozen from `native-contact-optimization-20261107/swept`; all400 active-input source pins verified. `prepared.json` and `ops-v12.json` contain exact receipts/hashes.
- Ordinary SDF skip applies only to three named passive-stop pairs with conservatively certified disjoint closed grid volumes. Preparation asserts every such pair has **zero margin**, as required by that proof. Unknown/budget-exhausted/moving-outside-certificate cases fall back to original search.

## Verified integration

Two separate fresh processes use the actual production RPC worker,10worlds/10physics threads, five tasks and paired held/near starts with varied friction. Both replay the same frozen current-policy Gaussian actions. All **61 physical payload arrays are byte-identical**. Final matched timing excludes policy loading, inference and shutdown: **22.1446s →16.5934s,1.3345× faster stepping**, over three controls/30world controls. Peak worker RSS8499.31MiB baseline /8489.15MiB candidate, below18GiB gate against20GiB retained leases. This short panel is not a fleet throughput guarantee.

The earlier test-wall comparison34.28s→18.22s included baseline policy setup and is not the matched speed claim. Original separate0.25ms benchmark1.83× full small-PPO update/247-pose1235-arrayoracle+9regressions remains source qualification, not a compounded fleet claim.

Two real remote workers/four worlds completed two actual PPO updates/16valid transitions with finite saved weights, unstable0. The migration rehearsal restored checkpoint31 policy, Adam, all1143 SGS histories, sampler RNG, counters and LR exactly; preview checkpoint SHA `da231eb231a86c18803c38223e58520a18f2e289b8394535daaa45957cbcc9ce`. It is a preview, not main training progress. Bank NPZ payloads unchanged; source-versioned manifest directory signature `b2ff5ccca6492fc94bd4f63cfaea4516f5ca95d8d77c70de475f8c44c566c333`.

Receipts: `production-worker-proof.json`, `distributed-smoke.json`, `migration-preview.json`, `deployment-ready.json`, `prepared.json`, `worker-qa2-{baseline,candidate}/result.json`. Actual world arrays and source-PPO/oracle receipts remain remote.

## Parent activation after resource return

First run `../priority_status.py` and require `ready_for_parent_handoff=true`. `handoff.py begin` and `allocated.py` refuse while state-build priority is active. Recheck current source pins and retain/adopt any dynamic CPU ledger; do not silently discard jobs if the ledger changed. No new approval is needed within the existing human faster-native authorization.

From this directory:

```sh
python3 handoff.py begin
python3 allocated.py close-checkpoint
python3 handoff.py close-workers
python3 allocated.py migrate
python3 ../start_v12.py
```

The start command owns a persistent controller session; do not launch duplicates. Close-checkpoint positively identifies only the current native-v10 learner and verifies its latest committed durable checkpoint before archiving. Migration must use that CURRENT native checkpoint, not blindly reuse preview31. Preserve all leases and paused PRIMARY processes. Create remote dashboard alias `v12 -> ops-v12` after migration and before routing the dashboard. Rewrite the full monitor contract with actual ops-v12 pins/run/source/continuation provenance. After a genuinely new saved PPO update, run `allocated.py prove`, verify durable checkpoint and then arm protected greedy scaling as separately configured. The monitor must not activate sources or choose CPU placement itself.

Do not import the unfinished19,200-state bank without its separate admission/source/SGS migration gates. The installed release deliberately retains the current1143-row bank pending that handoff.
