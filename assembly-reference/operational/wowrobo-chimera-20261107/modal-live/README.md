# Active run: native rollback v9

Latest human rejected bare MuJoCo transitions. [Native checkpoint rollback and evidence](rollback-v9/README.md). Barev8 and earliercloud/compute descriptions below are historical, not active instructions.

# Active run: faster bare MuJoCo v8

Latest human requests faster physics first, then CPU planning. [Active run and measured PPO evidence](bare-v8/README.md). Native cloud pools below are historical; no old native runtime auto-restart.


## 2026-11-07 [codex] Continuous Modal and capacity phase7

Human rejects any artificial Modal cutoff: keep the existing continuous4092physical-core pool running;43200s function leases automatically renew. No billing purchase/upgrade/cash charges authorized. Last authenticated balance243.27USDcredits,500USDusagecap/213.27headroom; estimated219–245USD/hour full Modal fleet, so more funding/headroom needed for sustained use. Funding estimates alert only and never stop the pool.

Same run-v6 now uses ops-v7/active-inputs-v7 with9600envs/960slots and307200target transitions per32-step PPO batch. Model8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879 and1143reset rows unchanged. Checkpoint25/966743 transitions archived checkpoint.before-capacity-v7.pt; exact policy/Adam/lr/SGS/RNG migration PASS, SHA df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b. Learner resumed25 on3606264; all40CPUleases retained, old worker leaves positively closed. GPU capacity gate218278400bytes PASS. ContinuousModal app ap-fiWb1Vv2oWbhu6zDdPc7vK retained, no duplicate.

Additional4092SAILtotalvCPUs human-authorized in source chat01a1b810-7172-7c43-94e2-dffaec1404e4:4088rollout+4relay,448processes. Its separate owner is provisioning after verified960-slot listener; not yet claimed fully ready. Shared sail-live/target.json publishes SAME run-v6/newops-v7 route. Budget409Modal+448SAIL+up to100Chimera=957/960slots; Chimera protected greedy ceiling1000CPUs, baseline400retained. No450/500Chimera cap or50-SAIL reservation remains authoritative. Autoscale-v7 armed; monitor invokes it only after actual committed PPO26 and source/finite/freshness gates. Dashboard aliasv6→ops-v7, recorders and lineage preserved. Latest model has not learned complete assembly; earlier success is mainly near/partial cover.

# Continuous hybrid training run-v6

The user explicitly rejects the assistant's one-hour Modal cutoff. Keep Modal running; automatically renew technical function leases. No elapsed-time shutdown or scheduled pool expiry.

Current app: ap-fiWb1Vv2oWbhu6zDdPc7vK. Controller: launch_continuous.py, persistent exec37042, exclusive continuous-owner.lock. Read active.json, live-continuous.log and cloud-status.json. Stop only by the per-pool stop-request.json (path in active.json), verified training completion, actual failure or real provider/funding barrier. Do not create a duplicate pool. Continuous-release.json pins source/tests; all earlier apps are stopped.

## Compute and credits

Target4092physical rollout cores=68x60+12/409processes, plus2-corebroker. Twelve distinct real native finite controls passed before dispatching the full fleet. Placement is still filling; provider running_inputs and current learner masks are authoritative, cumulative initialization/first-control IDs are not current utilization.

Verified billing snapshot243.27USDcredits/286.73creditusage/0charges,500usagecap (213.27headroom). Estimate219–245USD/hour at full scale, about one hour of existing credit/headroom. No topup needed to start; more credits/headroom needed to sustain. No purchase, plan upgrade or out-of-credit charges authorized. cloud_status.py estimates funding conservatively and requests parent reconciliation before low funding; estimates never stop the pool.

Technical function leases43200s automatically roll over inside this app. Healthy calls remain untouched; normal completed calls are replaced, unexpected exits surface. There is no global pool timer. Broker TLS epochs are privately refreshed through an ephemeral directory, with atomic0600worker tickets and exact-owned bridge replacement. Retry coordination generations prevent the inherited300s give-up from becoming permanent after GPU failover. Tiny broker uses nonpreemptible CPU for stability; bulk physics keeps normal rates. Five regression tests pass, including actual inherited retry-loop recovery after12minute outage and duplicate-free renewal. No physics/PPO/RNG/worker-message math changes.

## Frozen learner

ROOT /large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4; RUN run-v6, ops-v6, dashboard aliasv6. Bank25prototypes+874accepted TRAIN variations/154EVALheldout/1143taskentries. Model8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879/nativepayload162; immutable runtimebundle2c991a016a71976d203b94440ce36e9d2178ace1c40890dda8adcc7cbcfe249b. Minimum-valid diagnostic, not full production qualification. Controls/velocities/warmstarts/friction preserved, all1143finite resets and native/admission parity0 verified.

5090configured environments/509slots,32steps/162880target transitions,5epochs/4PPOminibatches/micro1024. Retain all40tenCPUChimera leases (400scheduler CPUs including pending/requeues), one approved GPU learner with idle fallback3598717/3606264 and exclusive locks. Latest observed v6update23 was finite/source-valid; runtime masks determine actual valid batch size. Native SDF physics remains the bottleneck. Earlier PPO14 had one256Mconstraint-arena warning/10guardflags; laterobservedupdatesunstable0. Preserve guards, record new events, qualify capacity changes separately.

Checkpoint13/269713exactpolicy/Adam/lr/counters continuation,213SGShistories mapped/930new. Actual checkpoints14andlater are saved; v6/live-proof.json has independent finitepolicy/Adam/model/bank/SGS proof. Passive recorder29groups/28bodyposes uses already transferred CPU arrays and bounded background writes. Dashboard http://localhost:8881/ serves actual complete success-preferred clips with failure fallback and exact identities.

## SAIL and slot budget

SAIL is separately human-authorized and preparing; [integration contract](SAIL-INTEGRATION.md). No active capacity claimed until receipts. Reserve50SAILten-envprocesses plus409Modal; protectedChimera cap500CPUs/50processes so all fit509slots. Baseline40leases retained, none cancelled. Compare provider-nativevCPUs honestly rather than treating them as Modalphysicalcores. Current Modal bridge has409channels; do not silently attach SAIL to it.

## Monitor

Same shared heartbeatACTIVE/30s/samechild; monitor-contract-v6.txt is current. PRIMARY intentional pause/source integrity and CUBE/benchmark retirements are preserved. Use approved networking for exact wrappers:

```sh
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py read
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py failover
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py scale
/tmp/assembly-modal-sdk/bin/python operational/wowrobo-chimera-20261107/modal-live/cloud_status.py
```

Scale only when actual new log>=14/fresh<=1200/source-valid/finite. Keep faithful test-only projections, own occupied-host protection, pending/uncertain fences, near<=180s/cache300s/max10newjobs percycle. Do not hotload states or sources. Old one-hour/no-renewal instructions are superseded. [Historical bounded v6](README-bounded-v6.md), [historical v5](README-v5.md), [state-release proof](v6/README.md).
