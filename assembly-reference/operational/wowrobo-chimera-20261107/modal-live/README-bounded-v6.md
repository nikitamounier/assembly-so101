# Active accepted-state hybrid run-v6

Run: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4/run-v6`.

25prototypes+874accepted TRAIN variations,154EVAL held out,1143task-conditioned rows. All reset observations finite; controls/velocities/warmstarts/friction preserved; native-vs-admission error0. Frozen model8c4783 remains unchanged. This is minimum-valid diagnostic training; the1800variation build and full task admission are separate. [Detailed release](v6/README.md).

Checkpoint13/269713transitions preserved exactly (policy/Adam/lr/counters);213identical SGS histories retained/930new rows. Migrated checkpoint652986897dc56afa83363184125556ae04dead21b0968f31e6c9c2c741be0069. Old learner/controller/40worker leaves closed, all leases retained.

5090configured environments/509slots,32steps=>162880target transitions;5epochs/4PPOminibatches,1024gradient chunks.400baseline Chimera schedulerCPUs including pending/requeues; protected additive growth toward1000. One GPU3598717 with approved idle3606264fallback. No SAIL.

Modal replacement appap-n2zsppGvqHCVbd2LaW89nM requests4092physicalrolloutcores,68x60+12. Packed12-worker finite-control gate passed; provider admitted all4092rolloutcores at00:26UTC (68large+1remainder; zero container/input errors), all409workers passed finite first control steps by00:28UTC. DO NOT count queued/dispatched or cumulative first-control workers as current active capacity. `cloud-status.json` records provider counts; current learner masks give actual environment utilization. Old appap-TbTXOSC4tVHGPRUZCV8lly hit100USD usage cap and was stopped; remaining425credits verified/usage cap raised500. Replacement3600s bound (~219–245USD/hour envelope), conservative expiry Nov8 01:07:37UTC; no automatic renewal or upgrade.

The learner's bounded authentication/initial admission fixes prevent stale worker handshakes from delaying recovery indefinitely. Physics, observations, worker protocol, PPO and step/observe code are unchanged. Passive recorder consumes already-transferred CPU poses with bounded background writes. Actual files live under run-v6/training-replays. First complete training failure replay has been verified through NPZ, HTTP200 delivery/cache and browser playback; successes remain absent until observed. Dashboard http://localhost:8881/ discovers activated versions; ROOT/v6 aliases isolated ROOT/ops-v6 operators for its accounting.

```sh
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py read
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py failover
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry_v6.py scale
/tmp/assembly-modal-sdk/bin/python operational/wowrobo-chimera-20261107/modal-live/cloud_status.py
```

Scale only after actual new iteration>=14 with finite/source-valid progress age<=1200s. `ops-v6.json` contains exact pins/job identities. The shared heartbeat remains ACTIVE with original unrelated pause/retirement contracts preserved. Use `start_v6.py` only if the current controller owner is absent; exclusive locks prevent duplicates. Do not restart paid pools automatically, hotload states, cancel leases or mutate the frozen physical source. [Historical v5 guide](README-v5.md).

First actual v6 update14 committed348513total/78800valid transitions with finite losses,799.3srollout/1.99sPPO includingfleet ramp. Baseline400Chimera+4092Modalphysicalcores arelive; greedycheck foundno<=180sadditionalcapacity andcontinues. This remainsdiagnostic:84.5%safetyterm;10instabilityflags plus one256MBconstraint-arena warning onworker3602575. Guard endsaffected slices; noarena/physicschangehotloaded. A separatelyqualifiedcapacityincrease needsmemory/parityproof.

Independent checkpoint loadPASS: new iteration14/348513transitions,SHA dd12737ddcfb4be1093d09ba8fdd770691e36c4679bbbbf5ceadf41f187bf0cd, finitepolicy/Adamtensors and1143-entrySGS/model/bankidentity verified. First full-fleet steady throughput is not yet measured:32steps at currently4490connected-capacity envs gives143680nominalvalidtransitions, while5090configured slots reserve162880targetforfutureChimeragrowth.
