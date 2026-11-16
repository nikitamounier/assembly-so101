# Accepted-state hybrid continuation

Active run: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4/run-v5`.

The latest request enables Modal alongside retained and greedily acquired Chimera CPUs, and increases useful environment capacity. The earlier Modal-off instruction is superseded. SAIL remains deferred. This remains a diagnostic run while full task/production admission is unfinished.

## Frozen state bank

Snapshot `states-20261107T225408Z` contains all25 minimum-valid prototypes and244 accepted TRAIN variations from the corrected scaling-states build.33 accepted EVAL variations remain held out. Rejected and unfinished draws are excluded; the1,800-state generator continues separately. The bank has513 task-conditioned rows:65 cover,65 connector,65 pin,187 upper and131 forearm. The common loose start/variations are conditioned on each of the five tasks; these are not additional physical draws.

`split-audit-v5.json` verifies no exact duplicate state/friction draws within or across the two variation splits. New native payload162 contains full physics, equality activation, actual motor commands, initial solver warmstart and three episode friction multipliers. Velocities are retained. Motion is allowed; the user minimum requires no unintended material overlaps and retained held parts.

The per-world friction hook runs in compiled C++ before force solving, with no Python callback in rollout substeps. It reconstructs the same friction constraints as the admission simulator; nominal multipliers retain the original fast path. Frozen native identity:

`8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`

Verification:513 finite reset observations with exact controls/warmstarts/friction (349 identical earlier witnesses reused,164 new rows checked); threaded native versus admission parity error0 for loose and held upper21; separate two-world/100-substep toy parity error0; H100 inference5090 and gradient chunk1024 used148MB peak allocated memory. The isolated parity/shape probes do not update the training policy.

## Checkpoint and compute

Old `minimum24/run-v3` stopped after checkpoint8/102573 valid transitions. Its sole learner/controller and40 owned worker leaves were positively closed; all CPU/GPU allocations remained retained. The new checkpoint preserves policy, Adam, learning rate and counters exactly, with fresh513-entry SGS history. Migrated checkpoint SHA:

`1af5a9bf8c22efe57daed18f679c2ebdd4c7da9564ab63ccc0df02c43aae3642`

Capacity:509 worker slots/5090 environments;32 control steps gives162880 target samples.5 epochs/4 PPO minibatches are unchanged. Microbatch1024 bounds GPU memory through gradient accumulation. Missing workers contribute no valid samples. Initial connection fraction.05 permits restart on Chimera alone.

- Chimera:40 retained ten-CPU leases=400 scheduler rollout CPUs, including pending leases. A separate greedy scaler has1000-CPU/100-process capacity,409 reserved cloud slots, near-start<=180s, five-minute forecast cache, pending/uncertain-submit accounting, own-host protection and faithful non-submitting job checks. The first valid scale cycle found no eligible near-start capacity and submitted nothing.
- Modal target:68 containers×60 physical cores plus one12-core container=4092 rollout cores,409 worker processes. Each large container runs six ten-thread workers; memory request48GiB/hard limit96GiB. A separate2-core broker provides outbound-only Chimera connectivity. The packed smoke gate requires all12 workers in the initial two containers to return finite real control steps before remaining calls are dispatched. This passed.
- Running versus queued capacity is reported separately in `cloud-status.json`. `active.json` readiness counts are cumulative and do not alone prove a worker is still running. Modal can take several minutes to place these large CPU containers. A fixed minimum of68 large containers and one remainder container now requests the full authorized pool directly.

Modal workspace verification: Starter100-container limit and529.99 available credits. The current pool is bounded7200s; the resource envelope is about219–245USD/hour before small startup/idle charges, within that account's credits. Team aggregate1500credits remain a planning assumption, not verified access in this workspace. The shared heartbeat reports before pool expiry; it does not renew paid compute automatically. Chimera continues when cloud workers leave. Prices: [Modal](https://modal.com/pricing).

Current mixed-provider evidence: iteration10 committed42620 valid transitions,67.43 valid transitions/s, finite losses and no native instability. This includes startup/ramp time and is not a steady full-fleet benchmark. Most terminated episodes hit safety stops in the harder expanded state distribution; this is not evidence of broad task learning. The separate six-case zero-action insertion audit passed: no safety stops/instability/native warnings, penetration0.13–0.15mm versus0.5mm limit. This covers representative upper21/forearm29 prototypes and variations for one40ms control interval, not every state or the full episode horizon.

## Operators

Run from the canonical repository. These wrappers use the pinned remote operators inside allocation3077705; credentials and tickets are never printed.

```sh
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry.py read
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry.py failover
python3 operational/wowrobo-chimera-20261107/modal-live/monitor_entry.py scale
/tmp/assembly-modal-sdk/bin/python operational/wowrobo-chimera-20261107/modal-live/cloud_status.py
```

Scale only after actual new log iteration>=9 and fresh finite/source-valid progress; inherited checkpoint8 is not new SGD. Same heartbeat remains ACTIVE; unrelated PRIMARY pause and CUBE/benchmark retirement are preserved. GPU3606264 is the learner, with explicitly approved idle3598717 fallback and inherited exclusive locks preventing duplicate learners. `ops-v5.json` contains exact job IDs/operator pins.

Transport: certificate-pinned TLS on both public legs, end-to-end learner mutual HMAC, and native/TLS framing conversion without gateway unpickling. The broker's credentials/certificate are ephemeral. Cloud code/assets contain no account API keys. The exact frozen runtime export SHA is `2c991a016a71976d203b94440ce36e9d2178ace1c40890dda8adcc7cbcfe249b`.

Earlier failed startup pools were stopped and reconciled to zero tasks. A TLS8-byte/native4-byte framing mismatch was fixed and verified with real HMAC and2MiB roundtrips. The successor also drains failed/closed handshakes within a step. Cloud container identities now use per-call UUIDs, because Modal containers can share hostname and process IDs. These operational fixes do not alter training physics or PPO objectives.

Recovery controller (only if its existing owner is absent; exclusive locks enforce ownership):

```sh
python3 operational/wowrobo-chimera-20261107/modal-live/start.py
```

Paid pool launch, only after the previous cloud app is confirmed stopped. This is the command used for the current bounded pool; do not launch a parallel duplicate:

```sh
ASSEMBLY_MODAL_TARGET=operational/wowrobo-chimera-20261107/modal-live/release-target.json \
ASSEMBLY_MODAL_RUN=run-v5 ASSEMBLY_MODAL_FULL=1 ASSEMBLY_MODAL_SECONDS=7200 \
/tmp/assembly-modal-sdk/bin/python operational/wowrobo-chimera-20261107/modal-live/launch.py
```
