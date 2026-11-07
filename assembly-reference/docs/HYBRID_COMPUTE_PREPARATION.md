# Main-run hybrid CPU preparation — 2026-11-07

**Current numerical target:** **4,092 Modal CPU units + retained cube CPU allocations + additional greedily acquired Chimera CPU allocations**. This supersedes the previous4,096cloud-CPU target. Additional Chimera capacity is on top of the retained baseline. Modal physical-core/vCPU interpretation, provider quota and learner-slot/throughput qualification remain unverified; no resources are activated by this target update. Historical calculations below retain their original context.

**Latest CPU ownership instruction:** When the main run starts, checkpoint and stop cube training processes, retain its CPU allocations, and transfer allocation/worker renewal ownership to the main run. Add further CPU jobs greedily without displacing owned work. Cube continues for now. Stopping an experiment is not permission to cancel its CPU allocations.

**Latest provider decision:** Modal only for paid rollout CPUs, plus greedily admitted Chimera capacity. SAIL is deferred until the team obtains a concurrency increase. The4,096cloud-CPU target remains a planning target requiring Modal unit/quota/throughput verification. Prior mixed-SAIL examples below are historical analysis. The disabled hybrid template now omits SAIL; no resources or paid subscriptions were started.

User target updated November 7: 4,096 concurrent cloud CPU units combined across SAIL and Modal, plus greedily admitted Chimera CPUs (400–500 baseline). Physical cores, vCPUs and worker threads must be normalized before provisioning; quotas, cost and end-to-end scaling remain unverified. The earlier 512-slot templates below are historical illustrative configurations and cannot admit this larger fleet; no template or live scaler was activated by this target update.

User budget assumptions: $1,500 aggregate Modal credits and $1,500 aggregate SAIL credits across three people. Only this account's credentials are verified. The cube and paused assembly experiments will not compete with the new run at launch; this preparation does not terminate either one.

## Implemented and verified

- Existing shared native-worker runtime in sibling `assembly-so101/assembly_policy/compute` supports Chimera, Modal and SAIL. Persistent authenticated TLS connections carry compact observations/actions; the bank, SGS sampler and policy stay on the learner. No cloud API on the control-step path.
- Added `python -m assembly_policy.compute hybrid --config CONFIG --ticket PRIVATE_SERVER_JSON` to supervise multiple pools. `--dry-run` prints commands without starting anything. Planned process counts cannot exceed learner slots; declared budgets aggregate by provider. These cost declarations are estimates, not provider billing enforcement.
- Modal requests now set CPU and memory upper limits as well as reservations. SAIL tickets are explicitly written mode 0600. Termination invokes provider cleanup; cloud runtime/lifetime limits remain a fallback after launcher death. Provider SDK versions verified locally: modal 1.6.3, sail 0.17.2; `requirements-providers.txt` pins them. The installed SAIL SDK still uses size labels even though current website documentation also describes vcpu_count. Do not silently mix incompatible SDK arguments.
- Both provider credentials passed read-only app listings. SAIL key is in the private workspace `.credentials/sail.env` (0600, directory 0700), outside both source repositories. Modal uses the existing local profile. No keys are in source/configuration/receipts. No new cloud compute has been launched.
- Added an allocated Slurm sidecar, source SHA 00c0bb42afccae9053a23f967cde3bd1b48aad2c807ba2359d6bd9f3a315b851. Remote `/large_storage/goodarzilab/nikitamounier/assembly/hybrid-autoscale-20261107.py`; state directory same basename without `.py`. Existing monitor allocation3077705 is used without resource changes.
- Sweeps CPU1,10,20,...1000 AND literal node1,10,20,...1000. Partition has29 physical nodes; larger node requests are classified impossible without an RPC. Every candidate excludes ALL hosts with any running job owned by the user. This is conservative: safe spare cores on those hosts are excluded too. Forecasts are advisory; ten-CPU submissions are separately probed with their exact single-node shape and fresh exclusions immediately before submission.
- Sweep is refreshed at most every5 minutes within the existing30-second heartbeat. When a NEW qualified run is registered, additive jobs are allowed for starts within180 seconds, up to configured CPU/worker-slot limits and10jobs/cycle. Pending capacity counts. Durable pre-submission intent blocks retry after an ambiguous submission. The sidecar never cancels jobs.
- Current mode is observation only because there is no active-run registration. First allocated full sweep produced101 CPU rows/101 node rows, no near candidate, and no submission. Full receipt: sibling `assembly-so101/operational/hybrid-20261107/initial-sweep.json`. Local hash-bound monitor entry: `operational/hybrid-20261107/cycle_once.py`; unique raw receipts retained alongside.
- Existing heartbeat `monitor-assembly-chimera-training` updated through the automation tool, preserving all previous prompt bytes, target thread, status and cadence. Prep scope remains independent of eventual old-run retirement. Full readback verified. Genuine scheduled adoption verified: the child executed the exact new launcher on its scheduled turn, with raw receipt20261107T152550.310018Z.json / SHA0e8c4800505924f1e0d9d787e1f64879f9453109cd2fa40979255dfa03707c19, exit0, observe-only, no submissions.

## Batch size and concurrency

| Setting | Main inherited target | Cube trial |
| --- | ---: | ---: |
| Environments N |8192|1024|
| Steps/environment T |32|32|
| Full rollout batch N*T |262144|32768|
| PPO minibatches/epoch |4|4|
| Samples/optimizer step (full occupancy) |65536|8192|
| PPO epochs |5|5|
| Optimizer steps/rollout |20|20|

Each transition is one environment/control step, not a complete assembly episode. The rollout is reused five times. A microbatch chunks a single optimizer minibatch for GPU memory, accumulating gradients before one optimizer step. It does not change the statistical batch. At the main defaults, the65536 microbatch fits one full optimizer minibatch; smaller chunks can preserve its gradient normalization. The new physical environment's memory footprint remains unmeasured.

Crucial implementation limit: RemoteVecEnv fixes NCONN slices at learner startup. Live worker processes fill those slots; late/preempted slices are inactive, with invalid transitions dropped. Thus the maximum B is N*T, while actual B is the number of valid transitions over the32 steps. Adding CPUs fills inactive slices and can increase actual B until target occupancy. It does NOT continuously rebalance all8192 environments over an arbitrary worker count. Adding jobs beyond the configured slots is useless and is blocked by the scaler. Autoscaling is operationally elastic but is not exact fixed-batch invariance during worker loss/join.

With the historical200 slots x2 threads,400 allocatedCPUthreads fill all slots. A proposed512 slot learner permits512 processes/1024 allocatedCPUthreads and16 environments/slot; its full batch requires512 connected slots. At only200–250baseline processes,3200–4000 environments are active and the upper bound per rollout is102400–128000 validtransitions. The illustrative hybrid template adds200cloud processes: with the400 CPUbaseline this yields400/512 slots,6400 environments and204800 validtransitions; with500baselineCPUs it yields450/512 slots and230400 validtransitions. These are exact steady-occupancy counts, not throughput measurements. Either provision all slots, choose a smaller cap before launch, or explicitly accept/log the ramp in effective batch. Do not advertise262144actualsamples before occupancy proves it.

There is no batch-derived maximum count of billed CPUs across providers: N=8192bounds independent environment simulations, NCONN bounds this implementation's worker processes, threads per process bounds allocations, and GPU update time, serialization, WAN latency, RAM, quota and budget bound useful scale. At one environment/process, extra native threads cannot split that single MuJoCo environment. The mathematical8192 environment bound is not a recommendation to buy8192 cores.

Recommendation: keep N8192/T32/epochs5/minibatches4 initially; select the worker-slot cap with the measured hybrid fleet, record actual B, and benchmark complete iteration time at400,500,750,1000 CPU-thread plans. Only then compare N16384 at matched transition and wall-clock budgets. Doubling N doubles rollout B; holding4 minibatches doubles samples/optimizer step. Holding minibatch size instead requires8 minibatches, yielding40optimizer steps/rollout. More rollout steps changes temporal credit-assignment/truncation and is not equivalent to merely adding environments. No automatic LR scaling is justified by CPU count.

## Upstream evidence

The [AssembleNIST README](https://github.com/suniique/assembly-nist) reports8192 environments, oneGPU learner/960 CPUworker processes/about30 nodes and roughly26 hours for10000 iterations; it offers512 environments for smaller machines. Its failure discussion emphasizes exploration/entropy/std, wrist action scales and contact physics/timestep. It does not document a batch-size sweep or claim8192is optimal for our bimanual task. The [trainer source](https://github.com/suniique/assembly-nist/blob/main/assembly_policy/train.py) supplies32 steps/5 epochs/4 minibatches/65536microbatch, and [worker script](https://github.com/suniique/assembly-nist/blob/main/scripts/slurm/workers.sbatch) uses one process/core in its reference cluster. Our two-thread workers and different geometry are separate choices.

## Credits, units and limits

[Modal pricing](https://modal.com/pricing): ordinary Functions $0.04716/physical-core-hour plus $0.007992/GiB-hour, before region premiums/egress/storage. A physical core is marketed as2 vCPUs, but equivalence is not measured MuJoCo speed. Sandbox CPU/RAM prices are3x ordinary Functions, so the adapter uses Functions. Starter currently lists100 containers; our25 container example fits that published limit.

[SAIL pricing](https://docs.sailresearch.com/sailboxes-pricing): $0.015/used-vCPU-hour, $0.008/used-GiB-hour, $0.0007/used-NVMe-GiB-hour, plus creation/volume charges. Memory includes guest kernel/filesystem cache. [Plan limits](https://docs.sailresearch.com/pricing) list200 reservedvCPUs/Free organization; credits do not lift this quota. Three distinct eligible Free organizations could provide600 reservedvCPUs in aggregate, but three people sharing one organization still share200. No other person's account/key/credit balance has been verified.

Illustrative200unit pools, assuming sustained full CPU use and2 GiB RAM/unit:

| Pool | Compute+RAM/hour | $1500 duration |
|---|---:|---:|
|Modal200 physical cores|$12.63|118.8h|
|SAIL200 vCPUs|$6.20|241.9h|

These units are deliberately different. Rates omit region premiums, network/disk/storage, image-build/startup and creation overhead. Our baseline Chimera400–500 scheduler CPUs is a scheduling assumption, not400–500 physical Modal cores. Measure native steps/second and complete PPO seconds/iteration per billed dollar before deciding the split. Combined credits are a planning envelope, not proof balances are consolidated or all quotas available.

## Launch handoff and remaining qualification

1. Finish/qualify the new WowRobo physical factory/reset bank in the canonical repository; freeze source, bank, observation and model identities. The existing worker package builds historicalSO101 geometry; it has NOT qualified or integrated the still-changing new photo-based target. Port the provider transport/orchestration to that frozen runtime or make an explicit compatible release. Never substitute old geometry.
2. Establish a reachable authenticated TLS endpoint. Current private Chimera hostnames/server tickets are NOT an established public cloud route. Run the existing persistent batch echo probe FROM one actual Modal worker and one actual SAILbox, then native-worker/full-PPO parity against the same frozen environment. Cloud credentials alone do not prove connectivity or speed. No wide public listener, network-security change, or credential redistribution was made here.
3. Select regions from measured RTT. The architecture currently synchronizes every control step; a WAN round-trip of20ms imposes at least0.64s per32-step rollout,100ms imposes3.2s, before compute. Heterogeneous slow workers can dominate the barrier. This is why more CPUs do not guarantee a faster iteration.
4. Register a fresh run and a chosen NCONN. `configs/hybrid_main.template.json` and `configs/autoscale_main.template.json` are disabled templates, not active jobs. Fill real paths/regions/worker-scriptSHA/base_job_ids, account for cloud workers including pending starts, and write the latter to the remote scaler state directory as `active-run.json`. Enable only for the qualified NEW run. Checkpoint and stop cube application processes through their owning controls, retire only cube monitoring, and KEEP its CPU allocations/holding steps. Adopt those leases and their renewal/requeue ownership for the main run, verify that all newly attached workers use the frozen main-run source/ticket, then register the adopted IDs as base_job_ids before enabling additive submissions. Do not cancel/release/reacquire them. The current observation monitor does not implement this handoff.
5. Run a bounded mixed-provider complete-PPO iteration and checkpoint-resume, compare valid sample counts/losses/environment identity, then enable additive scale-up. Cloud plan costs are estimates; use actual billing/quotas. Pending jobs must not be counted as connected workers.

Preparation is implemented; a production mixed-cloud rollout and automatic submission on a registered new learner remain unverified. No main training run was launched or old experiment stopped.

Focused verification:14tests passed across autoscaling, hybrid-plan limits, provider cleanup and TLS transport; Ruff passed. Provider APIs were authenticated read-only; cloud execution/native cross-provider parity remains pending.

## Account check — November 7

Read the authenticated SAIL Billing page for organization `nikitamounier`: plan **Free**, remaining balance **$501.00**, entirely free credits; purchased credits $0.00. The page offers Pro at $250/month with up to 10,000 reserved vCPUs. No higher/custom quota is displayed. Combined with the published Free-plan limit, use **200 reserved vCPUs as this organization’s working cap** unless SAIL confirms an exception. The billing tooltip says configured vCPUs count across running Sailboxes and in-progress starts/recovery, regardless of CPU utilization; there is no separate box-count limit. No subscription, purchase, resource creation or account setting was changed.

## MuJoCo Warp rollout economics — November 7

User asked CPU/GPU rollouts per dollar before choosing cloud fleet. Closest published NVIDIA SO-101 physics-only test: RTX PRO 6000 Blackwell workstation, 8192 worlds, ~4.924M world-steps/s; single Threadripper core46–60k =>82–107 CPU-core equivalents. Fixed-control CUDA-graph physics microbenchmark; excludes full task/SGS/policy/PPO, not our assembly workload and not a SAIL host benchmark. Source: https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/main/tutorials/sim2real-blogs/notebooks/mujoco/README.md . Its linked full-task BENCHMARK.md was unavailable/404, so no full-task ratio claimed.

No public SAIL GPU rental price/API option found in official pricing/SDK docs; cannot provide a measured same-provider GPU/CPU ratio. Illustrative Modal RTX PRO6000 GPU line item $3.0312/h vs SAIL $0.015/vCPU-h: GPU must match202.08SAILvCPUs for compute-only break-even. With assumed2GiB RAM/vCPU at$0.008/GiB-h, threshold97.78SAILvCPUs; GPU-host CPU/RAM/network/region fees excluded and raise this threshold. These are arithmetic thresholds, not core/vCPU equivalence assumptions. Sources: https://modal.com/pricing ; https://docs.sailresearch.com/sailboxes-pricing .

Same-modal compute-only comparison: ~$3.03/GPU-h buys64.27Modalphysical-core-hours at$0.04716/core-h. If its CPU cores matched the benchmark core and rented GPU matched workstation GPU throughput, measured82–107core equivalents imply1.28–1.67xGPU samples/dollar before othercharges. Neither hardware equivalence was measured. Do not assume10xGPUeconomicgain or replace4096cloudCPUs with arbitraryGPUcount.

MJWarp optimizes batched throughput, can lose single-world latency; our8192env batch is promising but needs actual paired fullrollout/PPO benchmarking. GPU float32/contact/solver differences require physical acceptance checks; customC++SDF/contact code needs an explicitWarp implementation, not a backend flag. Current docs describe Warp SDF hooks, so do not repeat old blanket “SDF unsupported” claims. https://mujoco.readthedocs.io/en/latest/mjwarp/ . No backend/source/livejob/spending change made.

## Account-specific GPU check — November 7

Authenticated SAIL Apps UI now directly displays **Reserved vCPUs 0 / 200**, with0activeSailboxes/noapps. Billing remainsFree/$501. No GPU allocation option or GPU-hour price is exposed in this account’s Apps/Billing navigation. Installed sail0.17.2 hardware catalog contains only CPU/RAM/disk sizes: s1vCPU, m4vCPU, l8vCPU, ultra_experimental64vCPU; no GPU shape. This is not a denial of private/custom GPU arrangements, which remain unverified. Do not quote a SAIL GPU price or treat inference-model access as arbitrary MuJoCo GPU rental.

Additional launch issue revealed by the hardware catalog: previous disabled SAIL template uses m with4workers×2threads=8threads but m exposes4vCPUs. Before activation, use a verified8vCPU shape(l in pinnedSDK), or reduce workers to2; add CLI capacity validation. Nothing has been launched, and this mismatch is not a live oversubscription incident.

## Retained cube CPU handoff contract

The current local cube cluster config lists35ten-CPU allocations (350schedulerCPUs). This is a configuration snapshot, not a fresh live Slurm inventory. At launch reread the current config and verify exact running/pending job IDs, hosts, resource counts, restart epochs and ownership; do not assume this list remains current or include other scopes' donated jobs.

Preserve durable cube policy/optimizer/sampler/checkpoint evidence, then stop only its own controller/learner/worker processes after guarding against cube auto-restart. Preserve the original batch/holding steps that keep leases alive. Transfer renewal/requeue behavior so new lease epochs launch only main-run workers rather than resurrecting cube or historical assembly workers. Retire only cube monitor conditions after the intentional stop. Main-run coordinator must be the sole owner of retained worker admission/restart and new CPU submissions. Distinguish scheduler allocation counts from connected workers and actual valid PPO transitions.

The disabled hybrid template now removes the fresh Chimera-submission pool and records an explicit retained_chimera handoff descriptor with the last-known35IDs and verified_live=false. The generic chimera_pool helper creates and ultimately cancels its own jobs, so it must NOT manage this retained baseline. Actual retained-worker adoption and renewal transfer must be implemented/verified at launch; descriptors alone do not perform it. Additional jobs use the dedicated scaler after adopted IDs are registered. No current process/job/monitor/renewal/source/registration was changed for this note.

## Monitor, dashboard and latest CAD launch contract

Human reaffirms greedy Chimera growth through the SAME monitor. Updated automation by appending exact4,092Modal + retainedcubeCPU + ADDITIONALChimera ownership/admission requirements. Fresh readback verifies all prior57111prompt characters preserved, including currentcubev8 and completed-benchmark retirement; new59477characters/status/cadence/target verified. No active-run registration, scaler/sourcepin, allocation or spend change. Greedy submissions require separately registered main-run capacity and fresh progress.

Main-run dashboard follows `cube-dashboard`'s actual return/reward/success selection, family cards/charts, reset/reward version markers, reporting windows and latest REAL successful selected-family replay with play/scrub. Use actual new assembly tasks/reset families; show progress/actual valid samples, Modal/retained/additionalChimera resources and runCAD/source/bank identity. Missing/stale/wrong-family clips never stand in for current success. Fresh cube reference-chat read reports v8 with11families, checkpoint8756/resume8784 and35CPUgroups; this is chat-reference evidence, not current live Slurm validation. Dashboard requirement is saved; new dashboard is not yet built.

Latest approved CAD is a LAUNCH requirement: inspect canonical source and current CAD/config manifests, including approved local work newer than remote HEAD. Use new approved photo-basedWowRobo target, not historicalHiwonderv13. Regenerate affected geometry/collision/reset artifacts and validation receipts; do not relabel old banks/checkpoints. Qualify exact model/observation/bank/validator identities, freeze in launch manifest, verify every worker bundle and display identity in dashboard. Later CAD edits do not silently enter the live run. Cube remains running during preparation; retain/adopt its CPU allocations only at main launch as already specified.

## November 7 mixed-provider smoke and latest-CAD audit

The earlier pending infrastructure smoke is now PASS: Modal and Chimera workers jointly completed two PPO updates (512 valid transitions), eight reset checks, exact local parity and exact checkpoint resume. Both temporary Modal apps are stopped with zero tasks. Full evidence: `outputs/hybrid-readiness-20261107/README.md` and the linked smoke result. This does not qualify current CAD.

Latest repaired CAD/reset review remains BLOCKED: zero physically admitted resets, nine nominal geometry failures, settling and loaded support-grip failures, and no current-WowRobo trainer integration. Main template remains disabled and now references the hashed blocked qualification receipt; live supervisor launch requires qualified evidence and unchanged artifacts. Four focused launch/plan tests pass.

Native current-CAD timing records160substeps/action versus cube20. Split-step overhead is approximately noise to4% in two bounded states; caching friction roles gains0–3%, so physics was not changed. Tiny cube WAN smoke performance is not a full-fleet or current-CAD speed qualification. Keep the4,092Modal + retainedChimera + additionalgreedyChimera plan pending physical/backend/throughput admission.
