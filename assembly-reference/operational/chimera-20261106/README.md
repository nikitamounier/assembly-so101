# Chimera assembly run, November 6, 2026

Source base:85c006a, local inherited-scale configuration changes. Remote root:
`/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a`.
Run:`runs/chimera-v13-8192-20261106`.

- Retained GPU allocation3598717:quick_preemptible,1GPU,8requested CPUs,64GiB,2h,requeue,USR1 warning300s. Training failures leave the allocation available.
- CPU allocation3598730:cpu_preemptible,500CPU tasks (508CPUs actually assigned on the first lease),2GiB/CPU,2h,requeue,USR1 warning300s. Initial faithful 500–1500 step10 sweep found720largest immediate request; subsequent queue shifts moved720to17:16:05Pacific. Actual start must be verified.
- Hold scripts requeue at warning; controller requests a graceful learner checkpoint fifteen minutes before either lease ends, then waits for renewed running leases. Ten-update checkpoints also protect preemption. Same job ID does not guarantee the same physical GPU after requeue.
- The initially requested720CPUs did not start; explicit preemptQoS/flexible nodes and reduction to500tasks started at17:17:54Pacific on17nodes. No other jobs were canceled.
- Controller and all observations execute in explicit srun steps under existing CPU3077705. Setup, generation, preflight and learner execute underGPU3598717; rollout workers execute underCPU3598730. Login only handles scheduler queries/submission and transport.
- Native v13 explicit all-loose bank:160TRAIN/160EVAL, generated locally; not the unfinished staged-subtask bank. No old-v11 prepared data relabeled.
- Trainer:8192env/32steps/10000iterations/500remote-workers,all-workers required at initial acceptance,CUDA,448points,nativeCPUphysics,safety1,contact0.4mm,SGSeps0.01,seed20261105,save10,no background eval. Inherited PPO hyperparameters retained.
- READY marker requires strict preflight plus actual CUDA smoke pass. No startup bypass.
- `controller.py` uses an exclusive kernel lock; child monitor observes and reports rather than modifying source/science or blindly restarting unknown processes. Parent owns fixes.
- `observe.py` emits native job states, source hash checks, actual update/log/checkpoint freshness, controller phase and errors. It excludes server.json authentication keys.

Cross-allocation step launches clear inherited SLURM_/SRUN_ binding variables and CUDA_VISIBLE_DEVICES before invoking srun with the target job ID. Initial launch errors are preserved in logs; no optimizer updates were lost. Directory relocation verified without an old-name symlink. Strict preflight31/31 andCUDA learner13/13 pass. Host model identity:abc9a9fcaec52d0db9d1c43b0238df66447d9915b7dc618eba9b3fe63cdb24c4.

Memory repair:500single-thread processes hit two nativeOOM events. The learner saved a durable iteration0 checkpoint using its inherited KeyboardInterrupt handler; only the old worker step3598730.0 was canceled. Retry uses250processes x2threads to reduce replicated Torch/SDF overhead, same500CPU allocation,8192environment target, source/physics/bank and GPU. CPU lease requeued once and was BeginTime pending for17:27:16Pacific at the last observation. First full PPO update remains unverified; no learned success claim.

## Independent-node pool (current)

At the human's explicit instruction, checkpoint12/2,887,384transitions was saved and the learner exited before the old gang3598730was canceled. Its former pool and first failed node attempts3600167–3600170 are historical.

The latest per-node screen selected400CPUs on4hosts; each actual job started18:41:58Pacific with its own2hlease,requeue and120sUSR1warning:

| Job | Host | CPUs | Two-thread workers |
| --- | --- | ---: | ---: |
|3600462|GPU36DC|140|70|
|3600463|GPU71E4|90|45|
|3600464|GPUC960|90|45|
|3600465|GPU1112|80|40|

Manifest:`artifacts_local/chimera/node-pool.json`; projections:`per-node-projections.json` beside it.200workers retain the8192environment target and all physics/PPO/bank/observation identities. Node jobs run onlyCPU tasks, even onGPU-named hosts. Each node requeues independently; `pool_controller.py` leaves the learner and remaining node jobs running. Admission allows50percentofplannedworkers after60s rather than waiting indefinitely for every node; active environment counts/valid transitions must be reported explicitly. All-worker loss still checkpoints and restarts; GPUlease expiry retains its previous checkpoint/requeue handling.

The first node scripts failed before worker startup because inherited SLURM_MEM_PER_NODE conflicted with job SLURM_MEM_PER_CPU. Fixed by clearing submission SLURM_/SRUN_/SBATCH_ variables and cleaning memory/binding fields inside each batch job; exact failed receipts retained. Source hashes include the new coordinator/node script/planner/dynamic observer. Remote inspection and orchestration remain insideCPU3077705. SameGPU3598717retained.

## Ten-CPU jobs (current, supersedes per-node allocation sizes)

Human requests10CPUs per job. Verified learner idle and durable checkpoint14/transitions3,144,389 before canceling old4jobs3600462–65. SameGPU3598717 retained (currentleaseGPU3B62).

Submitted40single-node/unpinned jobs3602571–3602610; each5tasks x2CPUs=10CPUs,20GiB/two-hour lease/requeue/warning120s. Pool target400CPU/200workers/8192env. Manifestgeneration20261107T052640Z at the same node-pool.json path; jobs may share hosts. Startupfloor5workers (one10CPUjob),accept30seconds/min-conn-frac5/200. This fixes the old50percentadmission deadlock that left180liveCPUunused below200CPUthreshold. Actual valid transitions/activeenvcounts remain distinct from8192target. PPO/physics/bank/observations unchanged.

Monitor handoffs now use minimal generic wakeups without detailed job/path/training payloads, after the old detailed capacity handoff was rejected by automatic approval review. Preserve old rejected/uncertain delivery records and full local receipts; parent reads evidence locally and owns repairs.

## Shared relay requeue recovery (November7)

The shared relay3077705 can requeue independently of the GPU and worker jobs. Requeueing its batch script does not restart an attached pool controller. The monitor detects this as `controller_stale`; parent verifies the relay epoch, controller lock, learner absence and durable checkpoint before restarting inside the renewed relay. Do not cancel or edit the protected relay.

`recover_relay.py` is a one-off guarded recovery for the November7 episode (checkpoint42, GPUepoch5, relayepoch45), not a general blind restart. It refuses active numeric GPU steps, an occupied controller lock, unchanged GPU epoch, stale source hashes, or checkpoint/log disagreement. Runtime source and source pins were unchanged. GPU3598717 and forty worker jobs3602571–3602610 retained; normal30second monitoring continued.
