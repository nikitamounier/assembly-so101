# Current WowRobo Chimera diagnostic — November 7

**Authoritative directory: `run-v2`. The earlier `run` startup is INVALIDATED.**

Remote root: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z`.

The user explicitly starts learning while CAD/reset fixes continue, following the getting-started chat handoff, with maximum useful Chimera CPUs and no Modal. This accepts the labeled diagnostic bank, not physical reset admission.

## Frozen training setup

- Current native WowRobo, all5tasks, real gravity/source contacts/passive stops, sparse solver,250µs physics/40ms control,448body-attached points,12actions. Reward+1 after0.4s stable seating within0.5mm/1degree, separate safety/timeouts. Original laws/solver/timestep/tolerances unchanged.
- Diagnostic training-debug-v1:47TRAIN rows,61records including eval, all explicitly unqualified. Final-module task uses existing all-loose starts with installed equalities off; no fabricated staged prerequisite. No local or remote production-qualified label was granted.
- Model identity `7b159e7d452eb9e598acca9044593ecb35d2745dd1d5bb9f6356ea5b84559aef`. Snapshot/deployment-v2.json +204active-input hashes. Later local changes never hot-load. Local diagnostic manifests may need rebuilding under later canonical changes; the frozen remote bank is independently bound.
-400retained assembly schedulerCPUs in40ten-CPU leases, including the5previously donated back to intentionally paused PRIMARY. All baseline IDs in cluster.json; pending/requeued counts are distinct from connected workers. Preserve original batch/holding processes. No allocation canceled, released or reacquired; other projects untouched.
- One10-thread worker/process per ten-CPU lease.100fixed connection slots/1024envs/32rollout steps =>32768target transitions/update; actual valid samples reflect joined slices. First35workers admitted358environments; remaining5join between steps. One GPU learner on already-used3606264; GPU3598717 was not repurposed.
- Fresh policy/optimizer/sampler;10000-update ceiling, checkpoint each update. The learner and every retained worker inherit exclusive locks across exec. Actual /proc inspection proves1learner and1sampled worker, matching lock descriptors and duplicate-lock rejection; ownership-proof-3606264.json and ownership-proof-3602598.json.

## Startup corrections and evidence

Portable archive dereferences native grid symlinks and rebinds absolute paths/hash references. Full Mac/Linux compiled geometry comparison: max9.37e-20mvertices/2.22e-16mgeom positions/1.50e-15quaternions, exact topology/indices.52observation mesh fingerprints were regenerated for platform roundoff only. platform-rebinding.json records the audit; no mesh or tolerance change.

Training uses accelerated exact surface-tree mode1, not exhaustive diagnostic oracle2. Current native workers require severalGiB; five processes per20GiB lease OOM. One threaded process fits. Native arena increased64→256MiB after constraint-capacity warnings; detected native warnings terminate conservatively, rather than allowing rewards from truncated constraints. This changes capacity/guard identity, not contact law.

The invalid first startup left two learners alive after launcher SIGTERM. Those logs/checkpoints are preserved under `run` with INVALIDATED.json. All identified old learners/workers were closed with positive zero-remaining /proc receipts, without lease cancellation; fresh run-v2 never resumes their policy/optimizer/sampler. Nested maintenance initially failed because of inherited CPU binding; corrected with explicit CPU binding and clean Slurm environment, then all closures verified. See startup-closure-all.json and per-job receipts on the remote root. Old startup536/320valid updates and episode successes must NOT be reported as clean-run learning results.

## Greedy CPU scaling and monitoring

Separate state `autoscale-v2`, bound to run-v2. Up to1000rolloutCPUs for this1024env batch;100worker slots/10CPUs per worker/cloud_workers0. Counts pending/base/additional jobs, checks actual committed SGD and progress<20min, throttles sweeps5min, max10new ten-CPU jobs/cycle. QueriesCPU1/10/.../1000 and feasible-node grid, excludes all own occupied hosts, exact near-start<=180s recheck before each submit; durable intent prevents ambiguous retries. First safe projection sweep found no near shapes and submitted nothing. Do not infer more CPUs are available from credits or configured limits.

The existing shared heartbeat remains30s, preserving PRIMARY's intentional pause and CUBE/benchmark retirement. New READ/SCALE use local argument-based monitor_entry.py and pinned remote scripts; actual stdout/stderr/returncode receipts are saved locally. Old deeply nested shell commands failed before SSH and were replaced. Keep parser failures separate from transport failures. Native/source failures or20min real progress stalls return to this parent; routine requeues stay quiet. Completion requires10000committed updates and a validated checkpoint at that counter. No new monitor/chat created.

Useful local commands from the workspace:

```sh
python3 robotics-assembly-3.0/operational/wowrobo-chimera-20261107/status.py
python3 robotics-assembly-3.0/operational/wowrobo-chimera-20261107/monitor_entry.py read
python3 robotics-assembly-3.0/operational/wowrobo-chimera-20261107/monitor_entry.py scale
```

Use the approved network tool route. `start.py` owns the persistent controller, source remains frozen, and ownership locks prevent duplicate learner/worker starts. Model/bank fixes require a documented new version or continuation. This is early diagnostic learning, not production readiness or proven task learning. A cube-style main dashboard/replay integration remains a separate pending requirement.
