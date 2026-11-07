# Oracle handoff for Nikita

**Goal:** Find learner/integration bugs first, then run a bounded larger experiment without claiming the assembly task is solved.

**Current shared environment:** main uses `shared-side-v8-five-tasks` through the same factory as Quest. Bootstrap installs the pinned geometry runtime and generates current reset/reference data for motor, horn, screw, retainer and yoke. Use [start a run](START_RUN.md) for the current commands. Earlier snapshot/results below are historical; see [current environment](CURRENT_ENVIRONMENT.md). Executed motor seating and hardware calibration remain unqualified. The complete Quest app is included under `teleop/`; see [teleop handoff](TELEOP_HANDOFF.md).

Run the commands in the root README on the allocated GPU node. This host has no Oracle SSH/scheduler credentials, so submission is performed from your cluster account. The default launcher is single-node/single-GPU; `CUDA_VISIBLE_DEVICES` from the scheduler is respected. Add your actual account/partition options when using Slurm. Do not assume every Oracle GPU/CPU architecture supports this pinned x86-64/CUDA build.

## Runtime and first check

`bootstrap.sh` installs the pinned environment, downloads the private release asset selected by `configs/runtime_bundle.json`, verifies its Git-pinned SHA-256 and every file, and builds both plugins against the installed MuJoCo. It does not regenerate CAD/SDFs or copy old training runs. If `gh` is unavailable on the node, download the private release elsewhere and set `RUNTIME_ARCHIVE=/shared/path/native-runtime-v2.tar.gz`. Original Hiwonder CAD is excluded; derivative hashes and its recorded import provenance are checked explicitly.

Run `healthcheck.py --output runs/learner-smoke` in a new directory. It performs actual native-environment/CUDA PPO updates and checks actor/critic gradients and parameter changes, normalizer updates, finite loss, expected transition counts, SGS outcome/RNG behavior, atomic checkpoint contents, exact restoration of saved model/optimizer/SGS state, and further updates after resume. Read `report.json`; the process exits unsuccessfully if a check fails. A second smoke invocation needs a new output path.

The checkpoint check deliberately removes the legacy sampler sidecar before resume. This tests the new self-contained checkpoint. It verifies learning-state restoration; exact simulator trajectory/RNG continuation is not implemented or claimed. The sampler control is a synthetic outcome test; actual episode outcomes are reported separately.

## Updating an existing checkout

```bash
git pull --ff-only
unset RUNTIME_ARCHIVE  # if previously set to the old v1 archive
bash scripts/cloud/bootstrap.sh
bash scripts/cloud/test_learning_h100.sh
```

The current runtime is `native-runtime-v2`, with the [Quest physical snapshot](CURRENT_ENVIRONMENT.md) and `data/so101/banks_quest_v2`. Bootstrap reads the release/asset from Git-pinned metadata. Existing v1 assets and banks remain intact; the new bank installs beside them. Start a fresh run: earlier layout/bank checkpoints are incompatible and are rejected. Every task now starts with all parts loose, so old prepared-start scores cannot be compared directly.

## Larger run

```bash
.venv/bin/python scripts/cloud/run.py --run-dir runs/oracle-debug-001
# Override within your actual allocation if necessary:
.venv/bin/python scripts/cloud/run.py --run-dir runs/oracle-debug-002 \
  --envs 128 --threads 16 --iterations 1000 --max-hours 4
# Or, from the repo root on Slurm:
sbatch --partition YOUR_GPU_PARTITION --account YOUR_ACCOUNT scripts/cloud/train.sbatch
```

The default 256-world/32-thread profile is a starting allocation, not an Oracle benchmark. Native physics can be CPU-bound even though PPO uses CUDA. Profile `t_sim`, `t_update`, GPU memory and total transitions/s before increasing environment count. Filling every cluster GPU is not a correctness test. The optional whole-helper GPU backend requires its separate qualified model/grid archive and new-machine validation; this runtime bundle deliberately excludes those research assets.

Outputs are `launch.json`, `preflight.json`, `trainer.log`, `log.jsonl`, `checkpoint.pt`, `sampler.pt`, kept policy snapshots and `exit.json`. `pi`, `vf`, entropy, KL, learning rate and std must remain finite; actor/critic updates and resumed counters must be real. PPO actor loss is not a monotonic objective. Track task successes, safety stops, timeouts and reset coverage independently. Zero loose-part success is an accepted current limitation, not evidence that the optimizer works or is broken.

The launcher refuses to overwrite an existing run. To resume, read the exact trainer command in `launch.json`, keep the same model/bank/task/config, and increase `--iters` if needed. The trainer resumes from `checkpoint.pt`. Use SIGTERM on the trainer or `scancel --signal=TERM JOB_ID` to request a saved stop after the current update; the wall-time bound also saves before exit. There is no guaranteed stop latency inside a physics/update call.

## What to send back

Share the Git commit, `launch.json`, smoke report, preflight, `trainer.log`, `log.jsonl`, exit status, first failing traceback/state and the latest checkpoint. Preserve failing runs. Redact cluster credentials and account information before sharing outside the private project. Do not treat prepared-reset success as tabletop acquisition, full sequencing or hardware transfer.

The handoff fixes minibatch remainder/small-valid-batch failures and embeds SGS atomically. Existing physics/objective imperfections remain visible. See `HANDOFF_VALIDATION.md` for measured checks and the historical SO101 status for task failures.

## One H100: allocated CPUs plus actual learning comparison

After accepting the private repo invitation, run on the allocated Linux x86-64 GPU node:

```bash
gh repo clone Alessandro-Maniscalco/robotics-assembly-3.0
cd robotics-assembly-3.0
bash scripts/cloud/bootstrap.sh
bash scripts/cloud/test_learning_h100.sh
```

The H100 script reads CPU affinity and `SLURM_CPUS_PER_TASK`, reserves two CPUs for the learner/coordination when possible, and chooses 512–4096 environments from that CPU budget. It preserves the scheduler's GPU visibility. Training defaults to four hours/10,000-update upper limits, whichever is reached first; saving happens at update boundaries. Smoke and assessment time are additional. These starting sizes are not an H100 throughput qualification. For example, 64 allocated CPUs select 62 simulation threads and 1024 worlds.

The script first checks actual gradients, weight changes and checkpoint restoration. It then runs the larger training job, retaining `initial_policy.pt` before the first PPO update, after normalization has seen the first real rollout. Finally it compares initial/final **mean policies on the same independent EVAL starts**: 16 episodes per task, five tasks, both policies. The assessment uses up to eight CPU workers. It reports task success, safety stops, timeouts and placement error; weight changes or lower placement error alone do not count as task success. The finite cohort is not a generalization proof.

Overrides are explicit, for example:

```bash
H100_ENVS=2048 H100_HOURS=8 bash scripts/cloud/test_learning_h100.sh
# Use fewer worlds if memory/throughput is poor:
H100_ENVS=512 H100_THREADS=32 bash scripts/cloud/test_learning_h100.sh
```

`H100_RUN_ROOT`, `H100_ITERS`, and `H100_TRIALS` also override defaults. Every run needs a new output directory. Monitor `train/trainer.log` and `train/log.jsonl` under that directory; inspect `smoke/report.json` and `train/learning-assessment/comparison.json`. `watch -n 2 nvidia-smi` shows GPU activity, while `t_sim` versus `t_update` shows whether native CPU physics is limiting throughput. This backend does not promise to fill an H100 with GPU physics.

The pinned cloud dependencies now include `trimesh`, required by primitive geometry observation sampling. The earlier lockfile omitted it; an existing asset-development environment had masked that setup dependency.
