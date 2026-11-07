# Start an experimental v11 training run

**Goal:** Start a bounded learner run from a fixed source snapshot while CAD repair continues separately.

The user explicitly authorizes this imperfect-physics experiment. `training-v11-20261106` fixes the v11 contact model and the selected 448-point observation. Use fresh weights. Native CPU physics supplies rollouts; one CUDA GPU trains the policy. This is not a claim of learned assembly or hardware calibration.

## Set up a separate checkout

Use Linux x86-64, an allocated NVIDIA GPU, Python 3.12 or `uv`, a C/C++ compiler, authenticated `gh` access to the private repository/release, and at least several GiB of free disk for dependencies, checkpoints and logs. Do not pull new CAD changes into this checkout during a run.

```bash
gh repo clone Alessandro-Maniscalco/robotics-assembly-3.0 robotics-assembly-training
cd robotics-assembly-training
git switch --detach training-v11-20261106
bash scripts/cloud/bootstrap.sh
```

Bootstrap installs the hash-pinned `native-runtime-v1` derived geometry, compiles the current native plugins, and generates the v11 all-loose bank/reference locally. No original vendor CAD is required. Generated banks and runs are not committed. For an existing separate clone, fetch the tag and detach at it before bootstrap.

## Run now with the checked all-loose bank

This explicit bank contains 32 TRAIN and 32 independent EVAL starts per task (320 total). SGS adapts probabilities over its 160 TRAIN entries; every start is all-loose. Sparse success may be hard to discover, particularly for later tasks whose predecessors must also be assembled. This option is useful for learner diagnostics and an initial baseline; it is not the staged curriculum.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -m scripts.cloud.healthcheck \
  --output runs/v11-smoke --bank-dir data/so101/banks_shared_v11 \
  --device cuda --envs 8 --steps 8 --threads 2
```

After `runs/v11-smoke/report.json` reports `passed: true`, start a one-hour pilot:

```bash
.venv/bin/python -m scripts.cloud.run \
  --run-dir runs/v11-pilot --bank-dir data/so101/banks_shared_v11 \
  --threads 8 --envs 64 --iterations 1000 --max-hours 1
```

This historical v11 pilot explicitly overrides the current source's inherited-scale defaults: 8,192 environments, 32 steps, 10,000 iterations, max_hours=1e9, and threads from `SLURM_CPUS_PER_TASK` (otherwise 8). SO101 physics/observation/reset settings remain separate. The fixed v11 tag itself is unchanged.

Use threads within your CPU allocation. The launcher runs strict native preflight and stops at the time budget or 1,000 updates, whichever comes first. For an overnight experiment change the run directory and `--max-hours 8`; inspect the pilot first. Use a new directory for each fresh run. The local DCF host currently lacks checkpoint disk headroom; these instructions do not bypass disk/resource requirements.

Compare initial and final policies on identical independent starts:

```bash
.venv/bin/python -m scripts.cloud.assess_learning \
  --run-dir runs/v11-pilot --bank-dir data/so101/banks_shared_v11 \
  --trials 16 --workers 4
```

Check finite losses, nonzero updates and checkpoint restore, then task success, safety stops, timeouts and placement errors. PPO losses alone are not evidence of task learning.

## Staged SGS option

Old v9/v10 prepared states are incompatible. The newer generator prepares prerequisites and leaves the active part in one of five families. A bounded pilot can be generated separately:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -m assembly_policy.so101.subtask_curriculum \
  --out data/so101/banks_shared_v11_subtask_pilot \
  --train-count 2 --eval-count 2 --workers 2
```

Only a complete, balanced, qualified build is promoted. This pilot targets 100 states across five tasks and five families; generation can reject candidates or fail to fill groups. Do not rename a `.building` folder into a usable bank. After successful promotion, pass that explicit bank to **all three** smoke, launch and assessment commands above. For separate family evaluation use `assembly_policy.so101.evaluate --families all` with the matching bank/checkpoint. [Generation, resume and larger-bank details](SUBTASK_CURRICULUM.md).

The launcher's omitted `--bank-dir` selects the larger `_subtask_256` bank and correctly fails if it is absent. The commands above deliberately select available data instead.

## Evidence and physical limits

See [snapshot validation and changed files](TRAINING_SNAPSHOT_V11.md). The shared model retains native contacts and rejects excessive overlap; task errors are not excused to make rewards pass. Guided insertions are force fixtures, not learned arm motions. Yoke insertion has little margin against the 0.4 mm material threshold, and a 25 mm off-center motor drop still exceeds its five-second settling-speed limit (15.8 versus 10 mm/s). Real-kit geometry, friction and forces remain uncalibrated. Keep final CAD visual review separate from this experimental publication.
