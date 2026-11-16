# Robotics assembly 3.0

**Goal:** Run one SO101 assembly environment for Meta Quest teleoperation and policy training, with reproducible setup and explicit physical limits.

**Current development:** v14 physics with the user-selected 0.5 mm acceptance limit. All 50 nearby-arm pilot states, strict preflight and CUDA checks pass. A bounded pilot experiment is active; the full 8,000-state production bank remains pending. [Run readiness](docs/RUN_READINESS.md) · [Review user intent by section](docs/user-intent/README.md). This collaboration snapshot follows experimental v13 (`85c006a`) and includes the SGS explorer and current training/teleop source.

**Pinned experimental training snapshot:** [start a bounded v11 run](docs/START_RUN.md) while CAD work continues. Use the fixed tag and an explicit validated bank; old v9/v10 SGS proposals are incompatible.

Start with [User Intent.md](<User Intent.md>) for current goals, scope, decisions and open questions. [AGENTS.md](AGENTS.md) requires assembly chats to keep it current; `MEMORY.md` preserves decision history.

This repository starts from the reviewed SO101 fork at `a01fedd98ef439eec2e6716e98b842565879b42b`, based on [AssembleNIST](https://github.com/suniique/assembly-nist) at `eb5d15e`. Upstream history, attribution and package paths are preserved. **Full loose-part assembly is not learned.** This handoff tests software health separately from task success.

## Teleoperate on your computer with Meta Quest

Use the self-contained [teleop app](teleop/README.md). On Linux x86-64 or Apple Silicon macOS with OpenGL drivers, C/C++ compilers, `uv`/Python 3.12, authenticated `gh`, and an authorized Quest USB connection:

```bash
gh repo clone Alessandro-Maniscalco/robotics-assembly-3.0
cd robotics-assembly-3.0
bash teleop/setup.sh
bash teleop/run.sh --quest
```

Select **Enter Quest VR**, release Y/grips and **press Y**. Hold either grip to move its arm, use the trigger for its jaw, X to recenter and B to Stop. Setup uses a separate `.venv-teleop` and needs no PyTorch/CUDA. Omit `--quest` for the desktop preview; detailed USB/graphics help and checks are in [teleop/README.md](teleop/README.md).

For an existing clone, use `git pull --ff-only` and rerun setup to rebuild the current native contact adapter. Friends need access to this private repository and its pinned runtime release. The shared physical environment is included; local training checkpoints, generated banks and review images are excluded from source Git. The full 8,000-state staged training bank is still being prepared. Inspect locally available states with the [SGS explorer](tools/sgs_viewer/README.md).

## Start on the Oracle GPU node

Prerequisites: Linux x86-64, an NVIDIA GPU/compatible driver, `g++`, Python 3.12 or `uv`, and authenticated access to this private repository's release asset.

```bash
gh repo clone Alessandro-Maniscalco/robotics-assembly-3.0
cd robotics-assembly-3.0
bash scripts/cloud/bootstrap.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/cloud/healthcheck.py \
  --output runs/learner-smoke
# After the smoke report passes:
.venv/bin/python scripts/cloud/run.py --run-dir runs/oracle-stress
```

The default launch profile restores the inherited training scale: **8,192 environments, 32 rollout steps, 10,000 iterations and effectively no trainer wall-time limit** (`max_hours=1e9`). This collects 262,144 transitions per iteration. CPU threads follow `SLURM_CPUS_PER_TASK`, falling back to 8; `--threads` overrides this. Native CPU physics, one CUDA learner and the SO101 safety/reset/observation settings remain selected. Saves occur every ten updates after strict preflight. The old 4h15m cap has been removed. The example `sbatch scripts/cloud/train.sbatch` still requests 32 CPUs, one GPU, 64 GB RAM and the selected partition’s default scheduler lease; choose the allocation/time limit explicitly for the full run. These requests are not a measured capacity requirement for 8,192 environments.

For an H100 with automatic CPU allocation and initial/final task assessment, run `bash scripts/cloud/test_learning_h100.sh` after bootstrap. See [Oracle handoff](docs/ORACLE_HANDOFF.md) for diagnostics, resume and cluster limits; [architecture](docs/ARCHITECTURE.md) for the current structure and proposed organization; [verification](docs/HANDOFF_VALIDATION.md) for actual local checks. No Oracle credentials are embedded, and no remote job is launched by cloning or bootstrapping.

For the current five-task source and a one-hour example, follow [start a training run](docs/START_RUN.md).

The selected policy input uses **256 combined mating points**, plus128 active-part and64 distal-finger points (448 total). For screw, the256 budget is shared equally between predecessor assembly and screwdriver. See [input specification](docs/SO101_MATING_INPUTS.md) and the [interactive visualization](docs/visualizations/so101-mating-budget.html). Download/open the HTML locally; GitHub's file viewer shows its source. The128-budget comparison remains available in the view. Old observation-layout checkpoints require fresh training.

## Current structure

```text
assembly_policy/
  so101/           # Hiwonder task, SO101 scene, native/Warp environments, resets and observations
  rl/              # SimBa networks, PPO support, SGS, vector/remote environments
  train.py         # shared learner, rollout, GAE, update, checkpoint and resume
  sim/             # original NIST/Franka reference, retained for compatibility
scripts/
  cloud/           # portable setup, runtime-bundle verification, smoke and bounded launches
  so101_*.py       # physics qualification, source checks, assets and research tools
teleop/            # standalone Quest app, CPU setup, host/control, stereo web client and tests
configs/           # versioned physical scene, pinned runtime and explicit stress profile
tests/             # learner edge cases and checkpoint regression checks
docs/              # architecture, handoff, current limitations and original reports
assets/            # upstream NIST/Franka reference assets
assets_local/      # ignored native SO101 runtime, installed from the private release
data/              # ignored reset banks, installed with the runtime
runs/              # ignored logs/checkpoints; new directory per experiment
```

Training and Quest call the **same canonical native factory**, `assembly_policy.so101.scene.model`, using `workcell.py` and `configs/so101_scene.json`. The `shared-side-v14-horn-support` repair uses receiving-frame targets, continuous original-surface contacts, compliant motor/base response and guarded table impacts, while retaining the [bare HOTO tool](docs/HOTO_S_A01_MODEL.md), centered layout, native fingertip closure and five tasks ending at yoke. The user approved the mating/drop review; the v14 baseline passes 82 core and 84 teleop tests. The user accepts the reviewed falls and selects 0.5 mm penetration/material acceptance; SGS completion remains outstanding. The published experimental v11 tag remains a separate fixed snapshot. See [repair status and review media](docs/ASSEMBLY_MATING_REPAIR.md). See the [current environment](docs/CURRENT_ENVIRONMENT.md). The repository-owned `teleop/` app supplies controller mapping and stereo rendering, with no independent physical overrides. Earlier reports retain their failed mechanics checks. The old training checkout remains historical.

Bootstrap installs the pinned v1 geometry runtime, builds the shared contact adapter and generates current reset/reference data locally. It keeps old banks and rejects incompatible resumes. V11 needs fresh reset/reference data; v9/v10 banks and older publication checks are incompatible or historical. Quest A offers native-qualified completed states through yoke, with motor sideways and horn onward upright. See [current environment](docs/CURRENT_ENVIRONMENT.md). **Executed motor insertion, screw driving and hardware fit/friction/compliance remain unqualified.** Valid prepared completion states and numerical health do not establish learned assembly.

## Learner fixes in this handoff

The [balanced subtask curriculum](docs/SUBTASK_CURRICULUM.md) prepares each placement with completed prerequisites, prioritizes receiving geometry/tool observations, and targets 256 TRAIN plus 64 EVAL starts per task/family. SO101 training defaults to the dedicated balanced bank and rejects missing/stale data. Original all-loose full-sequence banks remain separately selectable. The earlier [v9 noisy-bank report](docs/SGS_CURRICULUM.md) is historical and does not qualify v10 states.

The PPO update now uses every valid sample when minibatches have a remainder, handles one valid sample without NaN variance/empty minibatches, and rejects an all-invalid rollout before optimization. New atomic checkpoints embed SGS state with the model and optimizer; complete old checkpoints can still use their sampler sidecar. Missing or mismatched sampler state fails explicitly. Actor/critic objectives and network/SGS equations are preserved; this is not a new learning algorithm.

Current task limitations include unlearned full assembly, a 15 s horizon and unqualified screw-driving mechanics. Subtask training starts with completed prerequisites, while full-sequence evaluation starts from all loose parts. Subtask success must be reported separately from full tabletop assembly. See [historical SO101 status](docs/SO101_STATUS.md). Numerical health, falling critic loss or finite gradients cannot establish successful assembly. Original [README and NIST results](docs/UPSTREAM_README.md) are retained as upstream context, not SO101 results.

## Development checks

```bash
uv pip install --python .venv/bin/python pytest
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```

No generated runtime, dataset, checkpoints or private plan enters Git. Keep new experiments in a fresh `runs/` directory. Retain failure traces and report success, safety stops, timeouts, throughput and numerical validity independently.
