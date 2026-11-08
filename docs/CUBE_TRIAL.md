# Cube handoff trial

This trial uses two facing SO-101 arms, 75 cm between the backs of their bases,
a 36 mm cube, and a target circle under the right arm. Physics is native MuJoCo
4.0.0: no custom collision library, SDF, plugin, equality attachment or contact
callback. Observations use 256 cube surface points, 128 circle perimeter points
and 64 gripper points.

## Setup and reset viewer

From the repository root, with Python 3.12+ and Node/npm installed:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[train,viewer,dev]'
.venv/bin/python -m scripts.setup_cube_trial
npm --prefix web ci
npm --prefix web run build
.venv/bin/python -m assembly_policy.viewer --scene cube-trial \
  --banks data/cube_trial/banks_handoff_bridge_v8 --port 8877
```

Open <http://localhost:8877/>. Setup downloads only checksum-pinned SO-ARM100
helper XML/STL assets; it needs no private vendor CAD or collision plugin build.
An existing matching asset cache can be used with
`--asset-source /path/to/assets_local/so101`.

The curated `runtime/cube-trial/` snapshot contains 1,408 training and 352 eval
states, with 128/32 per family. Setup verifies their checksums, copies them to
ignored `data/`, and adjusts only the copied viewer manifest's local path-based
signature. The portable physics signature and every NPZ/context byte are retained.
`reset-context.npz` is required: it preserves actual five-frame policy history,
previous action, recent right grasp and carry prefix for dynamic reset states.

Chronological families are loose, left pickup, converging transport, pre-handoff,
partial right contact, light right grasp, dual grasp, post-handoff, right transport,
near target, and placed held. Original IDs 0–8 are preserved; the two contact
families are IDs 9–10. Pre-handoff has no receiver contact. In the added contact
families, the left still supports the cube. Right transport snapshots came from
successful dual-grasp/post-handoff policy episodes; these are not a demonstration
of successful full-task policies. Bank provenance is included beside the states.
`cube_trial.build` generates the earlier nine-family authored bank; it does not
reproduce the policy-derived v8 snapshot from scratch without its source rollout
artifacts. Use the published, checksum-verified snapshot for this trial.

## Rewards and episode ending

The current source implements `placement_v5_handoff_bonus`:

- +0.5 once per episode for a completed handoff. A loaded two-pad left grasp must
  have been observed; after pickup, the right must sustain a loaded two-pad grasp
  airborne for 120 ms while neither left pad supports the cube. Qualification is
  checked every native 2 ms substep. An interrupted hold restarts the timer.
- Post-handoff/right-transport/near-target/placed-held resets start with this bonus
  consumed, using their authoritative bank family. Merely touching is insufficient.
- +1 for a qualified placement and release: whole cube inside the circle with
  2 mm tolerance, actual loaded cube–table contact, right two-pad grasp within
  100 ms before touchdown, right jaw >=0.275 rad, no supporting pad load above
  0.01 N on any of the four pads, linear speed <0.02 m/s, angular speed <0.5 rad/s,
  and three consecutive qualifying 40 ms checks. Left jaw angle is irrelevant.
- Each control step subtracts
  `2e-5 * (sum(action²) + sum(joint_velocity²)/25 + sum(normalized_torque²))`.

Outside-circle table contact after pickup terminates as a failure at the next
40 ms control endpoint, without timeout bootstrapping. Initial pickup dragging
is allowed. Free-flight gaps do not themselves invalidate an episode. Otherwise
the limit is 10 seconds (250 control steps). A handoff bonus is nonterminal;
it does not require the subsequent placement to succeed. Maximum positive reward
is 1.5 before costs. SGS still uses binary **placement** success, not bonus payout.

## Training and status

A local training invocation, after setup:

```sh
.venv/bin/python -m assembly_policy.cube_trial.train \
  --bank-dir data/cube_trial/banks_handoff_bridge_v8 \
  --run-dir runs/cube-handoff --envs 64 --steps 32 --iters 10000 \
  --n-part 256 --n-fix 128 --eps-q .01 --device cpu --no-eval
```

Choose environment/device budgets for your machine. SGS weights each bank entry
by `0.01 + sqrt(p * (1-p))`, where p is its last 50 placement outcomes; 0% and
100% entries receive the same floor. Replay chunks are 32 control steps, not
whole episodes. Resume checks enforce matching physics, bank, observations and
environment signatures; do not silently reuse a checkpoint from a different
reward/environment contract.

The recorded Chimera run ended at iteration 10,000 / 321,738,533 transitions with
the v4 placement reward, 10-second limit and immediate off-target failures.
Pre-handoff remained at 0%; the full task is unsolved. The +0.5 handoff reward is
implemented and tested in source but has **not** been trained on that run.
No learned policy weights or production checkpoints are included here.

`operational/cube-trial-20261107/` preserves historical cluster orchestration and
migration source. These scripts contain experiment-specific paths and assumptions;
they are not general launch commands. Current allocation IDs, config/pins, SSH
access, monitor automation state, auth keys and run receipts are operator-local
and excluded. Publishing this package neither launches jobs nor changes resources.

## Live dashboard

```sh
web/node_modules/.bin/vite build --config dashboards/cube/vite.config.mjs
.venv/bin/python dashboards/cube/server.py --port 8880
```

Open <http://localhost:8880/>. With no remote arguments, the dashboard is offline:
it displays the scene and any existing local cache, and performs no SSH requests.
To attach to a configured experiment using an already authorized CPU allocation:

```sh
.venv/bin/python dashboards/cube/server.py \
  --remote-root /absolute/remote/checkout --monitor-job YOUR_CPU_JOB \
  --ssh-host YOUR_SSH_ALIAS --port 8880
```

The remote checkout must contain its operator `cluster.json` and exporter. Each
export is read-only and runs in that named Slurm allocation, never on the login
node. It does not submit, cancel or restart jobs. `server.json`/auth keys are never
read. The server binds only 127.0.0.1. The browser polls every 3 seconds and the
allocated bridge every 10 seconds. Failed reads preserve stale data with status.

Each family card selects its curve and actual latest successful episode.
The optional single failed pre-handoff diagnostic is clearly labeled and never
counts as a success. State files (`live-cache.json`, optional `run-context.json`
and `diagnostic-pre-handoff.json`) belong under ignored
`artifacts_local/cube-dashboard/`, or a `--state-dir` supplied by the operator.
The diagnostic fallback also requires a matching local operator cluster config.
Generated frontend output, screenshots and live data are excluded from Git.
The deployment's completed-run dashboard reports its observed reward contract;
its footer describes placement qualification, not proof that v5 was trained.

## Verification

```sh
.venv/bin/python -m pytest tests/test_cube*.py -q
.venv/bin/python dashboards/cube/verify_replay.py
```

Tests cover native grasp/placement contacts, one-time handoff reward, reset
consumption, history/context, failure vs timeout, telemetry and bank provenance.
Replay verification uses a temporary native-MuJoCo episode and never writes a
production success. Licenses for downloaded helper assets and Three.js are in
`licenses/`; curated snapshot provenance is in `runtime/cube-trial/NOTICE.txt`.
