# Assembly training dashboard

Open http://localhost:8881/. The design follows the existing cube dashboard on8880, in a separate server.

The dashboard uses the assembly workspace layout. Place copies of
`assembly-dashboard/` and `cube-dashboard/` beside this `assembly-so101/`
checkout and the `robotics-assembly-3.0/` checkout used by your deployment.
Use the frontend dependencies from this checkout's `web/` directory. The
read-only training relay, model assets and saved-state bank are supplied by the
existing authorized workspace. The Vercel preparation lives under `vercel/`.

From the workspace root:

```sh
assembly-so101/web/node_modules/.bin/vite build --config assembly-dashboard/vite.config.mjs
PYTHONPATH=assembly-so101 assembly-so101/.venv/bin/python assembly-dashboard/server.py --port 8881
```

The bridge performs one read-only SSH observation every15seconds, inside existing Slurm allocation3077705. It reads bounded training logs, controller/configuration, reset manifest and checkpoint **file metadata**. It never deserializes policy/Adam checkpoints, reads worker-auth/server tickets, changes jobs or runs physics. Once per60seconds it reads only the small saved `sampler.pt`, using CPU PyTorch in the same allocation, and joins entry outcomes to the exact selected NPZ order and per-family cap. Modal app/container status is read every45seconds when the already configured SDK is available.

Run selection follows the local release targets and discovers activated versioned continuations under their roots. A prepared checkpoint is not treated as an activated learner: both command and learner-process records are required. Old/new histories retain version labels and separate smoothing; current task statistics only use the active run. Handoff and unavailable/stale observations are explicit. Modal goal/request/container counts remain separate from measured contributing environments.

Curves cover five assembly tasks, each task's starting states, reset-family totals, safety exits, throughput and policy KL. Episode-return curves remain unavailable unless the learner logs them.

The **Latest task rollouts** row provides one playback selector for every task. It selects the most recently recorded complete success across that task’s starts, falling back to its latest complete failure only when no success is saved. It uses only the active run and matching model identity. Missing recordings remain labeled rather than substituted. Clicking a task or its rollout selector plays that task-wide selection in the shared player. The viewer’s **Starting state** dropdown lists named states by full stage ID and recording availability; an explicit state remains pinned through live updates, with success-first/failure-fallback playback within that state. Its Automatic option returns to task-wide selection. Selecting a starting-state table row also pins that state. New recordings update the selection automatically. One on-demand player is shared to avoid five simultaneous CAD renderers or extra training work.

Actual per-start training replays use an opt-in observational learner bootstrap. Stage `record_training.py` with the canonical `assembly_policy/wowrobo/telemetry.py` beside it in a versioned continuation, preserving the original training CLI arguments and physics/worker source. The main-run chat owns checkpoint-safe activation and latest-state migration. `recorder-release.json` pins both files.

The recorder keeps at most one in-progress episode per(task,starting state), not one per parallel world. It copies already-received compact body poses at80ms spacing plus the exact terminal frame, skips initial desynchronized partial episodes and lost-worker fragments, and writes fixed success/failure NPZ slots on one bounded background writer. Success is always preferred; a complete failure is the fallback until success exists. Refreshes are throttled to300seconds except the first success. Queue saturation drops recording work, never training transitions. A5090-world/96-body/29-start synthetic observer benchmark measured0.056ms mean and0.494ms p99 per control, with14.15MiB frame buffers; compression/I/O is off the control path.

The dashboard reads only the small replay index during normal polling. It downloads, validates and caches a selected clip on demand, with at most one SSH replay fetch in flight and four decoded clips retained. Recorded geometry is loaded once per run from its exact review model, with body order and run/model/bank/reward/observation identity checks. Browser playback changes saved visual poses only: no local simulation or policy inference. Static scenes render on demand, active playback is capped at30renderfps, and offscreen/hidden playback pauses. The approved reset remains explicitly labeled when no complete clip exists or when selected manually.

Checks:

```sh
cd assembly-dashboard
../assembly-so101/.venv/bin/python -m unittest discover -p 'test_*.py'
```

Generated `dist/`, `live-cache.json`, server logs and screenshots are local artifacts.

Per-environment starting-state views show all approved starts, actual active-bank entry counts, nominal versus variation counts, held-out EVAL counts when recorded, sampled episodes, visited entries and SGS sampling probability. SGS success pools the retained last50 outcomes per entry, not all-time task success. Historical per-start curves begin with the first dashboard sampler observation; earlier per-start curves are not invented. Clicking a start selects its curve and saved complete training episode, preferring success and otherwise showing failure. The approved prototype reset can be selected separately. Missing starts remain labeled not in active bank.

Live verification on 2026-11-07: run-v6 recorder activated with 1,143 task-conditioned TRAIN rows from the latest 874 accepted TRAIN variations plus 25 prototypes (154 EVAL variations held out). Actual forearm floor-grasp episode `1794096940086011547-24`, iteration14, safety failure,0.08s/2frames passed identity/SHA validation, exact-scene reconstruction, cached HTTP200 and browser playback controls. Remaining starts populate as complete episodes are captured; no success is invented. All12 dashboard tests pass.

## Unified training and state workspace

The dashboard now hosts **Training** and **States & variations** at one origin. Open `/#training` or `/#states`. The latter embeds the existing native reset explorer through `/states/`, retaining its complete state navigation, variation controls, Three.js, point/reward layers and official MuJoCo WebViewer. Training graphs, sampler tables and recorded episode replay remain available on the Training tab.

`state-viewer-target.json` is an operator-owned hot-swappable pointer, never an HTTP resource:

```json
{
  "url": "http://127.0.0.1:8892",
  "bank_status": "building",
  "accepted_train": 1500,
  "accepted_eval": 300,
  "target_train": 16000,
  "target_eval": 3200,
  "updated_at": 1794113258,
  "snapshot_label": "Previous completed bank; the current accepted snapshot is uploading",
  "training": {
    "state": "paused",
    "checkpoint": 31,
    "reason": "Training is paused while the expanded reset bank is prepared"
  }
}
```

Counts above illustrate the schema; the live file is maintained by the generation/publishing owner with actual published counts. The snapshot label distinguishes a previous completed collection from the current fresh-namespace build. `/api/workspace` exposes only the selected progress/training fields, not the private loopback URL. Changing the target port or bank revision reloads an already-open embedded viewer while retaining its selected assembly stage/rendering mode where possible.

The proxy accepts only a configured `http://127.0.0.1:PORT` target, without credentials, query, fragment or arbitrary path. It routes known viewer API endpoints and static/native-viewer assets. Unknown API/private-file routes and path traversal are rejected. It cannot forward to a URL from a request parameter. Public HTTPS Origins must match the external request host, then are rewritten to the fixed internal Origin expected by the local reset viewer. WebSocket traffic is relayed under the same `/states/` origin; model upload channels stay disabled. HTML/JS/JSON URLs and computed viewer network requests are scoped so `/api/scene` cannot accidentally select the dashboard's preview scene.

Public-facing routes are `/`, built `/assets/*`, the bounded dashboard APIs (`/api/live`, `/api/scene`, `/api/preview`, `/api/episode`, `/api/replay-scene`, `/api/workspace`) and the restricted `/states/` proxy. No directory-root static handler, credential/config file, raw live-cache file, checkpoint weights or arbitrary SSH/file API is exposed. Telemetry and selected CAD/reset/replay geometry are the intended site content. The dashboard itself listens on loopback; a public tunnel is a separate operator action.

Cross-origin isolation headers are enabled on the shell and embedded viewer to support the official MuJoCo renderer. Local proxy regression checks cover Cloudflare-style HTTPS Origin rewriting, root asset/API URL scoping, studio JSON URL rewriting, WebSocket relay, denied foreign Origins and denied private/traversal paths. Existing training/replay regression tests remain unchanged and pass.

Unified deployment verification: the prepared frontend builds successfully and all13 existing dashboard/replay tests pass. `server-unified-startup.json` records the actual loopback process command/PID, HTTP200 shell/workspace/telemetry/state catalog/scene/native-viewer checks, HTTPS-public-Origin studio startup, denied foreign Origin and denied private/traversal routes. It records telemetry field names only for the route audit. The large native geometry response is compressed for public clients (222.4MB JSON →47.8MB gzip in the verified collection), with compression offloaded from the event loop; raw geometry JSON is not unnecessarily decoded/re-written. Decoded upstream `Content-Encoding`/`Content-Length` headers are removed before serving, and upstream requests ask for identity encoding. The operator's `restoring` training state remains explicitly visible until restoration is complete; bank completion alone never turns the status into live training.

## Cube handoff training

Training now has **Assembly / Cube handoff** experiment links. `/#training/cube`
embeds the existing cube dashboard through fixed same-origin `/cube/` routes.
Its reset viewer link uses `/cube-states/`; the original assembly state library
continues at `/states/`. HTML/JS/API and MuJoCo WebViewer traffic stay in their
own namespace. Cube proxy routes expose only GET/HEAD for the scene, metrics,
initial state, family replay and frontend assets; private files and mutation APIs
are denied. Destination ports are fixed loopback services (cube 8880, resets8877),
never request-supplied URLs. The existing public tunnel URL is unchanged.

The cube view shows the completed historical run at iteration10,000, including
all11family curves and success-first family replays with the explicitly labeled
pre-handoff failed diagnostic. This website integration does not restart training
or deploy the subsequently added local +0.5 handoff reward. Start the existing
cube-dashboard/server.py service on8880 alongside this server to keep these routes
available. Existing assembly training and state-viewer services remain separate.
