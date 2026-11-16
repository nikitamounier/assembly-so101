from pathlib import Path
import json

p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
r=read('ppo-result.json');rec=r['record'];n=read('instrumented-probe.json');b=read('baseline-probe.json');h=read('identity.json')['hardware'];ph=read('byte-parity.json');w=rec['workers']
remote=sum(s['seconds'] for s in r['steps']);critical=w['ct_max']*.001*32;mean=w['ct_mean']*.001*32
text=f'''# Original native WowRobo flame profile — November7,2026

[Open the interactive flame graph](flamegraph.html). Select learner, either real worker, or the native pair-timer hierarchy. Click a frame to zoom, reset to return, search names, and hover for exact seconds. No external scripts or network are required.

## Actual capture and isolation

This is a new run, not a reuse of the older SO101 profile. Frozen accepted-v4 native source/model `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`; original native checkpoint25/966743 SHA256 `df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b`. All191 active-inputs-v9 pins verified before and after. Original unoptimized surface-tree SHA `3df103be0e3b8cfde509a00da787ad753fc40cd4686db9a4b45163be70e88d8f`; no candidate bounds optimization or bare approximation. Native SDF, generalized supplements, per-episode friction, guards,0.25ms/160substeps/40ms action, sparse solver, tree mode1, original five tasks/rewards,448observation points.

Isolated output under frozen ROOT/native-flame-profile-20261107, using spare retained3606264 on {h['host']}. Live run-v9, its workers/checkpoint, all retained allocations and monitors were not changed. One bounded complete32-step PPO update from25→26, with two original worker protocol processes and two worlds; original policy/Adam/SGS/RNG/counters restored by the original checkpoint loader. Bank is exact training-accepted-v6,1143task-conditioned rows; no held-out rows. PPO5epochs/4minibatches/1024gradient chunks and original reward settings. This small diagnostic batch is not a production-fleet throughput benchmark.

| Full PPO metric | Measured |
|---|---:|
| Qualification | {'PASS' if r['qualified'] else 'FAIL'} |
| Valid transitions | {rec['valid_transitions']} /64 |
| Control rounds | {len(r['steps'])} |
| Full collection→checkpoint wall | {r['elapsed_s']:.6f}s |
| Collection incl inference/GAE | {sum(s['duration'] for s in r['spans'] if s['name']=='collect'):.6f}s |
| Wrapped remote steps | {remote:.6f}s |
| PPO update | {rec['t_update']:.6f}s |
| Checkpoint save | {sum(s['duration'] for s in r['spans'] if s['name']=='save'):.6f}s |
| Sum per-round worker maximum | {critical:.6f}s |
| Sum per-round worker mean | {mean:.6f}s |
| Ideal compute imbalance gap | {critical-mean:.6f}s |
| Remote-step residual above worker maximum | {remote-critical:.6f}s |
| Numerical instability flags | {rec['unstable']} |

Residual includes dispatch, serialization, result placement, sampling and scheduling; it is not pure network latency. Worker maximum and mean derive from all32actual worker reply compute timers. Localhost transport removes wide-area provider variation. The two-worker mixture cannot characterize400+worker tail behavior. The ideal balancing gap is a bound, not a promised improvement.

## Physical cores and SMT

Reserved8Slurm logical CPU slots map to4physical cores. Original step affinity: `{h['affinity']}`. Each simulation worker is pinned to one logical CPU on its own distinct physical core, nthread1; the learner uses a third physical core and the spare H100. See worker hardware JSON and learner hardware in ppo-result.json. Do not call8scheduler slots8physical cores. GPU allocation is reused, not newly submitted. Initialization/model build is outside the learner iteration window; worker wall graphs include their served initialization and idle barriers.

## Native collision attribution

Separate auxiliary panel: four single-world zero-action control intervals, each160substeps. These are measured timer widths, not production-weighted proportions or native stack-unwind samples.

| Start | Original rollout s | Instrumented rollout s | Original SDF s | Supplement s | Calls | Tree nodes | Leaf samples |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for orig,row in zip(b['rows'],n['rows']):
    q=row['pairs'];text+=f"| {row['stage']}{' variation' if row['variation'] else ''} | {orig['rollout_s']:.6f} | {row['rollout_s']:.6f} | {sum(x['stats'][1] for x in q):.6f} | {sum(x['stats'][2] for x in q):.6f} | {sum(x['stats'][0] for x in q):.0f} | {sum(x['stats'][3] for x in q):.0f} | {sum(x['stats'][4] for x in q):.0f} |\n"
text+='''
The loose start's three calls/substep are now identified. Both expensive internal pairs return zero contacts throughout this control interval:

| Loose pair | Original SDF s | Supplement s | Leaf samples | Returned contacts |
|---|---:|---:|---:|---:|
'''
for q in n['rows'][0]['pairs']:
    s=q['stats'];text+=f"| {' ↔ '.join(q['names'])} | {s[1]:.6f} | {s[2]:.6f} | {s[4]:.0f} | {s[8]:.0f} |\n"
text+='''
The first two are printed-module/own-servo-pivot relationships (authoritative body names: forearm_module↔target_servo_3_pivot, upper_arm_module↔target_servo_2_pivot). They are persistent internal articulated assembly geometry, rather than the active cover insertion. Zero contacts at these poses does not authorize deleting a passive stop interaction over hinge motion.

Upper far-side variation's largest combined pair costs:

| Pair | Original SDF s | Supplement s | Calls | Returned contacts | Zero-contact calls |
|---|---:|---:|---:|---:|---:|
'''
for q in n['rows'][-1]['pairs'][:8]:
    s=q['stats'];text+=f"| {' ↔ '.join(q['names'])} | {s[1]:.6f} | {s[2]:.6f} | {s[0]:.0f} | {s[8]:.0f} | {s[13]:.0f} |\n"
for row in n['rows']:
    total=sum(sum(q['stats'][1:3]) for q in row['pairs']);zero=sum(sum(q['stats'][1:3]) for q in row['pairs'] if q['stats'][0]==q['stats'][13]);text+=f"\n{row['stage']}{' variation' if row['variation'] else ''}: pairs returning zero contacts on every invocation account for {zero:.4f}s/{total:.4f}s ({100*zero/total:.2f}%) of pair collision time."
text+='''

Every invoked source-SDF pair in this panel, including physical_helper grasps, uses40initpoints×40iterations. These are invoked budgets, not proof1600iterations executed. Tree pruning is active: nodes, AABB/grid rejected nodes and surviving leaf samples are recorded per pair. Contact counts are sums across substeps, not simultaneous manifold size. Supplement net-added contacts exclude replacements that leave the manifold size unchanged. The table in the viewer contains every executed pair, its exact bodies, calls, budgets, node/leaf/rejection counts, original and final returned contacts. Lexical role labels are hints; exact geometry/body names should guide qualification.

Prioritize conservative rejection/caching or a qualified restricted surface region for the persistent internal pairs, then the true shoulder_connector↔servo_2 insertion interfaces. Ordinary SDF is distributed across many upper-stage pairs; its largest variation pair is base_side_cover_printed_0↔upper_arm_module_printed_0. Restricting supplements alone cannot remove that search cost. Role-specific budgets need replay/contact/hinge/insertion qualification. No optimization was deployed here.

## Parity, measurement limits, and files

All72arrays across four complete160-substep trajectories are byte-identical: trajectories, sensors, state/qpos/qvel/warmstart, contacts/forces, observations, reward, done and every info field. Both compressed NPZ files also have the same SHA256. See byte-parity.json. Instrumentation delegates original collision math unchanged; only private source copies contain timers/counters. Original build inputs, compiled libraries and instrumented C hash are retained in instrumentation.json and identity.json.

The single sequential original/instrumented timing pass is an overhead check with host noise, not a precise overhead estimate or speedup result. Different probe durations do not establish an optimization. Main PPO uses uninstrumented original native libraries. Worker/learner wall sampling targets20ms and can be biased by GIL scheduling. Native C++ rollout threads have no Python frames. Cluster perf_event_paranoid=4 restricts hardware sampling, so the native view is explicitly a timer hierarchy down to geometry-pair SDF/supplement phases, not invented C function stacks. Observation refresh is directly timed in the auxiliary panel; its small contribution can also be inspected at _refresh/observe frames in worker wall captures. No production-wide observation percentage is inferred from four starts.

Reproducible scripts: common.py, instrument.py, probe.py, capture.py, run.py. Run on the original frozen ROOT with the pinned virtualenv, PYTHONPATH=ROOT/assembly-so101, WOWROBO_CANONICAL_ROOT=ROOT/robotics-assembly-3.0, SO101_SURFACE_MODE=1 and OMP/OPENBLAS/MKL_NUM_THREADS=1. PROFILE_ROOT=ROOT, PROFILE_OUT=ROOT/native-flame-profile-20261107. Full exact launch command is in launch.sh. run.py requires the approved spare allocation3606264, checks every source pin and checkpoint, executes original/instrumented probes, then a bounded two-worker PPO. Review allocation availability before rerunning. Never point PROFILE_OUT at a live run or use mutable local environment code.

Local export excludes the per-run authentication key/server ticket and37MBcheckpoint. Frozen remote artifacts retain the isolated checkpoint. complete.json records closure of every owned child and post-run source verification. Viewer: flamegraph.html; raw sampled stacks: learner.speedscope.json, worker-0.speedscope.json, worker-1.speedscope.json. Rebuild with `python3 build_report.py`; write the report with `python3 write_report.py`.
'''
(p/'README.md').write_text(text)
phase=read('instrumented-timed-probe.json')
extra='''

## Fresh native phase hierarchy and sampled worker detail

The additional phase-clock pass resolves full native stepping through forward/position/collision/narrow phase and the same-pass pair SDF/supplement branches. Child widths use exclusive residuals, so inclusive timers are not double counted. This pass independently passes all72byte-identical arrays against the original replay. Native-phase-validation.json binds the small C monotonic clock source/library.

| Start | Native step s | Narrow phase s | Narrow share | Constraint solve s | Refresh s |
|---|---:|---:|---:|---:|---:|
'''
for row in phase['rows']:
    t=row['native_timers'];step=t['mjTIMER_STEP']['seconds'];narrow=t['mjTIMER_COL_NARROW']['seconds']
    extra+=f"| {row['stage']}{' variation' if row['variation'] else ''} | {step:.6f} | {narrow:.6f} | {100*narrow/step:.3f}% | {t['mjTIMER_CONSTRAINT']['seconds']:.6f} | {row['refresh_s']:.6f} |\n"
extra+='\nSampled leaf-frame times across the real PPO worker lifetime (includes initialization, outside learner capture):\n\n| Process | Wall capture s | Samples | Native rollout call site s | Observation _refresh s | Worker receive / learner selector s |\n|---|---:|---:|---:|---:|---:|\n'
for name in ['learner','worker-0','worker-1']:
    d=read(name+'.speedscope.json');pr=d['profiles'][0];by={}
    for stack,weight in zip(pr['samples'],pr['weights']):
        leaf=d['shared']['frames'][stack[-1]]['name'];by[leaf]=by.get(leaf,0)+weight
    get=lambda start:sum(v for k,v in by.items() if k.startswith(start+' ('))
    extra+=f"| {name} | {sum(pr['weights']):.6f} | {len(pr['samples'])} | {get('rollout'):.6f} | {get('_refresh'):.6f} | {get('_recv')+get('select'):.6f} |\n"
extra+='\nThese sampled observation/receive times overlap other processes and have20ms sampling resolution; they are not summed with the native auxiliary panel or substituted for explicit worker reply timers. Worker0spends about192s in native stepping while worker1spends about81s and waits about114s, making the barrier imbalance visible directly. The learner spends about194s in selectors.select; that is worker waiting, not evidence of slow networking.\n'
with (p/'README.md').open('a') as f:f.write(extra)
print(p/'README.md')
