# Retired native SAIL fleet — bare MuJoCo handoff

Latest direct human instructions in the main training chat prioritized faster
MuJoCo and deferred CPU expansion. Main activated the incompatible bare-v8
physics release on retained Chimera CPUs and requested old native cloud cleanup.
The stop marker remains authoritative; both launchers now refuse any resource
creation when it is present. Do not restart this old native image for run-v8.

Fresh `cleanup-proof.json` confirms zero active/paused SAIL boxes and zero
reserved vCPUs in this app, controller stopped, cleanup_errors=[]. All source,
native integration proofs, provider receipts, and root private `.env` remain.
The previous provider proof did verify4092allocated vCPUs, but full448-worker
native qualification was unfinished before this release was superseded.

An observational30-second guest-exec timeout incorrectly triggered the initial
large controller's global cleanup. Recovery preserved13loaded worker hosts,
tested failure containment, and adopted them without model restarts. The newer
explicit bare-v8 stop then correctly retired the remaining native fleet. The
recovery controller treats observation failures as degraded evidence, never a
reason to terminate healthy peer workers. Do not reuse the legacy global-fail
health-check policy in a new release.

Main owns the versioned bare worker image and useful CPU sizing. Current
canonical handoff: `MAIN-OWNER-RELEASE-HANDOFF.md`. The following describes the
historical native deployment, not currently running compute.

## Historical native deployment

The user explicitly authorized SAIL on November 7 and supplied the existing
credential. Root `.env` stores `SAIL_API_KEY`, mode 0600, ignored by root Git.
Never print the key or include `.env` in worker images.

Authenticated account: nikitamounier, 5,000 reserved-vCPU allowance and $501
credit at admission. No plan upgrade or purchase was made.

`launch.py` owns this separate pool and preserves the main thread's Modal,
Chimera allocations, learner, frozen physics, training bank, and PPO settings.
The user subsequently explicitly increased SAIL to **4,092 rollout vCPUs**,
additional to Modal and Chimera. Prepared full fleet: 63x64 +7x8 +4 =4092
reserved TOTAL SAIL vCPUs:4088rollout and4for the TLS broker. The4-vCPU tail
machine serves as broker rather than slowing every rollout barrier with a
ten-environment slice on only4cores. Seven9-thread processes per64-vCPU
box, one8-thread process per small box:448worker processes,
4480environment slots,4025simulation threads. 957worker slots support this
plus409Modal and up to100Chimera workers; recommend960slots/9600envcapacity.
The full launcher waits for this capacity BEFORE paid box provisioning.

The INITIAL pool20261108T020159Z used seven64-vCPU boxes with
six10-thread processes each:448reserved vCPUs/42workers/420simulationthreads.
Its first six workers passed real finite controls. A transient provider
placement-change error during a status read stopped that initial controller;
it exited and cleanup_errors=[] confirms its owned boxes were terminated.
Do not count its448vCPUs as current capacity. The full replacement adds tested
bounded retries for idempotent I/O; exhausted transient observation failures
are reported without killing healthy workers. Permanent errors and actual
worker exits still fail visibly. Current ownership is in `active.json`.

The main training thread owns the checkpoint-preserving learner capacity
expansion and publishes `target.json`. The replacement reads this route while
waiting for>=957worker slots before provisioning boxes. A new frozen root
or Python runtime is rejected for separate image qualification. Keep the
frozen model/physics/bank and training state unchanged.

The image uses the exact existing `modal-live/runtime.tar.gz` at its original
absolute paths, Python 3.12 amd64, and pinned worker requirements. The bridge
uses pinned TLS on both public legs with learner HMAC intact. Worker ticket
generation renews after learner outages. No account API key enters the image
or any Sailbox. Pool ownership and box IDs are in `active.json` and the dated
`receipt.json`; readiness counts are cumulative, not live capacity.

The first box must return all seven finite actual controls before the remaining
boxes start. The same native worker protocol validates the model signature.
Failures terminate owned boxes and preserve unrelated compute. There is no
one-hour pool cutoff. The main training owner must request pool shutdown on
training completion; this controller watches `stop-request.json`. The revised
full launcher also stops automatically after learner iteration10000.

```sh
/private/tmp/assembly-sail-live-sdk/bin/python -u robotics-assembly-3.0/operational/wowrobo-chimera-20261107/sail-live/launch.py
/private/tmp/assembly-sail-live-sdk/bin/python robotics-assembly-3.0/operational/wowrobo-chimera-20261107/sail-live/status.py
```

To stop this pool only, write `{"stop": true}` to `sail-live/stop-request.json`.
Do not launch a second owner, stop the learner, or cancel Chimera leases.
