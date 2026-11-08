# General empty-pair separation and associative surface cache

Implemented and qualified November7,2026, after the hinge profile. Canonical changes promoted locally; live runtime belongs to main-run owner.

## Changes

1. Extend the existing conservative whole-grid separation certificate beyond named passive motor stops to external grid-SDF pairs. Only skip ordinary search when margin<=0 and a bounded-motion geometric proof succeeds. Unknown, near-contact, non-grid or budget-exhausted cases retain full search. Supplemental surface contact remains active: the volume certificate alone does not justify skipping its thresholded surface samples.
2. Replace each single swept-candidate cache slot with four relative-pose regions. Look up all four using the existing conservative coverage test, and replace round-robin only when none covers the query. This handles alternating-world eviction without relying on mjData pointer identity or approximating contact geometry. Candidate sample order and exact contact arithmetic remain unchanged.

Three promoted source files:
- assembly-so101/plugin/so101_surface_tree.h
- robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_cache.h
- robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py

`source-before/` holds previous canonical sources; `promote.py` applies guarded changes and records final hashes. The primary tested native source differs from promoted source only in explanatory comments. Native compilation passed; updated Python syntax passed.

## Correctness

- Four-stage/two-world/three-repeat ablations: all120 saved state/reward/flag arrays identical for separation-only, cache-only and combined candidates against v13 baseline.
- Exhaustive independent mode2 surface scan vs accelerated mode1 across29saved stages/two worlds each:174 acceleration/sensor/contact arrays exactly equal. Raw receipt mistakenly described25stages; oracle-original.json preserves it, corrected receipt derives29from174/6.
- Full actual32step/64transition PPO from same archived native32policy+Adam, seeded diagnostic resets: policy and Adam checkpoint tensors/counters exact; scientific metrics equal; unstable0, same one safety termination/three episodes. This is a diagnostic update, not exact production continuation.
- Ten-world/ten-thread packed replay:80saved output arrays exact. Two nominal/variation reset states are repeated five times, with distinct action rows; this checks worker shape but does not reproduce ten independent production resets.

## Performance

Matched actual-PPO total117.473→92.607seconds:1.2685x throughput. Collection116.896→92.074seconds. Process CPU191.912→149.393seconds,22.15%less. One matched pair only; no fleet speedup or NIST ratio claim. Both processes2logicalSMT siblings[2,98]/onephysicalcore; native2threads.

Ten-world/ten-thread packed sum of four warmed-stage intervals8.6352→6.2667seconds:1.3780x. Same affinity[2,14,16,18,94,98,110,112,114,190]. PeakRSS8350.16→8375.78MiB (+25.61MiB). One measured warm interval/stage, shape test rather than production-weighted forecast.

Two-world four-stage warm interval sums: laterbaseline7.5415seconds, separation5.9703, cache7.2162, combined4.9328 (~1.53xcombined). First baseline14.4341seconds was much slower; preserve raw results and use the later repeat rather than quoting the inflated first comparison. No statistically robust independent component speedups claimed.

Separate atomic-counter instrumentation (not primary throughput runs): originalcache70832hits/7888rebuilds, four-waycache76246hits/2474rebuilds. Rebuilds reduced68.64%; retained candidates on rebuild3403367→1313220 (61.41%less). This directly supports the alternating-region eviction hypothesis. Counters aggregate native threads and main refresh; not per-world attribution.

All measurements .5ms/80substeps, frozen native-v13 source/bank, isolated existing3598717 allocation. No source/checkpoint/process/CPU-role change to live training. Artifacts remote: ROOT/native-contact-optimization-20261107/contact-separation-20261107. `experiment.py`, `qualify_next.py`, `oracle.py`, `packed.py`, `final_checks.py` preserve commands and harness. Source builds private; compilation/startup excluded. Primary early timing included only wall time; PPO additionally records process CPU time.

## Deployment and next work

Main owner must stage these three coupled source files in a new versioned runtime, repin source/library identity, run its full-bank and representative production-worker checks, and continue from the current native checkpoint under existing authorization. Do not replace or patch the running release in place. The enlarged19200-state bank was not used in these benchmarks.

Implemented this iteration: empty-pair certificates and cache retention. Positive-contact patch maintenance, analytic feature substitutions, structural attachment simplification and fleet scheduling remain research/implementation follow-ups. Keep actual contact searches when separation cannot be proved; no geometry/budget approximation was introduced.
