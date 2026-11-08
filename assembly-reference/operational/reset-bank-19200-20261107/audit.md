# 19,200-reset expansion: immutable quota and seed audit

The authorized target is **25 stages × (640 TRAIN + 128 EVAL)**: **16,000 training, 3,200 evaluation, 19,200 accepted variations total**. Nominal viewer references are additional rows, not part of either variation quota.

## Recommended dataset boundary

Generate a fresh bank in `reset-bank-19200-20261107-v1/plan` using the frozen existing runtime and original native physical profile. Keep the completed 1,800-state bank immutable as a separate legacy dataset and proof source. Do not copy its attempt receipts into the new source identity, rewrite its qualification identities, or count them as newly sampled 640/128 slots.

Old states remain physically qualified for their recorded model. That does not make them draws from the new quota-conditioned sampler:

- Old TRAIN had24 easy/24 medium/12 hard slots; new TRAIN needs256/256/128. Appending new slots60..639 without remapping would give220/280/140 combined, not40/40/20.
- Old EVAL used12 difficulty strata; new EVAL uses128. An old EVAL slot and its sampled difficulty are generally not a draw from the same-numbered new stratum.
- Canonical run identities include the configuration and quotas. An old receipt cannot simply acquire the new identity. Legitimate reuse would require an explicitly multi-source/rebinned dataset manifest and careful stratum accounting, which adds complexity and is unnecessary for a clean fresh bank.

## Mathematically correct fresh sampling

Keep the canonical numeric seed mapping unchanged:

```text
seed = slot + attempt * quota_for_split
quota_train = 640
quota_eval = 128
```

This ensures `seed % quota == slot` on every retry and `seed // quota == attempt`. TRAIN slots0..255 sample difficulty in[0,1/3), slots256..511 in[1/3,2/3), and slots512..639 in[2/3,1]. EVAL slotj samples `(j + U)/128`, forj=0..127. Rejection retries stay in their original stratum rather than silently becoming easier.

Use a fresh, explicit RNG namespace for both TRAIN and EVAL while leaving the numeric seed unchanged. For example:

```text
key = split + ':' + stage_id + ':' + integer_seed
namespaced_key = 'wowrobo-reset-19200-20261107-v1' + NUL + key
effective_seed = little_endian_uint64(SHA256(namespaced_key)[0:8])
```

A launch-only module-local `hashlib` proxy can prefix **only** the exact sampler key, delegating XML/CAD/grasp/manifest digest calls unchanged. Do not globally modify `hashlib.sha256`: state, control, file and physical-model hashes must retain ordinary SHA256 semantics. The randomizer currently calls SHA256 for the sampler key **and** scene/component checks, so the proxy must distinguish them. Do not pass a huge numeric seed offset as a substitute: it also changes modulo strata and makes `seed//quota>0` select retry/alternate grasps on the first draw.

Record the namespace, effective seed, stage, split, numeric seed, slot and attempt in each candidate's metadata; export a per-row `seed_namespace` column as well as numeric `seed`. Freeze/hash the launch wrapper and bind that hash into the new configuration/provenance. This makes wrapper-based RNG behavior reviewable even though the canonical source guard does not itself inspect a module-local runtime proxy. Different namespaces must not be conflated when numeric seeds coincide.

Continue native overlap and actual held-grip retention admission, passive joints, original contact fields, actuator limits, and approved noise amplitudes. A candidate is counted only after admission. Distinct source identities and explicit seed namespaces preserve reproducibility without mutating the old runtime or old bank.

## Partition and packed-job shape

Twelve partitions per stage yield exactly300 independent workers:

| Partition IDs | TRAIN slots | EVAL slots | Total per worker |
|---|---:|---:|---:|
|0..3|54|11|65|
|4..7|53|11|64|
|8..11|53|10|63|

Use `slot % 12 == partition` independently in TRAIN and EVAL. These12 sets are disjoint and their union covers all768 slots per stage. Preserve global slot indices and numeric seed mapping; do not restart per-partition slot numbering atzero. A worker proves completion of its own63–65 accepted slots, not the full-stage manifest's640/128 quota. Final stage publication occurs only after all12 partitions merge with verified source/state/control hashes and exactlyone accepted receipt per split/slot.

A practical packed shape is **12 workers per job,24 CPUs and192 GiB**, one node per job:25 jobs for300 active workers,600 allocated CPUs,4,800 GiB total reservation. This stays below a100-job ceiling only if existing jobs leave room; count all currently charged jobs rather than assuming the full ceiling is free. Actual node capacity/availability still governs how many jobs can run immediately.

Bind each child to its own two allocated CPUs **before** importing Embree/MuJoCo or constructing mesh caches. OMP/MKL/OpenBLAS thread limits alone do not reliably constrain Embree/TBB initialization. At approximately10 GiB observed RSS per worker,16 GiB reserved per worker provides useful headroom; measure actual peak RSS on hard final/insertion phases and reduce packing if needed, rather than accepting OOM or changing physical checks.

Keep inherited per-partition file leases, atomic receipts, global claim/close fencing and source-identity checks from the successful previous farm. Packed jobs should requeue and resume from durable receipts. The final seal must independently verify16,000 TRAIN and3,200 EVAL, duplicates, matching physical model/state specification, saved friction and all native admission proofs. No failed candidate or nominal reference may fill a quota.

## Local mathematical checks performed

Read-only inline regression against the current quota helpers passed:

- TRAIN counts256/256/128 for attempts0,1 and159.
- Every EVAL difficulty stays within its own1/128 stratum at those attempts.
- Numeric seed modulo and retry ordinal are unchanged.
- Twelve partition sets have sizes65/64/63 as above, are disjoint and cover exactly768 split/slot keys.
- A fresh namespace changes the RNG key digest.

No old source, active job, remote runtime or completed bank was changed for this audit. This document is the owned artifact; workers, resource allocation and finalizer implementation remain with the main task.
