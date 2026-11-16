# 19,200-state completion return

These are separate from the completed1,800-bank helpers and viewer8878. They do not submit/cancel jobs or mutate the old runtime, existing banks, source assets, qualification outcomes or training.

## Allocated completion exporter

From the new controller's allocated step:

```sh
PYTHONPATH=/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4/robotics-assembly-3.0 \
  /large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python \
  /path/to/new/finalize.py \
  --root /large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1 \
  --project /large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4/robotics-assembly-3.0 \
  --plan /large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1/plan \
  --source-root /large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4
```

Defaults are640TRAIN+128EVAL per25distinctstarts =19,200. Optional `--train/--eval` parameterize both exporter and ingestion for bounded testing or later explicitly authorized quotas. The exporter checks the sealed manifest/model SHA, per-stage counts, finite states/controls,3positive finite friction columns and their category names, duplicate full snapshots, and equality to the unchanged source physical-model identity. It calls the frozen admitted-only publisher and exports under the **new** root's `exports/reset-bank-result.tar.gz` with a SHA receipt. Untouched Linux-native bank/receipts, new-plan launch/source records and old-runtime parity/assets provenance are preserved. No grids/vendor meshes are copied into the return archive.

New plan may use original `config.grasp_library` and original support/physics profile; those are copied by pointer/fallback for provenance. A stage's shard `run.json` must use the existing publisher-supported status `complete`, `pilot_complete` or `incomplete_rejections`; exact quotas are checked independently before publishing.

## One-shot local completion collector

When the parent has provided its live dedicated controller/read-allocation ID:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python \
  operational/reset-bank-19200-return-20261107/collector.py \
  --job-id <LIVE_CONTROLLER_ALLOCATION_ID>
```

It polls only that allocation through `srun --overlap`; it submits/cancels nothing. Three consecutive failed observations stop with explicit failure status. `--once` reads one status and returns if no completion receipt yet. The allocation must remain live through the read; if the controller exits first, supply another explicitly authorized live read-allocation ID. The collector does not invent/reuse an unrelated allocation.

Defaults create new paths:

- `outputs/reset-bank-19200-completion-20261107/{status.json,return-receipt.json,ingest.log,viewer.log}`
- `outputs/reset-bank-19200-chimera-returned-v1` — untouched Linux artifacts
- `assets_local/wowrobo/randomization-bank-19200-chimera-display-v1` — verified Mac display only
- First available viewer port in8891–8910;8878 is untouched.

Collector waits for an exact19,200/640/128 completion receipt, downloads the archive, calls ingestion, starts a separate viewer, and verifies its catalog exposes19,200variations before `status=complete`. It does not open browser UI; the parent can do that after readiness.

`ingest.py` can also run directly with `--archive`, `--sha256`, `--output`, `--display` and optional `--workspace`. It verifies the archive/file hashes, unchanged raw assets, original Mac point fingerprints, per-stage admitted display quotas, native payload counts and friction values. It maps the **old immutable runtime root** to the Mac workspace, rather than the new result-root path. Only display XML loading paths and top-level model-loading SHA fields change; nested native qualification metadata and returned native-bank files remain untouched Linux provenance. It does not claim repeated native admission on the Mac.

## Verification

Ruff and allCLIhelp checks pass. The new native quota validator passes the actual completed1,800-bank with explicit60/12 testing arguments and rejects that same bank when640/128 are requested; it cannot relabel old data as19,200. Archive checksum/path-traversal rejection checks and collector malformed-allocation-ID checks pass without networking. Full19,200 publication and viewer readiness remain to be verified against the new completed bank.


## Display metadata memory bound

Full ingestion now streams `variation_by_task` through `compact.py` before normal JSON loading. Full Linux manifests, receipts, source artifact archives and qualification hashes remain untouched. Identical frozen physics profiles are shared once by SHA; duplicate full physical metrics are replaced only in the display copy by exact critical minimum witnesses, load/retention/control/overlap/threshold fields. Noise, seeds, slots, friction, state/control hashes, support/grasp poses and admission outcomes are unchanged. `compaction.json` records source/final SHA, byte counts and removed duplicate keys. This same compactor was verified on the actual18,911-row preview (2,159,647,360-byte Linux manifest →188,827,247-byte final Mac display) with original point/asset contracts and all accepted counts passing. Existing collector invokes ingestion in a fresh subprocess, so no collector restart is needed.
