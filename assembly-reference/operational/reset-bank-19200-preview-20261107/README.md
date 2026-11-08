# Accepted-only partial snapshot

`snapshot.py --root RUNROOT --output RUNROOT/previews/<unused-tag>` runs only inside an authorized allocated Slurm step. It does not lease/stop workers or change generation files. It enumerates atomic partition receipts, selects native-admitted entries, checks stage/split/slot64-way ownership, namespace `wowrobo19200-v1`, native model `1b071d9b...`, finite/hash-correct qualified and review archives, and deduplicates `(stage,split,slot)`.

For every selected row it converts the actual native INTEGRATION state with the matching physical/visual models and checks the existing display archive has identical qpos/qvel/controls (only its clock may differ). The verified immutable display archive is then reused directly. Original accepted receipts and qualified NPZ bytes are retained in one `accepted-source-artifacts.tar`, avoiding thousands of temporary NFS writes. Combined native snapshot arrays and all25reference poses accompany accepted-only displayed variations.

Output is a frozen partial preview, always `complete=false, preview_only=true`, never a fullquota seal. The archive `<tag>.tar.gz` and receipt `<tag>.receipt.json` contain exact accepted counts and hashes. New completions after enumeration remain for the next snapshot.

`ingest-preview.py --archive ARCHIVE --sha256 SHA --output NEW_RETURN_DIR --display NEW_DISPLAY_DIR` verifies hashes, native model/namespace, original rawassets/pointcontract, per-stage partial counts≤640TRAIN/128EVAL, native payloads/friction and actual accepted totals. It preserves untouched Linux provenance, maps only the display XML paths to the local workspace and changes top-level display model-loading signatures. It does not claim repeated native admission on the Mac. Empty stage/split variation sets are allowed; all25existing reference poses remain reviewable.

Snapshot execution and local viewer startup are coordinated by the parent; these helpers submit/cancel no jobs and do not overwrite existing banks/viewer8878.


## Actual publication

Snapshot18,911accepted rows,15,875TRAIN+3,036EVAL,25reference poses, partialtarget19,200. Original firstsnapshot hit48GiB OOM; its incomplete folder was preserved. The corrected snapshot reuses one native/visual data pair and identical immutable profile maps, with measured peak10.0GiB. Current source snapshot accepted-20261108T045310Z passed all state/control/pose/model/namespace checks. Original Linux manifest2.16GB and all native/original receipt artifacts remain byte-verified. Local display-only streaming compaction yields188.8MB, with one shared frozen profile and exact critical witness/noise/control/pose metadata. Local151rawassetSHA and448-point contract checks pass. Viewer http://localhost:8892/ has18,911nativeeligiblevariations+25refs; fullMuJoCoUI remains enabled. Publication receipt outputs/reset-bank-19200-preview-20261107/preview-status.json. No generation/training modifications.
