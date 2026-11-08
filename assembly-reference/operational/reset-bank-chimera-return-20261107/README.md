# Completion-only Linux bank return

These helpers do not modify generation, training, source assets or admission outcomes.

## On Chimera

Upload `finalize.py` to the isolated job root's `launch/finalize.py`. The allocated driver calls it only after the sealed native bank reaches all25×(60TRAIN+12EVAL)=1,800 states. It independently checks quotas/finite states, publishes the existing admitted-only review collection, and creates:

- `exports/reset-bank-result.tar.gz`
- `exports/return-receipt.json` with exact SHA256, byte count and Slurm job ID.

The archive contains `review/` (25 nominal references plus1,800 admitted variations), untouched `native-bank/`, model/asset/parity provenance and per-file return hashes. Runtime grids/mesh assets are not duplicated. A completed export is idempotent if its checksum is unchanged; interrupted publication folders are preserved under a timestamped name.

## Back on the Mac

Download the archive and receipt. Use the receipt's exact SHA:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python \
  operational/reset-bank-chimera-return-20261107/ingest.py \
  --archive /tmp/reset-bank-result.tar.gz \
  --sha256 <SHA256_FROM_RETURN_RECEIPT> \
  --output outputs/reset-bank-chimera-returned-v4 \
  --display assets_local/wowrobo/randomization-bank-1800-chimera-display-v4
```

Both output directories must be new. Ingestion checks the archive and every returned file; requires passed Linux model-array parity; verifies frozen raw native asset hashes against local OBJ/STL/bin files; relocates only the display XML's loading paths; updates only top-level display model-signature fields; verifies the original point contract against the actual Mac-compiled model without regenerating its fingerprints; and checks all60/12 per-stage admitted rows. All native-bank files and qualification metadata remain unchanged Linux provenance. It does **not** claim repeated native admission on the Mac.

After verification, open the usual viewer using the new `--display` directory as `--layout`. Its local-relocation.json records source Linux identities and the Mac display signature.
