# SGS state explorer

**Goal:** Inspect every saved entry in current-model banks through cached images and one native-pose 3D view.

From the repository root, with the existing SO101 runtime installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/so101_sgs_viewer.py
.venv/bin/python scripts/so101_sgs_viewer.py --serve
```

Open <http://127.0.0.1:8766>. For a remote Linux host, forward the port from your computer: `ssh -N -L 8766:127.0.0.1:8766 YOUR_HOST`. An exported folder needs only this static server; browsing never loads or steps MuJoCo. `--out` selects the export folder (default `artifacts_local/sgs-viewer`); `--banks` selects the parent containing published bank directories. Re-export to add new published banks/states. Content-keyed images are reused; the index is replaced only after all entries and source checks pass.

The initial collection is the prepared SGS bank. Choose **All current collections** to include older all-loose baseline entries with matching CAD. Historical CAD and stale validators are listed under the export identity; they are not rendered using current geometry. TRAIN and independent family EVAL are indexed completely. The loose-only EVAL alias is not counted twice. Provenance rows must exactly match the saved NPZ states, family and seed, and pass existing receipt checks.

Each entry has side, top, close-up and thumbnail images. The 3D inspector reuses Quest's geometry export and renderer, applying native body poses immediately without interpolation. Drag to orbit, wheel to zoom, Shift-drag to pan. **Focus part** and **Whole scene** change only the inspection camera. There is no physics replay in this version.

Filter by collection/task/family/split/qualification/review, search an ID or seed, click coverage cells, or use previous/next and left/right arrows. Review and flag marks stay in local browser storage; **Export my reviews** downloads an independent JSON file. They never change bank qualification or sampler weights. Missing qualification metrics in baseline banks are explicitly unrecorded, not zero.

The grid virtualizes visible rows and lazily loads thumbnails. Full images, metadata and the shared geometry load on selection; only eight detail records are cached. The lightweight index stays in memory. Source receipts include bank hashes, model/validator identity and unchanged-state checks for every entry. Image caches bind geometry, saved state/control identity, camera settings, resolution and exporter source. Generated files stay outside source Git.

Checks:

```bash
.venv/bin/python -m pytest -q tests/test_so101_sgs_viewer.py
node --test tools/sgs_viewer/core.test.mjs
```

The optional `browser.test.mjs` runs against the served export using an existing Playwright/Chromium installation. Set `PLAYWRIGHT_MODULE` to its `index.mjs` path and `SGS_BROWSER_EXECUTABLE` if needed. It checks images, filtering, review persistence after reload, native 3D and a separate 10,000-entry synthetic metadata fixture; reports/screenshots go to `artifacts_local/sgs-viewer-browser`. Synthetic entries exist only in that test browser's intercepted index.

**Local verification, 2026-11-07 UTC:** 50 qualified pilot states plus 320 current-CAD all-loose entries; 1,110 native images and 370 thumbnails, all decoded successfully. Every render passed exact saved-state parity. Eleven Python tests, three JavaScript tests and the browser check passed. The synthetic 10,000-entry browser opened in 371 ms with 15 active cards and about 15 MB JavaScript heap in local headless Chromium. These timings are local measurements, not remote-network or full 10,000-state rendering benchmarks.
