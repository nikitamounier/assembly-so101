# Run the complete assembly viewer

## Setup

Check out `nikita/so101-acceleration-cleanup`, then run from the repository root:

```sh
bash scripts/setup_viewer.sh
bash scripts/run_viewer.sh
```

Open **http://localhost:8876/** on that computer. `localhost` refers to your own
computer; the GitHub repository supplies the source and assets, not a hosted
website. Setup requires internet access for Python packages. Subsequent launches
use the checked-in assets and frontend without downloading models or rebuilding
CAD. The first observation overlay can take longer while source surface samples
are prepared.

Setup creates `.venv-viewer` using Python 3.12. Install `uv` or provide
`python3.12` first. Apple Silicon macOS and Linux use the same scripts; Windows
can run them under WSL. Neither CUDA nor a native collision-plugin compiler is
required for this saved-state viewer.

The launch script also works from another working directory. To choose a free
port or open the browser automatically:

```sh
bash scripts/run_viewer.sh --port 8878 --open
```

Use Ctrl+C to stop the viewer. The server binds only to `127.0.0.1`.

## What is included

- Both camera-equipped SO101 arms and all repaired target components.
- The exact `randomization-bank-repaired-cover-support-v2` collection: 25
  nominal/reference states and 75 generated variations, 100 states total.
- Original validation and variation metadata, including failures.
- Scene rendering with optional advanced MuJoCo inspection.
- Both final-module comparison clips (real assembly and physics simulation).
- Seating reward boxes/tolerances and the configured 448-point policy overlay.
- The latest printable pin in `cad/wowrobo/`.

Select a state in assembly order, switch between its reference and variations,
and use the filters to inspect validation results. Scene is the default view.
“Advanced inspection” starts MuJoCo Studio on demand; “Close inspection” unloads
its viewer and stops its extra process. Leaving the page also requests shutdown.
Both views inspect the same saved state without advancing physics.

## Runtime identity

`runtime/wowrobo-viewer/` contains the relative-path XML, 31 referenced mesh files,
50 saved-state archives, frozen point/reward configurations and validation
metadata. `snapshot.json` records SHA-256 checksums for every runtime file;
launch verifies them before loading anything. A missing or edited file produces
a clear error instead of silently loading a different scene. Restore it with
`git restore runtime/wowrobo-viewer` if you want the published snapshot again.

The original model identity is
`8f63127e728cac0eee631384d3f181ba7b958d8c248492a419bf961996fd6b9e`.
The portable XML has a different file checksum because its absolute mesh paths
were replaced with relative paths. Saved states, mesh bytes, physical parameters
and the observation/reward contracts are preserved; the loader retains their
original model identity. This does not redirect your local viewer to another
bank or adopt later equilibrium experiments.

The included geometry is sourced from the Apache-2.0 SO-ARM100 and MIT ICL-so101
projects; revisions, source locks, notices and full licenses are in
`runtime/wowrobo-viewer/licenses/`. Original vendor STEP files, training
checkpoints, receipts from compute providers, credentials and unrelated runs
are not required and are excluded.

## Development

The frontend is already built and committed. After editing it:

```sh
npm --prefix web ci
npm --prefix web test
npm --prefix web run build
```

The normal viewer CLI remains available for other local banks. For the frozen
collection, the explicit command is:

```sh
.venv-viewer/bin/python -m assembly_policy.viewer \
  --snapshot runtime/wowrobo-viewer --port 8876
```

To export a new source layout, use `scripts/export_viewer_snapshot.py --help`
and a new output directory. Do not regenerate or overwrite this version during
ordinary launch. The files under `assembly-reference/` preserve companion
assembly source for review; this viewer does not import that copied package.

The viewer reports geometric and recorded validation evidence. Hardware motor
calibration and full dynamic training admission remain separate work.
