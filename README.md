# SO101 assembly viewer

Run the complete WowRobo viewer that was shown at `localhost:8876`: both arms,
the repaired assembly parts, 25 reference poses and 75 variations, validation
details, seating reward regions, policy point clouds, both renderers and the
final-module comparison clips.

```sh
git clone --branch nikita/so101-acceleration-cleanup https://github.com/nikitamounier/assembly-so101.git
cd assembly-so101
bash scripts/setup_viewer.sh
bash scripts/run_viewer.sh
```

Open **http://localhost:8876/**. Setup uses Python 3.12 with
[uv](https://docs.astral.sh/uv/) if installed, or `python3.12` and pip. macOS and
Linux are supported; Windows users can use WSL. No Node installation, CAD
downloads, GPU, training checkpoint or sibling repository is required.

[Viewer setup and controls](docs/VIEWER.md) ·
[Printable stabilizer pin](cad/wowrobo/README.md) ·
[Source overview](docs/REVIEW_SNAPSHOT.md)

The versioned runtime is an inspection snapshot, not a physically admitted
training bank or a hardware controller. It preserves the actual validation
statuses, including rejected states.

## Current assembly and training source

The current native MuJoCo and experimental Warp code, reset-bank generation, tests and research reports are included in [`assembly-reference/`](assembly-reference/). See [source overview](docs/REVIEW_SNAPSHOT.md) and [snapshot hashes](docs/ASSEMBLY_SOURCE_SNAPSHOT.json). This work is published to **nikitamounier/assembly-so101**; upstream names in historical notes describe provenance. Generated training data, checkpoints, credentials and original vendor CAD stay local.
