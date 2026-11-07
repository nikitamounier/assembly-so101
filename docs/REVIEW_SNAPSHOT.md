# Collaboration snapshot — November 7, 2026

Current SO101 state/variation viewer, cube-trial and hybrid compute source for review. The canonical assembly geometry and intent live at https://github.com/Alessandro-Maniscalco/robotics-assembly-3.0 .

Frontend checks: `cd web && npm ci && npm test`. Python setup is defined in `pyproject.toml`; provider dependencies are listed in `requirements-providers.txt`. The CLI entry points are `python -m assembly_policy.viewer` and `python -m assembly_policy.compute`; inspect `--help` for options.

Generated banks, vendor assets, checkpoints, run receipts and credentials are excluded. CAD-backed viewer operation requires the matching private/local runtime assets and bank. Viewer reference and variation statuses are review evidence; they do not establish physical training admission. Canonical repaired resets still have material, settling and support-retention failures. No main assembly launch or experiment change is part of this publication.
