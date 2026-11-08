set -eu
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/publish_wowrobo_equilibrium_resets.py --source assets_local/wowrobo/photo-resets-repaired-cover-support-v2 --results outputs/equilibrium-reset-20261107/minimum-v6 outputs/equilibrium-reset-20261107/minimum-v6-source-fingers outputs/equilibrium-reset-20261107/minimum-v6-source-fingers-refined outputs/equilibrium-reset-20261107/minimum-v6-final21 --output assets_local/wowrobo/initial-valid-resets-v1 --minimum-only
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-reset-20261107/verify_published.py
