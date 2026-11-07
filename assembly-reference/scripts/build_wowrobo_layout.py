"""Generate inspectable nominal and randomized initial workcell layouts."""

import argparse
from pathlib import Path

from assembly_policy.so101.wowrobo_layout import ROOT, InitialLayout

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets", type=Path, default=ROOT / "assets_local/wowrobo/components-v2"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "assets_local/wowrobo/initial-layout-v2"
    )
    p.add_argument("--count", type=int, default=32)
    a = p.parse_args()
    layout = InitialLayout(a.assets)
    signature, bank = layout.export(a.output, a.count)
    print(f"{a.count} layouts: {bank}; model {signature}")
