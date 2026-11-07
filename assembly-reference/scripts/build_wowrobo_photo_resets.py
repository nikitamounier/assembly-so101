"""Export photo-directed kinematic candidates for inspection, never training admission."""

import argparse
from pathlib import Path

from assembly_policy.so101.wowrobo_photo_resets import ROOT, PhotoResetSequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=ROOT / "assets_local/wowrobo/components-v2"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets_local/wowrobo/photo-resets-v3"
    )
    args = parser.parse_args()
    print(PhotoResetSequence(args.assets).export(args.output))
