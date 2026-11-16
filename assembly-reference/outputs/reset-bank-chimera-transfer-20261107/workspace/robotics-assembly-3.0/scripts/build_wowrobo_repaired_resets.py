"""Export versioned corrected-CAD references without changing frozen photo poses."""
import argparse
import json
from pathlib import Path

from assembly_policy.so101.wowrobo_repaired_resets import (
    DEFAULT_ASSETS,
    DEFAULT_OUTPUT,
    RepairedResetSequence,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,default=DEFAULT_ASSETS)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--no-ik",action="store_true")
    args=p.parse_args()
    print(json.dumps(RepairedResetSequence(args.assets).export(args.output,solve_arms=not args.no_ik),indent=2))
