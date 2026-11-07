"""Export exact frozen derived CAD into an isolated review component collection."""

import argparse
import json

from assembly_policy.so101.wowrobo_repaired_assets import (
    DEFAULT_OUTPUT,
    DEFAULT_READY,
    export_repaired_assets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ready", default=DEFAULT_READY)
    args = parser.parse_args()
    path, report = export_repaired_assets(args.output, ready_path=args.ready)
    print(json.dumps({"output": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
