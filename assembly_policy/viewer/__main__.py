"""Launch a local reset browser with interchangeable rendering backends."""

import argparse
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

from .server import STATIC, ResetApplication

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Inspect saved SO-101 environment resets")
    parser.add_argument(
        "--banks", type=Path, action="append", help="Bank directory or parent; repeat for multiple roots"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--scene", choices=("so101", "cube-trial", "wowrobo-components", "wowrobo-layout"), default="so101"
    )
    parser.add_argument("--components", type=Path, help="Generated WowRobo component directory")
    parser.add_argument("--layout", type=Path, help="Generated initial workcell layout directory")
    parser.add_argument("--snapshot", type=Path, help="Self-contained viewer snapshot directory")
    parser.add_argument("--studio-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--open", action="store_true", help="Open the browser")
    args = parser.parse_args()
    if args.snapshot and (args.banks or args.layout or args.components or args.scene != "so101"):
        parser.error("Use --snapshot on its own, without --scene, --banks, --layout or --components")
    if not (STATIC / "index.html").exists():
        parser.error("Build the viewer frontend: npm --prefix web ci && npm --prefix web run build")
    default_banks = ROOT / ("data/cube_trial" if args.scene == "cube-trial" else "data/so101")
    options = {}
    if args.snapshot:
        from .snapshot import load_snapshot

        try:
            snapshot, manifest = load_snapshot(args.snapshot)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Cannot load viewer snapshot: {exc}")
        args.scene = snapshot.pop("scene")
        default_banks = snapshot.pop("roots")
        options.update(snapshot)
    if args.scene == "wowrobo-components":
        if args.components is None:
            parser.error("--components is required for the WowRobo component review")
        from .components import component_collection

        model, signature, bank = component_collection(args.components)
        options = {"model": model, "model_signature": signature}
        default_banks = bank
    if args.scene == "wowrobo-layout" and not args.snapshot:
        if args.layout is None:
            parser.error("--layout is required for the initial workcell")
        from .components import layout_collection

        model, signature, bank = layout_collection(args.layout)
        options = {"model": model, "model_signature": signature}
        default_banks = bank
        import json

        for key, filename in (("point_config", "wowrobo_point_features.json"), ("reward_config", "wowrobo_reward_regions.json")):
            contract = args.layout / filename
            if contract.exists():
                options[key] = json.loads(contract.read_text())
    viewer = ResetApplication(
        args.banks or (default_banks if args.snapshot else [default_banks]),
        args.studio_python,
        args.port,
        scene=args.scene,
        **options,
    )
    if args.snapshot and len(viewer.catalog.entries) != manifest["state_count"]:
        parser.error("Viewer snapshot state count differs from its manifest")
    if args.open:
        webbrowser.open(f"http://localhost:{args.port}")
    print(
        f"Reset browser: http://localhost:{args.port} · {len(viewer.catalog.entries)} compatible saved states",
        flush=True,
    )
    web.run_app(viewer.app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
