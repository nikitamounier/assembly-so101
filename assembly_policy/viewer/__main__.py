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
    parser.add_argument("--studio-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--open", action="store_true", help="Open the browser")
    args = parser.parse_args()
    if not (STATIC / "index.html").exists():
        parser.error("Build the viewer frontend: npm --prefix web ci && npm --prefix web run build")
    viewer = ResetApplication(args.banks or [ROOT / "data/so101"], args.studio_python, args.port)
    if args.open:
        webbrowser.open(f"http://localhost:{args.port}")
    print(
        f"Reset browser: http://localhost:{args.port} · {len(viewer.catalog.entries)} compatible saved states",
        flush=True,
    )
    web.run_app(viewer.app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
