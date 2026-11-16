"""Start bounded native worker pools or measure their network path."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path

from .bundle import build_bundle
from .pool import run_workers
from .probe import measure, serve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    h = commands.add_parser("hybrid", help="Supervise a mixed provider fleet")
    h.add_argument("--config", type=Path, required=True)
    h.add_argument("--ticket", type=Path, required=True)
    h.add_argument("--dry-run", action="store_true")
    for name in ("local", "chimera", "modal", "sail"):
        p = commands.add_parser(name, help=f"Run a bounded {name} CPU worker pool")
        p.add_argument(
            "--ticket",
            type=Path,
            required=True,
            help="Private server.json or probe.json from the learner host",
        )
        p.add_argument("--hosts", type=int, default=1)
        p.add_argument(
            "--workers",
            type=int,
            default=5 if name == "chimera" else 4,
            help="Persistent worker processes per host",
        )
        p.add_argument("--threads", type=int, default=2, help="Native physics threads per process")
        p.add_argument("--memory-mib", type=int, default=20480 if name == "chimera" else 16384)
        p.add_argument("--seconds", type=float, default=3600)
        if name in ("modal", "sail"):
            p.add_argument(
                "--region", required=True, help="Place workers close to the learner; provider-specific region"
            )
        if name == "modal":
            p.add_argument("--cpus", type=float, default=8, help="Physical-core reservation")
        elif name == "sail":
            p.add_argument("--sail-size", choices=("s", "m", "l", "ultra_experimental"), default="m")
        elif name == "chimera":
            p.add_argument("--run-dir", type=Path, required=True, help="Shared learner run directory")
            p.add_argument("--partition", default="cpu_preemptible")
            p.add_argument("--qos", default="preempt")
    p = commands.add_parser("bundle", help="Freeze source and installed geometry into a cached image input")
    p.add_argument("output", type=Path)
    p = commands.add_parser(
        "certificate", help="Create a self-signed certificate pinned by private worker tickets"
    )
    p.add_argument("directory", type=Path)
    p = commands.add_parser("probe", help="Measure batch RTT/throughput before allocating a training pool")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--ticket", type=Path)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--run-dir", type=Path, default=Path("runs/probe"))
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--advertise-host")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    args = parser.parse_args(argv)
    if args.command == "hybrid":
        from .hybrid import launch

        launch(args.config, args.ticket, args.dry_run)
    elif args.command == "certificate":
        args.directory.mkdir(parents=True, exist_ok=True)
        key, cert = args.directory / "key.pem", args.directory / "cert.pem"
        if key.exists() or cert.exists():
            raise FileExistsError("Refuse to replace an existing certificate")
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "30",
                "-subj",
                "/CN=assembly-so101",
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        key.chmod(0o600)
        print(json.dumps({"certificate": str(cert), "key": str(key)}))
    elif args.command == "bundle":
        print(json.dumps(build_bundle(args.output)))
    elif args.command == "probe":
        if args.serve:
            serve(args)
        elif args.ticket:
            print(
                json.dumps(measure(json.loads(args.ticket.read_text()), samples=args.samples, envs=args.envs))
            )
        else:
            parser.error("probe requires --serve or --ticket")
    else:
        if min(args.hosts, args.workers, args.threads, args.memory_mib, args.seconds) <= 0:
            parser.error("Pool resources and duration must be positive")
        if args.command == "local" and args.hosts != 1:
            parser.error("local runs one host; use --workers to select process count")
        if args.command == "modal" and (args.cpus <= 0 or args.workers * args.threads > args.cpus):
            parser.error("Worker threads exceed the Modal CPU reservation")
        if args.command == "modal" and args.seconds > 86340:
            parser.error("Modal pool duration must leave room within the function timeout")
        ticket = json.loads(args.ticket.read_text())
        if args.command in ("modal", "sail") and not ticket.get("tls_ca"):
            parser.error("Cloud workers require a TLS ticket")
        if args.command == "local":
            if ticket.get("kind") == "probe":
                print(json.dumps(measure(ticket)))
            else:
                print(json.dumps(run_workers(ticket, args.workers, args.threads, args.seconds)))
        else:
            from . import providers

            def terminate(*_):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, terminate)

            getattr(providers, args.command + "_pool")(args, ticket)


if __name__ == "__main__":
    main()
