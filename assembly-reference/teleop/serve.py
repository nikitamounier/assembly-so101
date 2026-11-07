"""Launch the shared SO101 simulation for Meta Quest; local simulation only, disabled at start."""
import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np

from .backend import HERE, ROOT, AssemblyFrames, AssemblyRuntime, load_scene
from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from .runtime import Journal, make_handler


def resume_description(model, description, path):
    """Restore measured physics only at process initialization, without authority."""
    saved = json.loads(Path(path).read_text())
    qpos = np.asarray(saved['qpos'], dtype=float)
    qvel = np.asarray(saved['qvel'], dtype=float)
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Resume state must match native dimensions and contain finite values')
    overlap = sampled_motor_base_overlap(model,qpos,ROOT/'assets_local')
    if overlap>.0005:
        raise ValueError(f'Cannot resume motor/base material overlap ({overlap*1000:.2f} mm); start the accepted loose layout')
    return dict(description, home_qpos=qpos.tolist(), initial_qvel=qvel.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18883)
    parser.add_argument('--seconds', type=float, help='Bounded diagnostic run; omitted for interactive use')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--record', action='store_true', help='Record full-rate state/input JSONL to output (off by default)')
    parser.add_argument('--resume-state', type=Path, default=os.environ.get('SO101_ASSEMBLY_RESUME_STATE'),
                        help='Initialize native poses/velocities from a local state; starts Disabled')
    parser.add_argument('--quest', action='store_true', help='Connect an authorized Quest over USB after startup')
    parser.add_argument('--adb', help='ADB executable (or TELEOP_ADB / adb on PATH)')
    parser.add_argument('--serial', help='Select one authorized Quest when multiple are attached')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('port must be between 0 and 65535')
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds<=0):
        parser.error('seconds must be positive and finite')
    model, description = load_scene()
    if args.resume_state:
        description = resume_description(model, description, args.resume_state)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'runs/teleop' / stamp
    journal = Journal(output) if args.record else None
    runtime = AssemblyRuntime(model, description, journal=journal)
    runtime.prepare_tasks()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runtime, HERE/'web'))
    server.daemon_threads = True
    http = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.1}, name='assembly-http')
    frames = None
    signal.signal(signal.SIGTERM, lambda *_: runtime.stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: runtime.stop_event.set())
    try:
        http.start()
        runtime.start()
        frames = AssemblyFrames(model, runtime.cameras)
        print(json.dumps(dict(url=f'http://localhost:{server.server_port}', backend='assembly', output=str(output))), flush=True)
        connection_attempted = False
        start = time.monotonic()
        while not runtime.stop_event.is_set() and (args.seconds is None or time.monotonic()-start < args.seconds):
            snapshot = runtime.measured_snapshot()
            if snapshot is not None:
                images, capture = frames.render(snapshot)
                runtime.cache_frames(images, capture)
                if args.quest and not connection_attempted:
                    from .connect_quest import connect
                    connection_attempted = True
                    try:
                        connect(port=server.server_port, adb=args.adb, serial=args.serial)
                    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
                        print(f'Quest USB connection: {error}. Simulator remains running. Connect later with python -m teleop.connect_quest --port {server.server_port}.',file=sys.stderr,flush=True)
            runtime.stop_event.wait(.025)
        if runtime.owner_error:
            raise RuntimeError(runtime.owner_error)
    finally:
        runtime.close()
        server.shutdown()
        server.server_close()
        http.join(timeout=3)
        if frames:
            frames.close()
        if journal:
            journal.write('events', dict(event='closed', health=runtime.get_health()))
            journal.close()


if __name__ == '__main__':
    main()
