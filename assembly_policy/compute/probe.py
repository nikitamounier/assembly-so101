"""Measure the real persistent transport with compact observation-sized batches."""

from __future__ import annotations

import json
import os
import secrets
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

from ..rl.transport import TLSListener, connect


def payload(envs):
    return dict(
        frame=np.zeros((envs, 53), np.float32),
        task=np.zeros(envs, np.int64),
        tcp_p=np.zeros((envs, 2, 3), np.float32),
        tcp_R=np.zeros((envs, 2, 3, 3), np.float32),
        part_p=np.zeros((envs, 11, 3), np.float32),
        part_q=np.zeros((envs, 11, 4), np.float32),
        finger=np.zeros((envs, 2), np.float32),
        reward=np.zeros(envs, np.float32),
        done=np.zeros(envs, bool),
    )


def serve(args):
    if args.bind_host not in ("127.0.0.1", "localhost") and not args.tls_cert:
        raise ValueError("External probes require TLS")
    key = secrets.token_bytes(32)
    address = (args.bind_host, args.port)
    listener = (
        TLSListener(address, authkey=key, certfile=args.tls_cert, keyfile=args.tls_key)
        if args.tls_cert
        else Listener(address, authkey=key)
    )
    ticket = dict(
        host=args.advertise_host or listener.address[0],
        port=listener.address[1],
        key=key.hex(),
        time=time.time(),
        kind="probe",
    )
    if args.tls_cert:
        ticket["tls_ca"] = Path(args.tls_cert).read_text()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "probe.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(ticket, stream)
    print(json.dumps({"ticket": str(path), "port": ticket["port"]}), flush=True)
    try:
        while True:
            conn = listener.accept()
            try:
                while True:
                    message = conn.recv()
                    if message is None:
                        break
                    conn.send(message)
            except (EOFError, OSError):
                pass
            finally:
                conn.close()
    finally:
        listener.close()


def measure(ticket, *, samples=100, envs=32):
    if samples < 1 or envs < 1:
        raise ValueError("Positive samples and environments are required")
    conn = connect(ticket)
    times = []
    message = payload(envs)
    try:
        for index in range(samples + 5):
            start = time.perf_counter()
            conn.send(message)
            received = conn.recv()
            if received["frame"].shape != message["frame"].shape:
                raise ValueError("Probe response shape differs")
            if index >= 5:
                times.append(time.perf_counter() - start)
        conn.send(None)
    finally:
        conn.close()
    p50, p95 = np.percentile(times, [50, 95]) * 1000
    size = sum(array.nbytes for array in message.values())
    return dict(
        samples=samples,
        envs_per_worker=envs,
        array_payload_bytes=size,
        round_trip_p50_ms=float(p50),
        round_trip_p95_ms=float(p95),
        payload_round_trip_mbps=float(2 * size * 8 / (np.mean(times) * 1e6)),
        control_steps_32_network_floor_s=float(32 * p50 / 1000),
        compute_ms_for_under_10pct_network=float(9 * p95),
        scope="Persistent batch echo; measures transport/serialization, excludes physics and policy",
    )
