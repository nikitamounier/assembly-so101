"""Exercise encrypted framing, certificate pinning and compact payload semantics."""

import concurrent.futures
import subprocess
from multiprocessing import AuthenticationError

import numpy as np
import pytest

from assembly_policy.compute.probe import payload
from assembly_policy.rl.remote import _pack
from assembly_policy.rl.transport import TLSListener, connect


@pytest.fixture
def certificate(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=assembly-so101",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def test_persistent_tls_round_trip_and_selector_readiness(certificate):
    cert, key = certificate
    listener = TLSListener(("127.0.0.1", 0), authkey=b"run-key", certfile=cert, keyfile=key)
    ticket = dict(host="127.0.0.1", port=listener.address[1], key=b"run-key".hex(), tls_ca=cert.read_text())

    def echo():
        conn = listener.accept()
        try:
            for _ in range(3):
                value = conn.recv()
                conn.send(value)
        finally:
            conn.close()

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(echo)
            conn = connect(ticket)
            for size in (1, 32, 128):
                data = payload(size)
                conn.send(data)
                assert conn.poll(5)
                received = conn.recv()
                for name in data:
                    np.testing.assert_array_equal(received[name], data[name])
            assert conn.sent_bytes > 0 and conn.received_bytes > 0
            conn.close()
            future.result(timeout=10)
    finally:
        listener.close()


def test_wrong_run_key_is_rejected(certificate):
    cert, key = certificate
    listener = TLSListener(("127.0.0.1", 0), authkey=b"correct", certfile=cert, keyfile=key)
    ticket = dict(host="127.0.0.1", port=listener.address[1], key=b"wrong".hex(), tls_ca=cert.read_text())
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(listener.accept)
            with pytest.raises(AuthenticationError):
                connect(ticket)
            with pytest.raises(AuthenticationError):
                future.result(timeout=10)
    finally:
        listener.close()


def test_external_cleartext_connection_is_rejected_before_dialing():
    with pytest.raises(ValueError, match="require TLS"):
        connect(dict(host="example.com", port=123, key="aa"))


def test_wire_uses_one_current_frame_not_five_frame_history():
    data = payload(32)
    frame = data.pop("frame")
    data.pop("reward")
    data.pop("done")
    data["proprio"] = np.tile(frame, (1, 5))
    packed = _pack(data, 2)
    assert packed["frame"].shape == (32, 53)
    assert "proprio" not in packed
    np.testing.assert_array_equal(packed["frame"], data["proprio"][:, -53:])
