"""Persistent authenticated TLS connections for CPU workers across providers."""

from __future__ import annotations

import pickle
import select
import socket
import ssl
import struct
from multiprocessing.connection import answer_challenge, deliver_challenge

MAX_MESSAGE = 256 * 1024 * 1024


class TLSConnection:
    """The small Connection interface used by the vector environment."""

    def __init__(self, sock):
        self.socket = sock
        self.sent_bytes = 0
        self.received_bytes = 0

    def fileno(self):
        return self.socket.fileno()

    def _read(self, size):
        result = bytearray(size)
        view = memoryview(result)
        while view:
            count = self.socket.recv_into(view)
            if not count:
                raise EOFError("Worker stream closed")
            view = view[count:]
        self.received_bytes += size
        return result

    def send_bytes(self, data):
        if len(data) > MAX_MESSAGE:
            raise ValueError("Worker message exceeds limit")
        self.socket.sendall(struct.pack("!Q", len(data)) + data)
        self.sent_bytes += len(data) + 8

    def recv_bytes(self, maxlength=None):
        size = struct.unpack("!Q", self._read(8))[0]
        if size > min(MAX_MESSAGE, maxlength or MAX_MESSAGE):
            raise OSError("Worker message exceeds limit")
        return bytes(self._read(size))

    def send(self, obj):
        self.send_bytes(pickle.dumps(obj, protocol=5))

    def recv(self):
        return pickle.loads(self.recv_bytes())

    def poll(self, timeout=0):
        return bool(self.socket.pending() or select.select([self.socket], [], [], timeout)[0])

    def close(self):
        self.socket.close()


class TLSListener:
    """Pinned-certificate TLS, followed by a per-run mutual HMAC challenge."""

    def __init__(self, address, *, authkey, certfile, keyfile, backlog=64):
        self.authkey = authkey
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.context.load_cert_chain(certfile, keyfile)
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(address)
        self.socket.listen(backlog)
        self.address = self.socket.getsockname()
        # Compatibility with the inherited listener's readiness interface.
        self._listener = self
        self._socket = self.socket

    def accept(self):
        raw, _ = self.socket.accept()
        raw.settimeout(10)
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = None
        try:
            conn = TLSConnection(self.context.wrap_socket(raw, server_side=True))
            deliver_challenge(conn, self.authkey)
            answer_challenge(conn, self.authkey)
            conn.socket.settimeout(None)
            return conn
        except BaseException:
            (conn.close if conn else raw.close)()
            raise

    def close(self):
        self.socket.close()


def connect(ticket):
    """Connect using a private ticket; never disable certificate verification."""
    key = bytes.fromhex(ticket["key"])
    if not ticket.get("tls_ca"):
        from multiprocessing.connection import Client

        host = ticket["host"]
        if host not in ("127.0.0.1", "localhost", "::1") and not ticket.get("trusted_network"):
            raise ValueError("External workers require TLS or an explicit trusted-network ticket")
        return Client((host, ticket["port"]), authkey=key)
    context = ssl.create_default_context(cadata=ticket["tls_ca"])
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    raw = socket.create_connection((ticket["host"], ticket["port"]), timeout=15)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn = None
    try:
        conn = TLSConnection(context.wrap_socket(raw, server_hostname="assembly-so101"))
        answer_challenge(conn, key)
        deliver_challenge(conn, key)
        conn.socket.settimeout(None)
        return conn
    except BaseException:
        (conn.close if conn else raw.close)()
        raise
