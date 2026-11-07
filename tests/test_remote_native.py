"""Native remote steps must match local steps, including resets and history."""

import json
import threading
import time

import numpy as np

from assembly_policy.rl.remote import RemoteVecEnv, _serve
from assembly_policy.rl.transport import connect
from assembly_policy.rl.vecenv import LocalVecEnv
from assembly_policy.so101.env import EnvConfig
from assembly_policy.so101.resets import reset, signature
from assembly_policy.so101.scene import model


def test_remote_native_rollouts_match_local_resets_and_histories(tmp_path):
    m = model(contact_monitor=True)
    state = reset(m, "motor", 17)

    def provider(n, *_):
        return np.zeros(n, np.int64), np.repeat(state[None], n, axis=0), np.zeros(n, np.int64)

    errors = []

    def worker():
        try:
            path = tmp_path / "server.json"
            deadline = time.monotonic() + 10
            while not path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Learner did not publish its ticket")
                time.sleep(0.01)
            conn = connect(json.loads(path.read_text()))
            try:
                _serve(conn, 1)
            finally:
                conn.close()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    cfg = EnvConfig(nthread=1, horizon_s=0.08, max_contact_depth=0.0005)
    remote = RemoteVecEnv(
        tmp_path, ["motor"], 2, cfg, provider, 1, accept_timeout=0, init_wait=1, model_signature=signature()
    )
    local = LocalVecEnv(["motor"], 2, cfg, provider)
    try:
        for _ in range(4):
            for name, values in local.observe().items():
                np.testing.assert_allclose(remote.observe()[name], values, atol=1e-6)
            action = np.zeros((2, 12), np.float32)
            actual, expected = remote.step(action), local.step(action)
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
            np.testing.assert_array_equal(actual[1], expected[1])
            for key in ("valid", "success", "safety", "timeout", "ep_len"):
                np.testing.assert_array_equal(actual[2][key], expected[2][key])
        assert (tmp_path / "server.json").stat().st_mode & 0o077 == 0
    finally:
        remote.close()
        local.close()
    thread.join(5)
    assert not thread.is_alive()
    assert not errors
