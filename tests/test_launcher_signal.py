"""A scheduler TERM must reach the owned trainer, allowing it to save and exit."""

import os
import signal
import subprocess
import sys
import time


def test_term_is_forwarded_to_trainer(tmp_path):
    child = tmp_path / "child.py"
    ready = tmp_path / "ready"
    saved = tmp_path / "saved"
    child.write_text(
        'import signal,time,sys\nfrom pathlib import Path\ndef stop(*_):\n Path(sys.argv[2]).write_text("saved")\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nPath(sys.argv[1]).write_text("ready")\nwhile True:time.sleep(.05)\n'
    )
    runner = 'import os,sys\nfrom scripts.cloud.run import run_trainer\nwith open(sys.argv[1],"w") as log:raise SystemExit(run_trainer(sys.argv[2:],os.environ.copy(),log))\n'
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            runner,
            str(tmp_path / "log"),
            sys.executable,
            str(child),
            str(ready),
            str(saved),
        ],
        start_new_session=True,
    )
    try:
        end = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < end:
            if proc.poll() is not None:
                raise AssertionError("Launcher exited before the child was ready")
            time.sleep(0.02)
        assert ready.exists()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
        assert saved.read_text() == "saved"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
