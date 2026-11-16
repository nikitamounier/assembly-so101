from common import *
import subprocess
import numpy as np

subprocess.run(['cc','-shared','-fPIC','-O2',str(OUT/'clock.c'),'-o',str(OUT/'clock.so')],check=True)
subprocess.run([sys.executable,str(OUT/'probe.py'),'instrumented','--native-timers'],check=True)
a=np.load(OUT/'baseline-parity.npz');b=np.load(OUT/'instrumented-timed-parity.npz')
assert a.files==b.files
checks={k:a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a.files};assert all(checks.values())
manifest=json.loads((ROOT/'active-inputs-v9.json').read_text());assert all(sha(ROOT/k)==v for k,v in manifest.items())
write(OUT/'native-phase-validation.json',dict(status='PASS',byte_identical_arrays=len(checks),source_manifest_verified_after=True,clock_source_sha256=sha(OUT/'clock.c'),clock_library_sha256=sha(OUT/'clock.so')))
