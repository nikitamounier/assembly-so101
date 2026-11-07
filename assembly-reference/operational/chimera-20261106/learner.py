"""Record the allocated learner identity and safely request its checkpoint exit."""
import json
import os
from pathlib import Path
import signal
import socket
import sys

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
RUN=ROOT/'runs/chimera-v13-8192-20261106'
identity=RUN/'learner-process.json'
assert os.environ.get('SLURM_JOB_ID')=='3598717','Learner must use retained GPU allocation'
if sys.argv[1:]==['--stop']:
    row=json.loads(identity.read_text())
    assert row['host']==socket.gethostname(),'Refuse stale learner from another host'
    path=Path('/proc')/str(row['pid'])/'cmdline'
    if path.exists():
        command=path.read_bytes().decode().split('\0')
        assert 'assembly_policy.train' in command and str(RUN) in command,'Refuse unrelated process'
        os.kill(row['pid'],signal.SIGTERM)
else:
    identity.write_text(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),job=os.environ['SLURM_JOB_ID']))+'\n')
    os.execv(sys.argv[1],sys.argv[1:])
