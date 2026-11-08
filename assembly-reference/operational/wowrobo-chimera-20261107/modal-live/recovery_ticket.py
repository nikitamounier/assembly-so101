"""Retry coordination only: no physics, policy, RNG, authentication changes."""
import json,time
from pathlib import Path

class RenewableTicket(dict):
    def __init__(self, initial, *, clock=time.monotonic, retry_period=30.):
        super().__init__(initial)
        self.clock,self.retry_period=clock,retry_period
    def __getitem__(self,key):
        if key=='time':
            # worker_main stops retrying a failed timestamp after300s. This
            # private coordination generation makes that wait temporary even
            # when the endpoint/key remain the same across learner failover.
            return (dict.__getitem__(self,key),int(self.clock()//self.retry_period))
        return dict.__getitem__(self,key)

def read_ticket(path):
    # One immutable snapshot per connection avoids mixing a new certificate
    # with the old endpoint if a broker is replaced concurrently.
    return json.loads(Path(path).read_text())

def atomic_ticket(path,value):
    import os
    path=Path(path);tmp=path.with_suffix('.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(value,stream)
    os.replace(tmp,path)

def rotate_completed(calls,spawn,*,timeouts=(TimeoutError,)):
    """Replace only a completed technical lease, preserving all healthy calls."""
    rotations=0
    for row in calls:
        try:outcome=row['call'].get(timeout=0)
        except timeouts:continue
        if not isinstance(outcome,dict) or outcome.get('status')!='lease_complete':
            raise RuntimeError('Unexpected worker termination: '+str(outcome))
        row['call']=spawn(row['role']);rotations+=1
    return rotations
