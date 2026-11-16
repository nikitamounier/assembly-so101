"""Actual production worker protocol, with read-only QA counters/RSS receipt."""
from pathlib import Path
import sys,json,resource,time
from assembly_policy.rl import remote
RUN=Path(sys.argv[1]);original=remote._serve;receipt=dict(steps=0)
class Conn:
 def __init__(self,c):self.c=c
 def __getattr__(self,n):return getattr(self.c,n)
 def recv(self):
  v=self.c.recv()
  if isinstance(v,dict) and 'state' in v:receipt.update(envs=len(v['state']),model_signature=v.get('model_signature'))
  return v
 def send(self,v):
  if isinstance(v,dict) and 'host' in v:receipt.update(threads=v['nthread'],protocol=v['proto'])
  if isinstance(v,dict) and 'reward' in v:receipt['steps']+=1
  return self.c.send(v)
def serve(c,nthread):
 try:return original(Conn(c),nthread)
 finally:
  receipt.update(maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,time=time.time());(RUN/'worker-rss.json').write_text(json.dumps(receipt,indent=2)+'\n')
remote._serve=serve
remote.worker_main(RUN,10,wait_s=300,poll_s=.2)
