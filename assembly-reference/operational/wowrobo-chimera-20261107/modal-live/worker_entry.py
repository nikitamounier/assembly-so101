"""Unmodified rollout protocol with finite startup/first-step cloud receipts."""
import json,sys,time
import numpy as np
from assembly_policy.rl import remote,transport

original_connect=transport.socket.create_connection
transport.socket.create_connection=lambda address,timeout=15:original_connect(address,timeout=90)
serve=remote._serve
class AuditConnection:
 def __init__(self,connection):self.connection=connection;self.first=True
 def __getattr__(self,name):return getattr(self.connection,name)
 def send(self,value):
  if isinstance(value,dict) and 'host' in value:
   value={**value,'host':'modal:'+config['worker_group']}
   print(json.dumps(dict(event='modal_worker_authenticated',host=value['host'],threads=value['nthread'])),flush=True)
  if isinstance(value,dict) and 'frame' in value:
   assert np.isfinite(value['frame']).all()
   print(json.dumps(dict(event='modal_worker_initialized',envs=len(value['frame']),finite=True)),flush=True)
  if self.first and isinstance(value,dict) and 'reward' in value:
   assert np.isfinite(value['reward']).all() and np.isfinite(value['obs']['frame']).all()
   print(json.dumps(dict(event='modal_first_control_step',envs=len(value['reward']),compute_seconds=value['ct'],finite=True)),flush=True);self.first=False
  return self.connection.send(value)
remote._serve=lambda conn,threads:serve(AuditConnection(conn),threads)
config=json.loads(sys.stdin.read())
remote.worker_main(None,config['threads'],wait_s=config['seconds'],poll_s=2,ticket=config['ticket'])
