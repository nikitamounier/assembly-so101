"""Exercise TLS verification, reverse pairing, and bidirectional framing with real sockets."""
import asyncio,ssl,subprocess,tempfile
from pathlib import Path
from gateway import agent,broker_servers

async def test():
 with tempfile.TemporaryDirectory() as td:
  cert=Path(td)/'cert';key=Path(td)/'key'
  subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=assembly-so101','-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
  async def echo(r,w):
   try:
    while data:=await r.read(65536):w.write(data);await w.drain()
   finally:w.close()
  target=await asyncio.start_server(echo,'127.0.0.1',0)
  host,port=target.sockets[0].getsockname()
  ready=asyncio.Event()
  # Reserve dynamic local ports for this isolated test.
  import socket
  with socket.socket() as a,socket.socket() as b:
   a.bind(('127.0.0.1',0));b.bind(('127.0.0.1',0));wp=a.getsockname()[1];rp=b.getsockname()[1]
  broker=asyncio.create_task(broker_servers(cert,key,'test-token',ready.set,15,wp,rp));await ready.wait()
  bridge=asyncio.create_task(agent(dict(host='127.0.0.1',port=rp,tls_ca=cert.read_text(),token='test-token'),dict(host=host,port=port),2,12))
  ctx=ssl.create_default_context(cadata=cert.read_text())
  try:
   # Untrusted certificates must be refused before any private protocol is sent.
   try:await asyncio.open_connection('127.0.0.1',wp,ssl=ssl.create_default_context(),server_hostname='assembly-so101')
   except ssl.SSLCertVerificationError:pass
   else:raise AssertionError('untrusted certificate accepted')
   async def check(marker):
    r,w=await asyncio.open_connection('127.0.0.1',wp,ssl=ctx,server_hostname='assembly-so101')
    data=bytes([marker])*1048576
    w.write(data);await w.drain()
    assert await asyncio.wait_for(r.readexactly(len(data)),8)==data
    w.close();await w.wait_closed()
   await asyncio.gather(check(17),check(83))
   print('PASS: two independent reverse TLS streams, 2 MiB exact roundtrip, untrusted certificate rejected')
   from multiprocessing.connection import Listener
   import sys
   sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'assembly-so101'))
   from assembly_policy.rl.transport import connect
   inner=Listener(('127.0.0.1',0),authkey=b'a'*32)
   def native():
    c=inner.accept()
    try:
     item=c.recv();assert item=={'payload':bytes(range(256))*8192};c.send({'echo':item})
    finally:c.close();inner.close()
   native_task=asyncio.create_task(asyncio.to_thread(native))
   # Retire the raw echo peers so the next stream is paired to the framed agent.
   bridge.cancel();await asyncio.gather(bridge,return_exceptions=True)
   bridge=asyncio.create_task(agent(dict(host='127.0.0.1',port=rp,tls_ca=cert.read_text(),token='test-token'),dict(host=inner.address[0],port=inner.address[1],framing='multiprocessing'),1,12))
   def client():
    c=connect(dict(host='127.0.0.1',port=wp,key=(b'a'*32).hex(),tls_ca=cert.read_text()))
    try:
     item={'payload':bytes(range(256))*8192};c.send(item);assert c.recv()=={'echo':item}
    finally:c.close()
   await asyncio.wait_for(asyncio.gather(asyncio.to_thread(client),native_task),10)
   print('PASS: real mutual HMAC and pickle messages, TLS 8-byte/native 4-byte framing bridge')
  finally:
   bridge.cancel();broker.cancel();await asyncio.gather(bridge,broker,return_exceptions=True);target.close();await target.wait_closed()

if __name__=='__main__':asyncio.run(test())
