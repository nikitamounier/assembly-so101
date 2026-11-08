"""Outbound-only Chimera bridge. Public legs use pinned TLS; learner HMAC stays intact."""
import asyncio
import contextlib
import json
from pathlib import Path
import ssl
import time

async def pipe(reader, writer):
    try:
        while data := await reader.read(262144):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()

async def splice(ar, aw, br, bw):
    tasks=[asyncio.create_task(pipe(ar,bw)),asyncio.create_task(pipe(br,aw))]
    try:
        await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        aw.close();bw.close()

async def framed_pipe(reader,writer,source_tls):
    import struct
    try:
        while True:
            if source_tls:
                size=struct.unpack('!Q',await reader.readexactly(8))[0]
            else:
                size=struct.unpack('!i',await reader.readexactly(4))[0]
                if size==-1:size=struct.unpack('!Q',await reader.readexactly(8))[0]
            if not 0<=size<=256*1024*1024:raise ValueError('Invalid frame size')
            writer.write(struct.pack('!i' if source_tls else '!Q',size))
            left=size
            while left:
                data=await reader.readexactly(min(left,262144));left-=len(data)
                writer.write(data);await writer.drain()
    finally:writer.close()

async def framed_splice(ar,aw,br,bw):
    tasks=[asyncio.create_task(framed_pipe(ar,bw,True)),asyncio.create_task(framed_pipe(br,aw,False))]
    try:await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);aw.close();bw.close()

async def broker_servers(cert,key,token,ready,seconds,worker_port=8741,reverse_port=8742):
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version=ssl.TLSVersion.TLSv1_2;ctx.load_cert_chain(cert,key)
    peers=asyncio.Queue(maxsize=512)
    async def reverse(r,w):
        try:
            supplied=await asyncio.wait_for(r.readline(),10)
            import hmac
            if not hmac.compare_digest(supplied.rstrip(),token.encode()):return
            print('gateway: reverse authenticated',flush=True)
            done=asyncio.Event()
            await asyncio.wait_for(peers.put((r,w,done)),10)
            await done.wait()
        finally:w.close()
    async def worker(r,w):
        peer=None
        try:
            print('gateway: worker TLS accepted',flush=True)
            while peer is None:
                candidate=await asyncio.wait_for(peers.get(),60)
                if candidate[0].at_eof() or candidate[1].is_closing():candidate[2].set()
                else:peer=candidate
            br,bw,done=peer
            bw.write(b'R');await bw.drain()
            print('gateway: paired worker and reverse stream',flush=True)
            await splice(r,w,br,bw)
        except (TimeoutError,ConnectionError,ssl.SSLError):pass
        finally:
            w.close()
            if peer:peer[2].set()
    a=await asyncio.start_server(worker,'0.0.0.0',worker_port,ssl=ctx,backlog=512)
    b=await asyncio.start_server(reverse,'0.0.0.0',reverse_port,ssl=ctx,backlog=512)
    async with a,b:
        ready()
        await asyncio.sleep(seconds)

async def agent(ticket,target,connections,seconds):
    ctx=ssl.create_default_context(cadata=ticket['tls_ca'])
    ctx.minimum_version=ssl.TLSVersion.TLSv1_2
    deadline=time.monotonic()+seconds
    async def slot(index):
        while time.monotonic()<deadline:
            w=None
            try:
                r,w=await asyncio.wait_for(asyncio.open_connection(ticket['host'],ticket['port'],ssl=ctx,server_hostname='assembly-so101'),20)
                print(f'agent slot {index}: reverse TLS connected',flush=True)
                w.write(ticket['token'].encode()+b'\n');await w.drain()
                assert await r.readexactly(1)==b'R'
                current=json.loads(Path(target['ticket_path']).read_text()) if 'ticket_path' in target else target
                lr,lw=await asyncio.open_connection(current['host'],current['port'])
                print(f'agent slot {index}: learner TCP connected',flush=True)
                if target.get('framing')=='multiprocessing':await framed_splice(r,w,lr,lw)
                else:await splice(r,w,lr,lw)
            except (OSError,EOFError,TimeoutError,asyncio.IncompleteReadError) as exc:
                print(f'agent slot {index}: {type(exc).__name__}',flush=True)
                await asyncio.sleep(2)
            finally:
                if w:w.close()
    tasks=[asyncio.create_task(slot(i)) for i in range(connections)]
    try:await asyncio.sleep(seconds)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)

if __name__=='__main__':
    import sys
    config=json.loads(sys.stdin.read())
    asyncio.run(agent(config['reverse'],config['target'],config['connections'],config['seconds']))
