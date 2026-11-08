import socket,threading,time
from multiprocessing.connection import Client
from learner_bootstrap import BoundedListener

def test_stalled_handshake_is_bounded():
    listener=BoundedListener(('127.0.0.1',0),authkey=b'test');listener.handshake_timeout=.15
    peer=socket.create_connection(listener.address);start=time.monotonic()
    try:
        try:listener.accept()
        except (EOFError,OSError):pass
        else:raise AssertionError('Unauthenticated peer admitted')
        assert time.monotonic()-start<1
    finally:peer.close();listener.close()

def test_mutual_authentication_and_messages_preserved():
    listener=BoundedListener(('127.0.0.1',0),authkey=b'test');got=[]
    def client():
        with Client(listener.address,authkey=b'test') as c:c.send({'payload':[1,2,3]});got.append(c.recv())
    thread=threading.Thread(target=client);thread.start()
    with listener.accept() as c:
        assert c.recv()=={'payload':[1,2,3]};c.send('ok')
    thread.join(2);assert got==['ok'];listener.close()

if __name__=='__main__':
    test_stalled_handshake_is_bounded();test_mutual_authentication_and_messages_preserved();print('PASS stalled deadline and legitimate mutual-auth roundtrip')
