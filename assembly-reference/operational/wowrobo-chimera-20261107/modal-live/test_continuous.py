import ast,json,os,tempfile,types,unittest
from pathlib import Path
from recovery_ticket import RenewableTicket,atomic_ticket,read_ticket,rotate_completed

class Clock:
    def __init__(self):self.now=0.
    def time(self):return self.now
    def sleep(self,seconds):self.now+=seconds

class Recovered(Exception):pass

def simulate(ticket_factory):
    # Execute the real inherited retry loop, rather than a reimplementation.
    source=Path(__file__).resolve().parents[4]/'assembly-so101/assembly_policy/rl/remote.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker_main')
    clock=Clock();calls=[]
    class Conn:
        def close(self):pass
    def connect(ticket):
        calls.append(clock.now)
        if clock.now<720:raise OSError('simulated12minute learner outage')
        assert ticket['key']=='same-auth-key';return Conn()
    def serve(conn,threads):raise Recovered()
    ns=dict(Path=Path,time=clock,np=types.SimpleNamespace(random=types.SimpleNamespace(random=lambda:0.)),print=lambda *a,**k:None,connect=connect,_nodelay=lambda x:x,_serve=serve,_NoSlice=type('NoSlice',(Exception,),{}))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'real_worker_main','exec'),ns)
    recovered=False
    try:ns['worker_main'](None,10,wait_s=1000,poll_s=2,ticket=ticket_factory(clock))
    except Recovered:recovered=True
    return recovered,calls

class RecoveryTests(unittest.TestCase):
    def test_lease_rollover_preserves_healthy_workers_without_duplicate_spawns(self):
        class Call:
            def __init__(self,done=False):self.done=done
            def get(self,timeout):
                if not self.done:raise TimeoutError()
                return dict(status='lease_complete')
        healthy=Call();expired=Call(True);calls=[dict(role='worker',call=healthy),dict(role='worker',call=expired)]
        spawned=[]
        def spawn(role):
            call=Call();spawned.append(call);return call
        self.assertEqual(rotate_completed(calls,spawn),1)
        self.assertIs(calls[0]['call'],healthy);self.assertIs(calls[1]['call'],spawned[0])
        self.assertEqual(rotate_completed(calls,spawn),0);self.assertEqual(len(spawned),1)
        spawned[0].done=True
        self.assertEqual(rotate_completed(calls,spawn),1);self.assertEqual(len(spawned),2)
    def test_unexpected_exit_is_not_silently_replaced(self):
        call=types.SimpleNamespace(get=lambda timeout:dict(status='nonfinite_failure'))
        with self.assertRaises(RuntimeError):rotate_completed([dict(role='worker',call=call)],lambda role:self.fail('must not resubmit bad physics'))
    def test_static_ticket_reproduces_confirmed_failure(self):
        recovered,calls=simulate(lambda clock:dict(time=42.,key='same-auth-key'))
        self.assertFalse(recovered);self.assertLess(max(calls),720)
    def test_same_endpoint_and_key_recover_after12minute_outage(self):
        recovered,calls=simulate(lambda clock:RenewableTicket(dict(time=42.,key='same-auth-key'),clock=clock.time))
        self.assertTrue(recovered);self.assertGreaterEqual(max(calls),720);self.assertLess(max(calls),750)
    def test_broker_epoch_refresh_is_private_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'ticket.json';a=dict(host='a',port=1,tls_ca='cert-a',key='unchanged');b=dict(host='b',port=2,tls_ca='cert-b',key='unchanged')
            atomic_ticket(p,a);snapshot=read_ticket(p);atomic_ticket(p,b)
            self.assertEqual(snapshot,a);self.assertEqual(read_ticket(p),b);self.assertEqual(p.stat().st_mode&0o777,0o600)

if __name__=='__main__':unittest.main()
