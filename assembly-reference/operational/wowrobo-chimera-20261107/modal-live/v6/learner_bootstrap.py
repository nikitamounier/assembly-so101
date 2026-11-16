"""Bound learner connection admission, then install the pinned passive replay observer.

No changes to worker messages, physics, observations, rewards, sampler or PPO.
"""
from pathlib import Path
import inspect,os,runpy,socket,textwrap,threading,time
from multiprocessing import connection

class BoundedListener(connection.Listener):
    handshake_timeout=10.
    def accept(self):
        c=self._listener.accept()
        probe=socket.socket(fileno=os.dup(c.fileno()))
        def abort():
            try:probe.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        timer=threading.Timer(self.handshake_timeout,abort);timer.daemon=True;timer.start()
        try:
            if self._authkey is not None:
                connection.deliver_challenge(c,self._authkey)
                connection.answer_challenge(c,self._authkey)
            return c
        except BaseException:
            c.close();raise
        finally:timer.cancel();probe.close()

def install_admission(remote):
    original=remote.RemoteVecEnv.__init__
    source=textwrap.dedent(inspect.getsource(original))
    needle='while len(conns) < n_conns:\n'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'        if time.time() - start >= accept_timeout and len(conns) >= max(1, int(min_frac * n_conns)):\n            break\n',1)
    ns={};exec(compile(source,'bounded_remote_admission','exec'),remote.__dict__,ns)
    remote.Listener=BoundedListener;remote.RemoteVecEnv.__init__=ns['__init__']
    print('[remote] bounded authentication10s and initial admission deadline enabled',flush=True)

if __name__=='__main__':
    import assembly_policy.rl.remote as remote
    install_admission(remote)
    runpy.run_path(str(Path(__file__).resolve().parent.parent/'replay-observer/record_training.py'),run_name='__main__')
