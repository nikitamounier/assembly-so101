import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from replay_io import reconstruct,remote_clip


class ReplayIOTests(unittest.TestCase):
    def payload(self):
        stream=io.BytesIO()
        q=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
        frames=np.array([[[1,2,3,*q]],[[2,3,4,*q]]],np.float32)
        np.savez(stream,frames=frames,times=np.array([0,.08],np.float32))
        raw=stream.getvalue()
        return dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),body_names=['part'],metadata=dict(id='actual'))

    def test_reconstruct_body_rotation_and_static_world_geom(self):
        result=reconstruct(self.payload(),[dict(body=0,pos=[1,0,0],quat=[1,0,0,0]),dict(body=-1,pos=[0,0,-.1],quat=[1,0,0,0])])
        np.testing.assert_allclose(result['frames'][0][0][:3],[1,3,3],atol=1e-6)
        np.testing.assert_allclose(result['frames'][1][1],[0,0,-.1,1,0,0,0],atol=1e-6)
        bad=self.payload();bad['sha256']='wrong'
        with self.assertRaises(AssertionError):reconstruct(bad,[])

    def test_remote_selection_always_prefers_complete_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'training-replays';root.mkdir()
            identity=dict(run=folder)
            (root/'manifest.json').write_text(json.dumps(dict(identity=identity,body_names=['part'])))
            (root/'success.npz').write_bytes(b'bounded fixture')
            clip=dict(id='success',file='success.npz',complete_episode=True,source='actual_training_episode',identity=identity,data_sha256=hashlib.sha256(b'bounded fixture').hexdigest())
            (root/'index.json').write_text(json.dumps(dict(clips={'task/start':dict(success=clip,failure=dict(id='failure'))})))
            self.assertEqual(remote_clip(folder,'task/start','success')['metadata']['id'],'success')
            with self.assertRaises(AssertionError):remote_clip(folder,'task/start','failure')
            (root/'success.npz').write_bytes(b'new episode before index commit')
            with self.assertRaises(AssertionError):remote_clip(folder,'task/start','success')


if __name__=='__main__':unittest.main()
