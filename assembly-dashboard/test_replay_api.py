import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer
import numpy as np

from server import Dashboard


class ReplayAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_on_demand_download_is_cached_and_scene_bound_to_run(self):
        dash=Dashboard.__new__(Dashboard)
        identity=dict(run='/fixture/run')
        meta=dict(id='complete-failure',task='cover',stage='01_pickup',identity=identity,success=False,
                  complete_episode=True,source='actual_training_episode',duration_s=.08,episode_return=-.1)
        stream=io.BytesIO();np.savez(stream,frames=np.array([[[0,0,0,1,0,0,0]],[[1,0,0,1,0,0,0]]],np.float32),times=np.array([0,.08],np.float32));raw=stream.getvalue()
        payload=dict(metadata=meta,body_names=['part'],data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
        dash.live=dict(run=identity['run'],replays={'cover/01_pickup':meta})
        dash.replay_cache=OrderedDict();dash.replay_scene=None;dash.replay_lock=asyncio.Lock();calls=[]
        async def read(expression):
            calls.append(expression)
            if expression.startswith('remote_scene'):
                return dict(identity=identity,scene=dict(geoms=[],meshes={}),layout=[dict(body=0,pos=[0,0,0],quat=[1,0,0,0])])
            return payload
        dash.replay_read=read
        app=web.Application();app.router.add_get('/api/episode',dash.episode_handler);app.router.add_get('/api/replay-scene',dash.replay_scene_handler)
        async with TestClient(TestServer(app)) as client:
            response=await client.get('/api/episode?task=cover&stage=01_pickup');self.assertEqual(response.status,200)
            clip=await response.json();self.assertFalse(clip['success']);self.assertEqual(len(clip['frames']),2)
            response=await client.get('/api/episode?task=cover&stage=01_pickup');self.assertEqual(response.status,200)
            self.assertEqual(len(calls),2,'Repeated playback must not repeat SSH downloads')
            response=await client.get('/api/replay-scene?run=wrong');self.assertEqual(response.status,409)
            response=await client.get('/api/episode?task=unknown&stage=01_pickup');self.assertIsNone(await response.json())


if __name__=='__main__':unittest.main()
