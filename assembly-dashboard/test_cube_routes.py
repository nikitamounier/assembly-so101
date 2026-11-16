"""Public cube routes keep APIs scoped away from the assembly dashboard."""
import unittest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from server import Dashboard, scoped_text


class CubeScopingTests(unittest.TestCase):
    def test_cube_html_and_fetch_bootstrap_use_cube_scope_once(self):
        raw=b'<html><head></head><body><script src="/assets/app.js"></script><a href="/cube/">Cube</a></body></html>'
        text=scoped_text(raw,'text/html','/cube').decode()
        self.assertIn('src="/cube/assets/app.js"',text)
        self.assertIn('href="/cube/"',text)
        self.assertNotIn('/cube/cube/',text)
        self.assertNotIn('/states/',text)
        self.assertIn("u.pathname='/cube'+u.pathname",text)

    def test_cube_javascript_and_existing_states_have_separate_apis(self):
        raw=b'fetch("/api/live");let a="/assets/model.bin";'
        self.assertIn(b'/cube/api/live',scoped_text(raw,'application/javascript','/cube'))
        self.assertIn(b'/states/api/live',scoped_text(raw,'application/javascript'))


class CubePrivateRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_paths_and_mutations_never_reach_upstream(self):
        dash=Dashboard.__new__(Dashboard)
        app=web.Application();app.router.add_route('*','/cube/{tail:.*}',dash.cube_proxy)
        async with TestClient(TestServer(app)) as client:
            for path in ('server.json','cluster.json','api/config','api/studio','api/studio/stop'):
                response=await client.get('/cube/'+path)
                self.assertEqual(response.status,404)
            response=await client.post('/cube/api/live')
            self.assertEqual(response.status,405)


if __name__=='__main__':unittest.main()
