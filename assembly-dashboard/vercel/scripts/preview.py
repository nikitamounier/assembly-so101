"""Loopback-only verification server for the prepared hosted transport."""
from pathlib import Path
import json
from aiohttp import web, ClientSession

ROOT = Path(__file__).resolve().parents[1] / "public"

@web.middleware
async def isolation(request, handler):
    response = await handler(request)
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response

async def assets(request):
    return web.json_response({'states/'+p.name:'http://127.0.0.1:8898/data/states/'+p.name for p in (ROOT/'data/states').iterdir() if p.suffix in ('.gz','.pack')})

async def api(request):
    async with ClientSession() as client:
        async with client.get('http://127.0.0.1:8881'+str(request.rel_url)) as response:
            return web.Response(body=await response.read(), status=response.status, content_type='application/json')

async def static(request):
    route = request.match_info['path'] or 'index.html'
    file = (ROOT/route).resolve()
    if file.is_dir(): file /= 'index.html'
    if not file.is_relative_to(ROOT) or not file.is_file():raise web.HTTPNotFound()
    return web.FileResponse(file)

app = web.Application(middlewares=[isolation])
app.router.add_get('/hosted-assets.json',assets)
app.router.add_get('/api/{path:.*}',api)
app.router.add_get('/{path:.*}',static)
web.run_app(app,host='127.0.0.1',port=8898)
