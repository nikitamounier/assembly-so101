"""Loopback assembly dashboard; polling has no scheduler/training mutations."""
import asyncio
import gzip
import hashlib
import json
import os
import re
import shlex
import time
from pathlib import Path
import sys
from collections import OrderedDict

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
from urllib.parse import unquote, urlsplit
from yarl import URL

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OPS = WORKSPACE / 'robotics-assembly-3.0/operational/wowrobo-chimera-20261107'
sys.path.insert(0, str(WORKSPACE/'assembly-so101'))
sys.path.insert(0, str(OPS/'modal-live'))


VIEWER_TARGET = HERE / 'state-viewer-target.json'
REPLAY_MODEL = Path('robotics-assembly-3.0/outputs/equilibrium-model-20261107/profile-centered-v6/source.xml')
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length', 'content-encoding'}
URL_PREFIXES = r'(?:api/|assets/|mujoco(?:/|[\"\x27`])|reference/|collision-boxes\.json|ui(?:/|[\"\x27`])|state(?:[/?]|[\"\x27`])|model(?:[/?]|[\"\x27`])|drop(?:[/?]|[\"\x27`]))'


def viewer_target(path=VIEWER_TARGET):
    value = json.loads(path.read_text()) if path.exists() else {'url': 'http://127.0.0.1:8878'}
    url = urlsplit(value.get('url', ''))
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
            or url.query or url.fragment or url.path not in ('', '/')
            or url.port is None or not 1024 <= url.port <= 65535 or url.port == 8881):
        raise ValueError('States target must be a fixed non-dashboard HTTP loopback port')
    return value, f'http://127.0.0.1:{url.port}'


def scoped_text(body, content_type, scope="/states"):
    """Scope original viewer assets/APIs without sharing dashboard API names."""
    text = body.decode('utf-8')
    text = re.sub(r'([\"\x27`])/(?=' + URL_PREFIXES + ')', lambda match: match.group(1) + scope + '/', text)
    if 'text/html' in content_type:
        text = re.sub(r'((?:src|href|action)\s*=\s*[\"\x27])/(?!/|' + re.escape(scope.lstrip('/')) + r'/)', lambda match: match.group(1) + scope + '/', text, flags=re.I)
        # Root-relative network URLs computed by the official MuJoCo client
        # also need the iframe scope, including websocket and beacon requests.
        bootstrap = r"""<script>
(()=>{
 const scope=(value)=>{const u=new URL(value,location.href);if(u.host===location.host&&['http:','https:','ws:','wss:'].includes(u.protocol)&&!u.pathname.startsWith('/states/'))u.pathname='/states'+u.pathname;return u.href;};
 const fetch0=window.fetch.bind(window);window.fetch=(input,options)=>fetch0(input instanceof Request?new Request(scope(input.url),input):scope(input),options);
 const open0=XMLHttpRequest.prototype.open;XMLHttpRequest.prototype.open=function(method,url,...args){return open0.call(this,method,scope(url),...args);};
 const WS=window.WebSocket;window.WebSocket=class extends WS{constructor(url,...args){super(scope(url),...args);}};
 if(window.Worker){const W=window.Worker;window.Worker=class extends W{constructor(url,...args){super(scope(url),...args);}};}
 if(navigator.sendBeacon){const beacon=navigator.sendBeacon.bind(navigator);navigator.sendBeacon=(url,data)=>beacon(scope(url),data);}
})();
</script>"""
        text = re.sub(r'(<head\b[^>]*>)', lambda match: match.group(1) + bootstrap.replace('/states', scope), text, count=1, flags=re.I)
    return text.encode('utf-8')


@web.middleware
async def isolation(request, handler):
    response = await handler(request)
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    return response


class Dashboard:
    def __init__(self):
        self.bc_experiment = os.environ.get('ASSEMBLY_DASHBOARD_EXPERIMENT') == 'bc'
        self.cache = HERE/('live-cache-bc.json' if self.bc_experiment else 'live-cache.json')
        self.live = dict(history=[], previous_history=[], connection='connecting')
        if self.cache.exists():
            try:
                self.live = {**json.loads(self.cache.read_text()), 'connection': 'connecting'}
            except json.JSONDecodeError:
                pass
        self.scene = None
        self.previews = {}
        self.provider = {}
        self.last_sampler_read = 0
        self.stage_history = self.live.get('stage_history', [])
        self.replay_cache=OrderedDict()
        self.replay_scene=None
        self.replay_lock=asyncio.Lock()
        catalog=json.loads((WORKSPACE/'robotics-assembly-3.0/assets_local/wowrobo/initial-valid-resets-v1/stages/manifest.json').read_text())
        self.catalog=catalog.get('stage_by_task',{})

    def targets(self):
        old = json.loads((OPS/'v3/target.json').read_text())
        if self.bc_experiment:
            target = json.loads((WORKSPACE/'bc-full-ppo-20261108/target.json').read_text())
            return target['python'], [dict(root=target['remote'], run=target['run'], version='bc-sgs')]
        candidates = [dict(root=old['remote'], run='run-v3', version='v3')]
        for release in (OPS/'modal-live/release-target.json',OPS/'modal-live/target.json'):
            if release.exists():
                new = json.loads(release.read_text())
                if all(c['root']!=new['remote'] for c in candidates):
                    candidates.append(dict(root=new['remote'], run='run-v4', version='v4', discover=True))
        return old['python'], candidates

    def preview(self):
        # Visual meshes only. No physics rollout or native training callback is loaded.
        import mujoco
        import numpy as np
        from assembly_policy.viewer.scene import visual_scene, scene_poses
        bank = WORKSPACE/'robotics-assembly-3.0/assets_local/wowrobo/initial-valid-resets-v1'
        m = mujoco.MjModel.from_xml_path(str(bank/'scene.xml'))
        scene = visual_scene(m)
        stages = dict(base_side_cover='05_cover_near', shoulder_connector='10_connector_near',
                      assembly_pin='15_pin_near', upper_arm_module='23_upper_near', forearm_module='31_final_near')
        stage_names=json.loads((bank/'stages/manifest.json').read_text())['task_names']
        for stage in stage_names:
            with np.load(bank/'stages'/(stage+'_train.npz')) as z:
                d = mujoco.MjData(m)
                mujoco.mj_setState(m, d, z['states'][0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_kinematics(m, d)
                self.previews[stage] = dict(poses=scene_poses(m, d, scene), stage=stage)
        for task,stage in stages.items():self.previews[task]=self.previews[stage]
        self.scene = scene

    async def bridge(self):
        while True:
            process = None
            try:
                python, candidates = self.targets()
                sampler=time.time()-self.last_sampler_read>60
                source = 'CANDIDATES='+repr(candidates)+';INCLUDE_SAMPLER='+repr(sampler)+'\n'+(HERE/'export.py').read_text()
                remote = shlex.join(['srun', '--jobid=3077705', '--overlap', '--exact', '--ntasks=1',
                                     '--cpus-per-task=1','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1', python, '-c', source])
                process = await asyncio.create_subprocess_exec('ssh', '-o', 'BatchMode=yes',
                            '-o', 'ConnectTimeout=15', 'chimera-login', remote,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 50)
                if process.returncode:
                    raise RuntimeError(stderr.decode()[-400:] or 'Allocated read failed')
                payload = json.loads(stdout)
                stages=payload.get('starting_states',{})
                previous=self.live.get('starting_states',{}) if self.live.get('run')==payload['run'] else {}
                if stages.get('sampler_available'):
                    self.last_sampler_read=time.time()
                    key=(payload['run'],stages['sampler_identity'])
                    if not any((r['run'],r['sampler_identity'])==key for r in self.stage_history):
                        self.stage_history.append(dict(run=payload['run'],version=payload['version'],**stages))
                        self.stage_history=self.stage_history[-1200:]
                elif previous.get('sampler_available'):
                    payload['starting_states']=previous
                payload['stage_history']=self.stage_history
                payload.update(connection='live', received_at=time.time())
                self.live = payload
                self.cache.write_text(json.dumps(payload))
            except asyncio.CancelledError:
                if process and process.returncode is None:
                    process.kill(); await process.wait()
                raise
            except Exception as error:
                if process and process.returncode is None:
                    process.kill(); await process.wait()
                self.live = {**self.live, 'connection': 'unavailable', 'connection_error': str(error)}
            await asyncio.sleep(15)

    async def providers(self):
        while True:
            process = None
            try:
                sdk = Path('/tmp/assembly-modal-sdk/bin/python')
                if sdk.exists():
                    process = await asyncio.create_subprocess_exec(str(sdk), '-m', 'modal', 'app', 'list', '--json',
                              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    out, _ = await asyncio.wait_for(process.communicate(), 25)
                    if process.returncode == 0:
                        apps = [a for a in json.loads(out) if a.get('description') == 'wowrobo-live-rollouts']
                        active = [a for a in apps if a['state'] not in ('stopped', 'stopping...')]
                        self.provider = dict(observed_at=time.time(), apps=active,
                            containers=sum(int(a.get('tasks') or 0) for a in active), status='observed')
            except asyncio.CancelledError:
                if process and process.returncode is None:
                    process.kill(); await process.wait()
                raise
            except Exception:
                self.provider = {**self.provider, 'status': 'unavailable'}
            await asyncio.sleep(45)

    async def live_handler(self, request):
        payload = {**self.live, 'modal': self.provider}
        if self.bc_experiment:
            path = WORKSPACE/'bc-full-ppo-20261108/primary/active.json'
            receipt = json.loads(path.read_text()) if path.exists() else {}
            count = len(receipt.get('worker_boxes', []))
            payload.update(experiment='bc', sail=dict(status=receipt.get('status'),
                reserved_cpus=(4 if receipt.get('broker_box') else 0)+64*min(count,7)+8*max(count-7,0),
                target_cpus=500, observed_at=receipt.get('heartbeat_unix'),
                finite_workers=receipt.get('unique_first_control_workers', receipt.get('finite_first_control_first_box',0))))
        payload['starting_catalog']=self.catalog
        handoffs=[]
        for handoff in (OPS/'modal-live').glob('handoff-active*.json'):
            try:
                item=json.loads(handoff.read_text())
                if item.get('old_run')==payload.get('run'):handoffs.append(item)
            except (OSError,json.JSONDecodeError):pass
        if handoffs:payload['handoff']=max(handoffs,key=lambda item:item.get('time_unix',0))
        active = OPS/'modal-live/active.json'
        if active.exists():
            receipt = json.loads(active.read_text())
            payload['modal_request'] = {k:receipt.get(k) for k in
                ('status', 'run', 'requested_physical_cores', 'hosts', 'seconds', 'controller_heartbeat_unix', 'app_id')}
        body = await asyncio.to_thread(json.dumps, payload, separators=(',', ':'))
        response = web.Response(text=body, content_type='application/json',
                                headers={'Cache-Control': 'no-store', 'Vary': 'Accept-Encoding'},
                                zlib_executor_size=256 * 1024)
        response.enable_compression()
        return response

    async def workspace_handler(self, request):
        try:
            value, target = viewer_target()
        except (OSError, ValueError) as error:
            return web.json_response({'states': {'available': False, 'error': str(error)}, 'training': {}}, status=503)
        states = {'available': True, 'href': '/states/', 'bank_status': value.get('bank_status', 'awaiting_sync'),
                  'target_train': value.get('target_train', 16000), 'target_eval': value.get('target_eval', 3200),
                  'accepted_train': value.get('accepted_train'), 'accepted_eval': value.get('accepted_eval'),
                  'snapshot_label': value.get('snapshot_label'),
                  'updated_at': value.get('updated_at'),
                  'revision': hashlib.sha256(str(value.get('bank_id') or value.get('revision') or target).encode()).hexdigest()[:16],
                  'qualification': 'Native overlap and held-grip retention; motion allowed'}
        training = value.get('training', {})
        training = {key: training[key] for key in ('state', 'checkpoint', 'reason') if key in training}
        return web.json_response({'states': states, 'training': training}, headers={'Cache-Control': 'no-store'})

    async def cube_proxy(self, request):
        return await self.states_proxy(request, target='http://127.0.0.1:8880', scope='/cube', cube=True)

    async def bc_proxy(self, request):
        return await self.states_proxy(request, target='http://127.0.0.1:8882', scope='/bc', cube=True)

    async def cube_states_proxy(self, request):
        return await self.states_proxy(request, target='http://127.0.0.1:8877', scope='/cube-states')

    async def states_proxy(self, request, *, target=None, scope='/states', cube=False):
        # The destination is operator-owned local configuration, never a request
        # parameter. Users cannot turn this endpoint into an arbitrary proxy.
        try:
            if target is None:
                _, target = viewer_target()
        except (OSError, ValueError) as error:
            raise web.HTTPServiceUnavailable(text='States viewer configuration unavailable') from error
        tail = request.match_info.get('tail', '')
        if '\\' in tail or any(piece in ('.', '..') for piece in tail.split('/')):
            raise web.HTTPBadRequest(text='Invalid viewer path')
        decoded = unquote(unquote(tail))
        if any(piece in ('.', '..') for piece in decoded.split('/')) or '\\' in decoded:
            raise web.HTTPBadRequest(text='Invalid viewer path')
        allowed_api = {'api/catalog', 'api/refresh', 'api/scene', 'api/state', 'api/select',
                       'api/reward-tolerance', 'api/studio', 'api/studio/stop'}
        allowed_static = tail in {'', 'index.html', 'collision-boxes.json', 'ui', 'state', 'model'} or tail.startswith(('assets/', 'reference/', 'mujoco/', 'ui/'))
        if not cube and not allowed_static and tail not in allowed_api:
            raise web.HTTPNotFound(text='Unknown states viewer route')
        if cube:
            allowed_api = {'api/live', 'api/scene', 'api/initial', 'api/episode'}
            if scope == '/bc':
                allowed_api = {'api/live', 'api/scene', 'api/preview', 'api/episode', 'api/replay-scene', 'api/workspace'}
            allowed_static = tail in {'', 'index.html'} or tail.startswith('assets/')
        allowed_post = {'api/select', 'api/refresh', 'api/reward-tolerance', 'api/studio', 'api/studio/stop'}
        if not allowed_static and tail not in allowed_api:
            raise web.HTTPNotFound(text='Unknown dashboard route')
        if request.method not in ('GET', 'HEAD') and not (not cube and request.method == 'POST' and tail in allowed_post):
            raise web.HTTPMethodNotAllowed(request.method, ['GET', 'HEAD'])
        if tail.rstrip('/').endswith('/drop') or tail == 'drop':
            raise web.HTTPForbidden(text='Model uploads are disabled')
        origin = request.headers.get('Origin')
        if origin:
            external = urlsplit(origin)
            # Cloudflare terminates HTTPS while this server stays on loopback
            # HTTP. Match the public host, not the local transport scheme.
            if external.scheme not in ('http', 'https') or external.netloc.lower() != request.host.lower():
                raise web.HTTPForbidden(text='Cross-origin viewer requests are disabled')
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() in {'accept', 'content-type', 'range', 'cache-control'}}
        headers['Accept-Encoding'] = 'identity'
        if origin:
            headers['Origin'] = target
        path = '/' + tail.lstrip('/')
        if request.query_string:
            path += '?' + request.query_string
        url = URL(target + path, encoded=True)
        try:
            if request.headers.get('Upgrade', '').lower() == 'websocket':
                protocols = [item.strip() for item in request.headers.get('Sec-WebSocket-Protocol', '').split(',') if item.strip()]
                async with self.viewer_client.ws_connect(url, headers=headers, protocols=protocols, max_msg_size=64 * 1024 * 1024) as remote:
                    local = web.WebSocketResponse(protocols=protocols, max_msg_size=64 * 1024 * 1024)
                    local.headers.update({'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp'})
                    await local.prepare(request)
                    async def relay(source, destination):
                        async for message in source:
                            if message.type == WSMsgType.BINARY:
                                await destination.send_bytes(message.data)
                            elif message.type == WSMsgType.TEXT:
                                await destination.send_str(message.data)
                            elif message.type == WSMsgType.ERROR:
                                break
                    tasks = [asyncio.create_task(relay(local, remote)), asyncio.create_task(relay(remote, local))]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await local.close()
                    return local
            async with self.viewer_client.request(request.method, url, headers=headers, data=await request.read(), allow_redirects=False) as upstream:
                body = await upstream.read()
                content_type = upstream.headers.get('Content-Type', 'application/octet-stream')
                outgoing = {key: value for key, value in upstream.headers.items()
                            if key.lower() not in HOP_HEADERS and key.lower() not in {'etag', 'last-modified'}}
                rewrite = any(kind in content_type for kind in ('text/html', 'javascript', 'text/css')) or (tail == 'api/studio' and 'application/json' in content_type)
                if rewrite:
                    body = scoped_text(body, content_type, scope)
                    if cube and 'text/html' in content_type:
                        body = body.replace(b'http://localhost:8877/', b'/cube-states/').replace(b'http://localhost:8877', b'/cube-states/')
                if rewrite or tail.startswith('api/'):
                    outgoing['Cache-Control'] = 'no-store'
                location = outgoing.get('Location')
                if location:
                    if location.startswith(target):
                        outgoing['Location'] = scope + location[len(target):]
                    elif location.startswith('/') and not location.startswith('//'):
                        outgoing['Location'] = scope + location
                    else:
                        raise web.HTTPBadGateway(text='Viewer redirect escaped its fixed local target')
                response = web.Response(body=body, status=upstream.status, headers=outgoing,
                                        zlib_executor_size=256 * 1024)
                if any(kind in content_type for kind in ('text/', 'javascript', 'application/json', 'application/wasm')):
                    response.enable_compression()
                return response
        except (ClientError, OSError, asyncio.TimeoutError) as error:
            raise web.HTTPServiceUnavailable(text='States viewer is updating. Please try again shortly.') from error

    async def replay_read(self, expression):
        from allocated_io import DIRECTORY
        python,_=self.targets()
        code=(HERE/'replay_io.py').read_text()+'\nprint(json.dumps('+expression+',separators=(\",\",\":\")))'
        args = ['--jobid=3077705','--overlap','--exact','--ntasks=1','--cpus-per-task=1',
                'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-c',code]
        # File-backed Slurm output avoids its stdout callback. Read and clean up
        # through the same SSH connection instead of opening two more sessions.
        template = shlex.quote(DIRECTORY + '/dashboard-replay.XXXXXXXX')
        remote = (
            'folder=$(mktemp -d ' + template + ') || exit 1\n'
            'trap \'rm -f "$folder/out" "$folder/err"; rmdir "$folder"\' EXIT\n'
            'srun --input=/dev/null --output="$folder/out" --error="$folder/err" '
            + shlex.join(args) + '\n'
            'status=$?\n'
            'if [ "$status" -eq 0 ]; then cat "$folder/out"; '
            'elif [ -f "$folder/err" ]; then cat "$folder/err" >&2; fi\n'
            'exit "$status"'
        )
        process = await asyncio.create_subprocess_exec(
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'chimera-login', remote,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(process.communicate(), 45)
            if process.returncode:
                raise RuntimeError(error.decode(errors='replace')[-300:])
            return await asyncio.to_thread(json.loads, output)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def load_replay_scene(self, run, payload):
        from replay_io import compile_scene, model_fingerprint
        identity = payload['metadata']['identity']
        path = WORKSPACE / REPLAY_MODEL
        fingerprint = payload.get('model_fingerprint')
        if fingerprint and path.is_file() and await asyncio.to_thread(model_fingerprint, path) == fingerprint:
            return await asyncio.to_thread(compile_scene, path, identity, payload['body_names'])
        return await self.replay_read('remote_scene('+repr(run)+')')

    async def visual_response(self, request, scene):
        if request.query.get('format') != 'compact':
            return web.json_response(scene)
        from replay_io import encode_scene
        use_gzip = 'gzip' in request.headers.get('Accept-Encoding', '').lower()
        def encode():
            data = encode_scene(scene)
            return gzip.compress(data, compresslevel=3) if use_gzip else data
        # Share the encoded payload across concurrent viewers of the same scene.
        key = (id(scene), use_gzip)
        cache = getattr(self, 'visual_cache', None)
        if cache is None:
            self.visual_cache = cache = OrderedDict()
        if key not in cache:
            cache[key] = (scene, asyncio.create_task(asyncio.to_thread(encode)))
            while len(cache) > 4:
                cache.popitem(last=False)
        data = await asyncio.shield(cache[key][1])
        headers = {'Vary': 'Accept-Encoding', 'Cache-Control': 'public, max-age=300'}
        if use_gzip:
            headers['Content-Encoding'] = 'gzip'
        return web.Response(body=data, content_type='application/json', headers=headers)

    async def episode_handler(self,request):
        from replay_io import reconstruct
        task=request.query.get('task','');stage=request.query.get('stage','');key=task+'/'+stage
        metadata=self.live.get('replays',{}).get(key)
        if metadata is None:return web.json_response(None,headers={'Cache-Control':'no-store'})
        cache_key=(self.live['run'],metadata['id'])
        if cache_key in self.replay_cache:
            self.replay_cache.move_to_end(cache_key)
            return web.json_response(self.replay_cache[cache_key])
        if self.replay_lock.locked():return web.json_response(dict(status='loading'),status=202)
        async with self.replay_lock:
            try:
                run=self.live['run'];identity=metadata['identity']
                cold = self.replay_scene is None or self.replay_scene['identity'] != identity
                payload = await self.replay_read(f'remote_clip({run!r},{key!r},None,{cold!r})')
                assert payload['metadata']['identity']==identity
                assert (payload['metadata']['task'], payload['metadata']['stage']) == (task, stage)
                if cold:
                    scene = await self.load_replay_scene(run, payload)
                    assert scene['identity'] == identity
                    self.replay_scene = scene
                replay=await asyncio.to_thread(reconstruct,payload,self.replay_scene['layout'])
                cache_key = (run, replay['id'])
                self.replay_cache[cache_key]=replay
                while len(self.replay_cache)>4:self.replay_cache.popitem(last=False)
                return web.json_response(replay,headers={'Cache-Control':'no-store'})
            except Exception as error:
                return web.json_response(dict(status='unavailable',error=str(error)),status=503)

    async def replay_scene_handler(self,request):
        if self.replay_scene is None:return web.json_response(None,status=404)
        if request.query.get('run')!=self.replay_scene['identity']['run']:raise web.HTTPConflict(text='Replay scene changed')
        return await self.visual_response(request, self.replay_scene['scene'])

    async def startup(self, app):
        self.viewer_client = ClientSession(timeout=ClientTimeout(total=120))
        self.tasks = [asyncio.create_task(self.bridge()), asyncio.create_task(self.providers())]
        self.preview_task = asyncio.create_task(asyncio.to_thread(self.preview))

    async def cleanup(self, app):
        for task in self.tasks: task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.viewer_client.close()


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8881);args=parser.parse_args()
    dash=Dashboard();app=web.Application(middlewares=[isolation], client_max_size=64*1024)
    app.router.add_get('/api/live', dash.live_handler)
    app.router.add_get('/api/workspace', dash.workspace_handler)
    app.router.add_get('/cube', lambda request: web.HTTPFound('/cube/'))
    app.router.add_get('/bc', lambda request: web.HTTPFound('/bc/'))
    app.router.add_route('*', '/bc/{tail:.*}', dash.bc_proxy)
    app.router.add_route('*', '/cube/{tail:.*}', dash.cube_proxy)
    app.router.add_get('/cube-states', lambda request: web.HTTPFound('/cube-states/'))
    app.router.add_route('*', '/cube-states/{tail:.*}', dash.cube_states_proxy)
    app.router.add_get('/states', lambda request: web.HTTPFound('/states/'))
    app.router.add_route('*', '/states/{tail:.*}', dash.states_proxy)
    app.router.add_get('/api/episode',dash.episode_handler)
    app.router.add_get('/api/replay-scene',dash.replay_scene_handler)
    async def scene(request):
        await dash.preview_task
        return await dash.visual_response(request, dash.scene)
    async def preview(request):
        await dash.preview_task
        task=request.query.get('task','base_side_cover')
        if task not in dash.previews:raise web.HTTPBadRequest(text='Unknown task')
        return web.json_response(dash.previews[task])
    app.router.add_get('/api/scene',scene);app.router.add_get('/api/preview',preview)
    async def index(request):
        if not dash.bc_experiment:
            return web.FileResponse(HERE/'dist/index.html')
        text = (HERE/'dist/index.html').read_text()
        text = text.replace('</head>', '<style>#training-projects,header nav{display:none!important}</style></head>')
        text = text.replace('Five parts.<br>One learning policy.', 'All-part BC + PPO.<br>Same native training setup.')
        text = text.replace('Modal live containers', 'Sail reserved CPUs')
        return web.Response(text=text, content_type='text/html')
    app.router.add_get('/',index)
    app.router.add_static('/assets',HERE/'dist/assets')
    app.on_startup.append(dash.startup);app.on_cleanup.append(dash.cleanup)
    web.run_app(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
