import {put} from '@vercel/blob';
import {readFile, writeFile} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.loadEnvFile(path.join(root,'.env.local'));
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));
const hashes = new Map();
const state = {started_at:Date.now()/1000, source:'http://127.0.0.1:8881', uploads:0, errors:[]};
const hash = value => createHash('sha256').update(value).digest('hex');
async function source(route) {
  const response = await fetch(state.source+route, {signal:AbortSignal.timeout(120000)});
  if (!response.ok) {
    const fallback = {'/cube/api/live':'../../cube-dashboard/live-cache.json','/bc/api/live':'../live-cache-bc.json'}[route];
    if (fallback) return {...JSON.parse(await readFile(path.join(root,fallback),'utf8')),connection:'archived'};
    throw Error('Local snapshot '+route+' returned '+response.status);
  }
  return response.json();
}
async function store(key, payload) {
  const json = JSON.stringify(payload);
  const digest = hash(json);
  if (hashes.get(key) === digest) return;
  const data = gzipSync(json, {level:4});
  await put('feed/'+key+'.json.gz',data,{access:'public',allowOverwrite:true,addRandomSuffix:false,contentType:'application/octet-stream',cacheControlMaxAge:60,multipart:data.length>4000000});
  hashes.set(key,digest);state.uploads++;
}
async function core() {
  const live = await source('/api/live');
  await store('live',live);
  await store('workspace',await source('/api/workspace'));
  state.latest_iteration=live.history?.at(-1)?.it;
  state.last_published_at=Date.now()/1000;
  await writeFile(path.join(root,'publication-feed-status.json'),JSON.stringify(state,null,2)+'\n');
  return live;
}
async function initial() {
  await store('scene',await source('/api/scene'));
  const manifest = JSON.parse(await readFile(path.join(root,'../../robotics-assembly-3.0/assets_local/wowrobo/initial-valid-resets-v1/stages/manifest.json'),'utf8'));
  const tasks = ['base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module',...(manifest.task_names||[])];
  for (const task of new Set(tasks)) {
    try {await store('preview-'+task, await source('/api/preview?task='+encodeURIComponent(task)));}
    catch (error) {console.error('Preview unavailable:',task,error.message);}
  }
  for (const scope of ['bc','cube']) {
    for (const route of ['live','scene',...(scope==='bc'?['workspace']:['initial'])]) {
      try {await store(scope+'-'+route,await source('/'+scope+'/api/'+route));}
      catch(error) {console.error('Archived observation unavailable:',scope,route,error.message);}
    }
  }
  for (const family of ['latest','all_loose','pickup','pre_handoff','handoff','inhand','partial','near']) {
    try {await store('cube-episode-'+family,await source('/cube/api/episode'+(family==='latest'?'':'?family='+family)));}
    catch(error) {console.error('Archived cube replay unavailable:',family);}
  }
}
async function replays(live) {
  if (live.run) await store('replay-scene-'+hash(live.run),await source('/api/replay-scene?run='+encodeURIComponent(live.run)));
  for (const name of Object.keys(live.replays || {})) {
    const split=name.indexOf('/'); if (split<0) continue;
    const task=name.slice(0,split),stage=name.slice(split+1);
    try {await store('episode-'+hash(name),await source('/api/episode?task='+encodeURIComponent(task)+'&stage='+encodeURIComponent(stage)));}
    catch(error) {console.error('Replay unavailable:',name,error.message);}
  }
}
function report(error) {
  state.errors=[...state.errors.slice(-9),{time:Date.now()/1000,message:error.message}];
  console.error(error.message);
}
await core();
if (process.argv.includes('--core-only')) process.exit(0);
await initial();
await replays(await core());
state.initial_complete=true;
await writeFile(path.join(root,'publication-feed-status.json'),JSON.stringify(state,null,2)+'\n');
if (process.argv.includes('--once')) process.exit(0);
// Metrics use the existing local read cache; replays are a separate slow loop.
async function metricsLoop(){for(;;){try{await core();}catch(error){report(error);}await sleep(20000);}}
async function replayLoop(){for(;;){await sleep(120000);try{await replays(await source('/api/live'));}catch(error){report(error);}}}
await Promise.all([metricsLoop(),replayLoop()]);
