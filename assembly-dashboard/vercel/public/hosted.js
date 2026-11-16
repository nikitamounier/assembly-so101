// Hosted transport only: saved poses, policy tensors and qualification stay intact.
const originalFetch = window.fetch.bind(window);
let assetMap;
const assets = () => assetMap ||= originalFetch('/hosted-assets.json').then(r => {
  if (!r.ok) throw Error('Published assets unavailable');
  return r.json();
});
async function decode(response) {
  if (!response.ok) throw Error('Published asset unavailable (' + response.status + ')');
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
}
async function data(name, range) {
  const url = (await assets())[name];
  if (!url) throw Error('Unknown published asset');
  const response = await originalFetch(url, range ? {headers:{Range:`bytes=${range[0]}-${range[0]+range[1]-1}`}} : {});
  if (range && response.status !== 206) throw Error('Snapshot range request was not honored');
  return decode(response);
}
const asJSON = value => new Response(JSON.stringify(value), {headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
const statePage = location.pathname.startsWith('/states/');
let current;
let indexPromise;
const blocks = new Map();
window.fetch = async (input, options = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url, location.href);
  if (statePage && url.origin === location.origin) {
    const route = url.pathname.replace(/^\/states/, '');
    try {
      if (route === '/api/catalog' || route === '/api/refresh') return asJSON(await data('states/catalog.json.gz'));
      if (route === '/api/scene') return asJSON(await data('states/scene.json.gz'));
      if (route === '/api/select') {
        const id = JSON.parse(options.body).id;
        const index = await (indexPromise ||= data('states/packed-index.json.gz'));
        const row = index[id];
        if (!row) return new Response('Unknown reset ID', {status:404});
        const [file, start, length] = row;
        const key = file + ':' + start;
        if (!blocks.has(key)) {
          blocks.set(key, data('states/' + file, [start, length]));
          if (blocks.size > 3) blocks.delete(blocks.keys().next().value);
        }
        current = structuredClone((await blocks.get(key))[id]);
        if (current.point_cloud?.available) current.point_cloud.flat = current.point_cloud.points.flat();
        return asJSON(current);
      }
      if (route === '/api/state') return current ? asJSON(current) : new Response('Select a reset first', {status:404});
      if (route === '/api/studio/stop') return asJSON({stopped:true});
      if (route.startsWith('/api/')) return new Response('This inspection control is available in the local dashboard.', {status:501});
      if (url.pathname === '/collision-boxes.json') return originalFetch('/states/collision-boxes.json', options);
      if (url.pathname.startsWith('/reference/')) return originalFetch('/states' + url.pathname, options);
    } catch (error) { return new Response(error.message, {status:503}); }
  }
  const archivedScope = /^\/(cube|bc)\//.exec(location.pathname)?.[1];
  if (archivedScope && url.origin === location.origin && url.pathname.startsWith('/api/')) url.pathname = '/'+archivedScope+url.pathname;
  const response = await originalFetch(archivedScope ? url.href : input, options);
  // Large immutable scenes are sent directly from Blob, outside Function limits.
  if (/^\/(?:cube\/|bc\/)?api\//.test(url.pathname) && response.ok && response.url.split('?')[0].endsWith('.gz') && !response.headers.get('Content-Encoding')) return asJSON(await decode(response));
  return response;
};
if (statePage) addEventListener('DOMContentLoaded', () => {
  for (const id of ['advanced-inspection','reward-apply']) {
    const button = document.getElementById(id);
    if (button) {
      button.disabled = true;
      button.title = 'Available in the local dashboard';
      new MutationObserver(() => {if (!button.disabled) button.disabled = true;}).observe(button, {attributes:true,attributeFilter:['disabled']});
    }
  }
});
