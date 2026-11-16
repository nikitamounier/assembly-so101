import {get, head} from '@vercel/blob';
import {createHash} from 'node:crypto';
import {Readable} from 'node:stream';

export function keyFor(path, params) {
  if (['live', 'workspace', 'scene'].includes(path)) return path;
  if (path === 'preview' && /^[a-z0-9_]{1,80}$/.test(params.get('task') || '')) return 'preview-' + params.get('task');
  if (path === 'episode' && /^[a-z0-9_]{1,80}$/.test(params.get('task') || '') && /^[a-z0-9_]{1,80}$/.test(params.get('stage') || '')) {
    return 'episode-' + createHash('sha256').update(params.get('task') + '/' + params.get('stage')).digest('hex');
  }
  if (path === 'replay-scene' && (params.get('run') || '').length < 512 && params.get('run')) return 'replay-scene-' + createHash('sha256').update(params.get('run')).digest('hex');
  return null;
}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'GET' && req.method !== 'HEAD') return res.status(405).end();
  const url = new URL(req.url, 'https://assembly.invalid');
  const path = url.searchParams.get('path') || url.pathname.replace(/^\/api\//, '');
  const scope = url.searchParams.get('scope') || '';
  if (!['','cube','bc'].includes(scope)) return res.status(404).end();
  let key = keyFor(path, url.searchParams);
  if (scope === 'cube' && path === 'initial') key = 'initial';
  if (scope === 'cube' && path === 'episode') {
    const family = url.searchParams.get('family') || 'latest';
    key = /^[a-z0-9_]{1,80}$/.test(family) ? 'episode-'+family : null;
  }
  if (key && scope) key = scope+'-'+key;
  if (!key) return res.status(404).json({error: 'Unknown dashboard route'});
  try {
    if (path === 'scene' || path === 'replay-scene') {
      const blob = await head('feed/' + key + '.json.gz');
      return res.redirect(302, blob.url);
    }
    const result = await get('feed/' + key + '.json.gz', {access: 'public', useCache: false});
    if (!result || result.statusCode !== 200) return res.status(503).json({error: 'Awaiting published observation'});
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Encoding', 'gzip');
    if (req.method === 'HEAD') return res.status(200).end();
    Readable.fromWeb(result.stream).pipe(res);
  } catch (error) {
    console.error('Snapshot read failed', error.name);
    return res.status(503).json({error: 'Published observation is temporarily unavailable'});
  }
}
