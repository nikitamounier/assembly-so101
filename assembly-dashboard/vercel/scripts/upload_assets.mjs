import {put, head} from '@vercel/blob';
import {readFile, writeFile, readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.loadEnvFile(path.join(root,'.env.local'));
const directory = path.join(root,'public/data/states');
const names = (await readdir(directory)).filter(name => name.endsWith('.pack') || ['catalog.json.gz','scene.json.gz','packed-index.json.gz'].includes(name));
const mapFile = path.join(root,'public/hosted-assets.json');
let urls = {};
try { urls = JSON.parse(await readFile(mapFile,'utf8')); } catch {}
for (const name of names) {
  if (urls['states/'+name]) continue;
  const bytes = await readFile(path.join(directory,name));
  const digest = createHash('sha256').update(bytes).digest('hex');
  const pathname = 'assets/' + digest + '/' + name;
  let result;
  try {
    result = await put(pathname, bytes, {access:'public',addRandomSuffix:false,contentType:'application/octet-stream',multipart:true});
  } catch (error) {
    if (error.name !== 'BlobAlreadyExistsError') throw error;
    result = await head(pathname);
    if (result.size !== bytes.length) throw Error('Published asset size mismatch');
  }
  urls['states/'+name] = result.url;
  await writeFile(mapFile, JSON.stringify(urls,null,2)+'\n');
  console.log(JSON.stringify({file:name,bytes:bytes.length,sha256:digest}));
}
console.log('Published immutable asset files:',Object.keys(urls).length);
