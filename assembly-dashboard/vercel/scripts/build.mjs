import {readFile, writeFile, mkdir, cp} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dashboard = path.resolve(root, '..');
const workspace = path.resolve(dashboard, '..');
await mkdir(path.join(root,'public/states'), {recursive:true});
await cp(path.join(dashboard,'dist/assets'), path.join(root,'public/assets'), {recursive:true});
const inject = html => html.replace('<head>', '<head><script src="/hosted.js"></script>');
await writeFile(path.join(root,'public/index.html'), inject(await readFile(path.join(dashboard,'dist/index.html'),'utf8')));
const viewer = path.join(workspace,'assembly-so101/assembly_policy/viewer/static');
await cp(path.join(viewer,'assets'), path.join(root,'public/states/assets'), {recursive:true});
await cp(path.join(viewer,'collision-boxes.json'),path.join(root,'public/states/collision-boxes.json'));
await cp(path.join(viewer,'reference'),path.join(root,'public/states/reference'),{recursive:true});
await writeFile(path.join(root,'public/states/index.html'), inject((await readFile(path.join(viewer,'index.html'),'utf8')).replaceAll('="/assets/', '="/states/assets/')));
for (const scope of ['cube','bc']) {
  const response = await fetch('http://127.0.0.1:8881/'+scope+'/');
  let html;
  if (response.ok) html = await response.text();
  else {
    html = await readFile(path.join(scope==='cube'?path.join(workspace,'cube-dashboard'):dashboard,'dist/index.html'),'utf8');
    html = html.replaceAll('="/assets/','="/'+scope+'/assets/');
    if (scope === 'bc') html = html.replace('</head>','<style>#training-projects,header nav{display:none!important}</style></head>').replace('Five parts.<br>One learning policy.','All-part BC + PPO.<br>Same native training setup.');
  }
  await mkdir(path.join(root,'public',scope),{recursive:true});
  await writeFile(path.join(root,'public',scope,'index.html'),inject(html));
  await cp(path.join(scope==='cube'?path.join(workspace,'cube-dashboard'):dashboard,'dist/assets'),path.join(root,'public',scope,'assets'),{recursive:true});
}
