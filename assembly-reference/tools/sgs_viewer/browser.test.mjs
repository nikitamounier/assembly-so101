// Optional end-to-end check against an already served export. No bank writes.
// PLAYWRIGHT_MODULE may point to an existing Playwright index.mjs installation.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {filterEntries} from './core.mjs';

const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.SGS_VIEWER_URL||'http://127.0.0.1:8766/';
const out=process.env.SGS_CHECK_OUTPUT||'artifacts_local/sgs-viewer-browser';
await mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.SGS_BROWSER_EXECUTABLE||undefined,
 args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--renderer-process-limit=2']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 const manifest=await (await page.request.get(`${base}manifest.json`)).json();
 const total=manifest.entries.length,initial=manifest.banks[0].entries,start=Date.now();
 await page.goto(base);await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`${initial} of ${total} entries`);
 await page.waitForFunction(()=>document.querySelector('#preview').complete&&document.querySelector('#preview').naturalWidth>0);
 const initialMs=Date.now()-start,initialCards=await page.locator('.card').count();
 await page.screenshot({path:`${out}/overview.png`,fullPage:true});
 await page.selectOption('#family','near');await page.selectOption('#split','train');
 const nearTrain=manifest.entries.filter(e=>e.bank===manifest.banks[0].name&&e.family==='near'&&e.split==='train').length;
 await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`${nearTrain} of ${total} entries`);
 assert.ok(nearTrain>0,'E2E fixture must include qualified near TRAIN starts');
 await page.click('#mark-reviewed');await page.selectOption('#review','reviewed');
 await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`1 of ${total} entries`);
 await page.click('#mark-flagged');await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`0 of ${total} entries`);
 await page.selectOption('#review','flagged');await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`1 of ${total} entries`);
 await page.click('[data-view=detail]');await page.waitForFunction(()=>document.querySelector('#preview').src.includes('-detail.jpg')&&document.querySelector('#preview').complete);
 await page.click('[data-view="3d"]');await page.waitForFunction(()=>document.querySelector('#three canvas')&&document.querySelector('#visual-message').textContent==='');
 await page.click('#focus-3d');await page.screenshot({path:`${out}/3d.png`,fullPage:true});
 await page.click('[data-view=side]');await page.click('#clear');await page.reload();
 await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`${total} of ${total} entries`);
 await page.selectOption('#review','flagged');await page.waitForFunction(expected=>document.querySelector('#count').textContent===expected,`1 of ${total} entries`);
 if(await page.locator('#error').isVisible())errors.push(await page.locator('#error').textContent());
 // This metadata-only fixture tests UI scale, not qualification or 10,000 real states.
 const synthetic={...manifest,entries:[...manifest.entries]};
 for(let i=total;i<10000;i++){const e=manifest.entries[i%total];synthetic.entries.push({...e,id:`synthetic-${i}`,seed:e.seed+10000000000+i});}
 const scalePage=await browser.newPage({viewport:{width:1440,height:1100}});scalePage.on('pageerror',error=>errors.push(error.message));
 await scalePage.route('**/manifest.json',route=>route.fulfill({json:synthetic}));
 const scaleStart=Date.now();await scalePage.goto(`${base}?bank=`);
 await scalePage.waitForFunction(()=>document.querySelector('#count').textContent==='10,000 of 10,000 entries'||document.querySelector('#count').textContent==='10000 of 10000 entries');
 const scaleOpenMs=Date.now()-scaleStart,scaleCards=await scalePage.locator('.card').count();assert.ok(scaleCards<40);
 await scalePage.evaluate(()=>{const s=document.querySelector('#scroller');s.scrollTop=s.scrollHeight;});
 const last=filterEntries(synthetic.entries,{}).at(-1).id;
 await scalePage.waitForFunction(id=>[...document.querySelectorAll('.card')].some(card=>card.dataset.id===id),last);
 const bottomCards=await scalePage.locator('.card').count();assert.ok(bottomCards<40);
 const cdp=await scalePage.context().newCDPSession(scalePage);await cdp.send('Performance.enable');
 const metrics=(await cdp.send('Performance.getMetrics')).metrics;
 const result={real_entries:total,initial_ms:initialMs,initial_dom_cards:initialCards,filters:true,review_persistence:true,
  images:true,native_3d:true,synthetic_metadata_entries:10000,synthetic_open_ms:scaleOpenMs,synthetic_dom_cards:scaleCards,
  synthetic_bottom_cards:bottomCards,synthetic_heap_bytes:metrics.find(m=>m.name==='JSHeapUsedSize')?.value,errors};
 await writeFile(`${out}/report.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));assert.deepEqual(errors,[]);
}finally{await browser.close();}
