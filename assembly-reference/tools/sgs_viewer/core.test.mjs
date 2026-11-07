import test from 'node:test';
import assert from 'node:assert/strict';
import {filterEntries,visibleWindow} from './core.mjs';

const entries=Array.from({length:10000},(_,i)=>({id:`state-${i}`,bank:i<5000?'one':'two',task:i%2?'motor':'horn',family:i%3?'loose':'near',split:i%5?'train':'eval',seed:i,status:'qualified',material_mm:i%17,depth_mm:i%11}));
test('every indexed state is reachable through bounded grid windows',()=>{
 const reached=new Set();for(let y=0;y<Math.ceil(entries.length/3)*240;y+=600){const w=visibleWindow(entries.length,y,600,3,240);assert.ok(w.end-w.start<=24);for(let i=w.start;i<w.end;i++)reached.add(i);}assert.equal(reached.size,10000);
});
test('filters combine without mutating source and preserve exact counts',()=>{
 const original=entries.map(e=>e.id);const actual=filterEntries(entries,{bank:'two',task:'motor',family:'near',split:'eval'});assert.deepEqual(actual.map(e=>e.seed).sort((a,b)=>a-b),entries.filter(e=>e.bank==='two'&&e.task==='motor'&&e.family==='near'&&e.split==='eval').map(e=>e.seed));assert.deepEqual(entries.map(e=>e.id),original);
});
test('reviews, exact state search and absent metrics stay meaningful',()=>{
 assert.equal(filterEntries(entries,{search:'state-9999'})[0].id,'state-9999');assert.equal(filterEntries(entries,{review:'flagged'},{'state-3':{status:'flagged'}}).length,1);assert.equal(filterEntries(entries,{review:'unreviewed'},{'state-3':{status:'flagged'}}).length,9999);
 const sorted=filterEntries([{...entries[0],id:'none',material_mm:null},{...entries[1],id:'zero',material_mm:0}],{sort:'overlap'});assert.equal(sorted[0].id,'zero');
});
