import test from 'node:test';
import assert from 'node:assert/strict';
import {TaskButtons,Buttons,InputQueue,statusLabel} from '../web/preview.mjs';

test('A advances only on a short release; a two-second hold resets once without advancing',()=>{
 const a=new TaskButtons();a.sample(false,0);
 assert.equal(a.sample(true,20).advanceTask,false);
 assert.equal(a.sample(false,120).advanceTask,true);
 a.sample(true,140);let resets=0;
 for(let t=160;t<=2500;t+=20)resets+=Number(a.sample(true,t).environmentReset);
 assert.equal(resets,1);assert.equal(a.sample(false,2520).advanceTask,false);
});

test('initial held A, controller loss, tracking gaps and B cancel task gestures',()=>{
 for(const interrupt of ['missing','gap','stop']){
  const a=new TaskButtons();assert.equal(a.sample(true,0).environmentReset,false);
  a.sample(false,20);a.sample(true,40);
  if(interrupt==='missing')a.sample(undefined,60);
  if(interrupt==='gap')a.sample(true,300);
  if(interrupt==='stop')a.sample(true,60,true);
  assert.equal(a.sample(false,320).advanceTask,false);
 }
});

test('production Buttons reserve A for task scenes and Stop clears a queued reset',()=>{
 const b=new Buttons(),released={a:false,b:false,x:false,y:false},options={simpleY:true,taskScenes:true};
 b.sample(released,0,options);assert.equal(b.sample({...released,a:true},20,options).save,false);
 assert.equal(b.sample(released,40,options).advanceTask,true);
 const queue=new InputQueue();queue.push({source_ms:50,actions:{environment_reset:true}});queue.push({source_ms:60,actions:{stop:true}});
 assert.deepEqual(queue.take(60).actions,{stop:true});assert.equal(queue.take(60),null);
});

test('status stays one short state without control instructions',()=>{
 assert.equal(statusLabel({state:{enabled:true}}),'ENABLED');
 assert.equal(statusLabel({state:{enabled:true},localHold:true}),'HELD');
 assert.equal(statusLabel({state:{stopped:true}}),'STOPPED');
 assert.equal(statusLabel({pending:true}),'ENABLING');
 assert.equal(statusLabel({connected:false}),'DISCONNECTED');
 assert.equal(statusLabel(),'DISABLED');
});
