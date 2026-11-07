// Production browser lifecycle functions with metadata-only XR/HTTP fixtures.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as geometry from '../web/preview.mjs';
import {OwnerEpoch,CohortGuard,Buttons,InputQueue,acquisitionReady} from '../web/preview.mjs';
const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
function fixture(){
 const c={ownerEpoch:new OwnerEpoch(),ownerSession:'old-session',syncReceiveMs:1,serverOffsetUpperMs:1,sequence:5,canReconnect:true,lastBindingAttempt:-Infinity,
  viewGuard:new CohortGuard(),captureSequence:100,captureState:{},captureAcquiredMs:100,frameSequence:{main:100},frameErrors:{},frameReceived:{},calibration:{},
  session:{visibilityState:'visible'},controlBinding:false,synthetic:false,document:{visibilityState:'visible'},performance:{now:()=>2000},
  state:{enabled:false,stopped:false},localStop:false,localHold:false,buttons:new Buttons(),queue:new InputQueue(),enablePending:true,enablePendingSequence:5,
  requireGripRelease:{left:false,right:false},rebases:{},anchors:{},edits:{},wasGrip:{},latestGrips:{},config:null,viewOffsetUpperMs:null,
  Buttons,statusLabel:geometry.statusLabel,syncState(){},message(){},api:async()=>{throw Error('unexpected API call')},$(){return{}},lastHoldReason:null,inflight:false,
  physical:()=>false,immersiveView:{ready:false,reset(){},accept(){}},sceneLoadGeneration:0,sceneLoading:null};
 c.ownerEpoch.observe('first');
 for(const [start,end]of [['function releaseLocal(','async function api('],['function observeOwner(','async function refreshDesktopConfig('],['async function connectControls(','function controlHint('],['async function sendQueued(','function enqueue(']]){
  const a=source.indexOf(start),b=source.indexOf(end,a+start.length);assert.ok(a>=0&&b>a);vm.runInNewContext(source.slice(a,b),c);
 }
 return c;
}
function cohort(owner,sequence){return {owner_id:owner,frame_sequence:sequence,acquired_ms:sequence,state:{owner_id:owner,tools:{left:{position:[0,0,0]},right:{position:[0,0,0]}}},frames:{main:'jpg',left:'jpg',right:'jpg'}};}

test('owner replacement accepts its lower camera counter and fences the old owner/in-flight decode',()=>{
 const c=fixture(),old=cohort('first',100);const token=c.viewGuard.begin();assert.ok(c.viewGuard.accept(token,old));
 c.queue.push({source_ms:2000,actions:{enable:true},recovery_held:false});
 assert.ok(c.observeOwner({owner_id:'second'}));
 assert.equal(c.ownerSession,null);assert.equal(c.captureSequence,null);assert.equal(c.queue.take(2000),null);assert.equal(c.localHold,true);
 assert.equal(c.viewGuard.accept(token,old),false);
 const next=cohort('second',1);assert.ok(c.observeOwner(next));assert.ok(c.viewGuard.accept(c.viewGuard.begin(),next));
 assert.equal(c.observeOwner(old),false);assert.equal(c.ownerEpoch.id,'second');
 assert.equal(c.observeOwner({...next,state:{...next.state,owner_id:'first'}}),false);
});

test('rebind refreshes config/clock and discards gestures, without sending Enable or grip',async()=>{
 const c=fixture(),calls=[];c.observeOwner({owner_id:'second'});c.localStop=true;
 c.queue.push({source_ms:2000,actions:{enable:true},recovery_held:false});
 c.api=async(path,options)=>{calls.push({path,options});return path==='/api/config'?{owner_id:'second',server_ms:3000}:{owner_id:'second',session:'fresh',server_ms:3001};};
 await c.connectControls();assert.equal(c.ownerSession,'fresh');assert.equal(c.localHold,true);assert.equal(c.localStop,true);
 assert.equal(c.queue.take(2000),null);assert.equal(c.requireGripRelease.left,true);assert.equal(c.requireGripRelease.right,true);
 assert.deepEqual(calls.map(x=>x.path),['/api/config','/api/session']);assert.equal(c.state.enabled,false);assert.equal(c.buttons.stopCycle,true);
 assert.equal(c.config.owner_id,'second');assert.equal(c.calibration,null);
});

test('a replaced-owner or ended-XR authentication response cannot bind',async()=>{
 for(const mode of ['owner_changed','xr_ended']){
  const c=fixture();c.observeOwner({owner_id:'second'});let finish;const pending=new Promise(resolve=>finish=resolve);
  c.api=async path=>path==='/api/config'?{owner_id:'second',server_ms:3000}:pending;
  const binding=c.connectControls();await Promise.resolve();await Promise.resolve();
  if(mode==='owner_changed')c.observeOwner({owner_id:'third'});else c.session=null;
  finish({owner_id:'second',session:'stale-grant',server_ms:3001});await binding;
  assert.equal(c.ownerSession,null);assert.equal(c.controlBinding,false);
 }
});

test('late input replies cannot disarm a newly bound connection',async()=>{
 const c=fixture();c.queue.push({source_ms:2000,actions:{enable:true},controllers:{}});let finish;
 c.api=async()=>new Promise(resolve=>finish=resolve);
 const sending=c.sendQueued();await Promise.resolve();c.observeOwner({owner_id:'second'});c.ownerSession='new-session';c.localHold=false;
 finish({owner_id:'first',accepted:false,reason:'unknown_or_superseded_session'});await sending;
 assert.equal(c.ownerSession,'new-session');assert.equal(c.localHold,false);assert.equal(c.canReconnect,true);
});

test('a superseded session on the same owner does not automatically reclaim another operator',async()=>{
 const c=fixture();c.queue.push({source_ms:2000,actions:{enable:true},controllers:{}});
 c.api=async()=>({owner_id:'first',accepted:false,reason:'unknown_or_superseded_session'});
 await c.sendQueued();assert.equal(c.ownerSession,null);assert.equal(c.canReconnect,false);assert.equal(c.localHold,true);
});

test('hidden XR does not create authority',async()=>{
 const c=fixture();c.ownerSession=null;c.session.visibilityState='hidden';let called=false;c.api=async()=>{called=true};
 await c.connectControls();assert.equal(called,false);assert.equal(c.ownerSession,null);
});

test('production green indicators require an enabled fresh connection and released nearby grips',()=>{
 const c=fixture();Object.assign(c,{sides:['left','right'],syncHybridControls(){},layout(){},panelsLocked:()=>false,physicalPresentation(){},
  config:{acquire_radius_m:.05},serverOffsetUpperMs:1000,calibration:{},captureState:{tools:{left:{position:[0,0,0]},right:{position:[0,0,0]}}},captureAcquiredMs:3000,
  state:{enabled:true,stopped:false,tools:{left:{position:[0,0,0]},right:{position:[0,0,0]}},grabs:{left:false,right:false},goals:{},server_ms:3000},
  latestGrips:{left:{position:[.01,0,0]},right:{position:[.01,0,0]}},controllerStatus:{left:{tracked:true},right:{tracked:true}},
  requireGripRelease:{left:false,right:false},wasGrip:{left:false,right:false},readiness:{},lastHead:null,hybridWanted:false,
  markers:{},colors:{},point(){},robotToXR:(_,p)=>p,recoveryHeld:()=>false,workingDepth:()=>.5,updateVRStatus(){}});
 const elements={};c.$=id=>elements[id]??=( {style:{}} );
 c.sub=(a,b)=>a.map((x,i)=>x-b[i]);
 // Use the exact readiness helper imported from the production browser module.
 c.acquisitionReady=acquisitionReady;
 for(const side of c.sides)c.markers[side]={measured:{material:{color:{set:value=>c.colors[side]=value}}},goal:{}};
 for(const [start,end]of [['function active(','function recoveryHeld('],['function controlHint(','function syncState('],['function syncState(','function physicalPresentation(']]){
  const a=source.indexOf(start),b=source.indexOf(end,a+start.length);vm.runInNewContext(source.slice(a,b),c);
 }
 c.syncState();assert.equal(c.colors.left,'#72e6a0');assert.equal(c.colors.right,'#72e6a0');assert.match(c.controlHint(),/GREEN/);
 c.localHold=true;c.syncState();assert.equal(c.colors.left,'#65dddb');assert.match(c.controlHint(),/PRESS Y/);
 c.localHold=false;c.wasGrip.left=true;c.syncState();assert.equal(c.colors.left,'#65dddb');assert.equal(c.colors.right,'#72e6a0');assert.match(c.controlHint(),/RELEASE GRIP/);
 c.controllerStatus.left.grip=true;c.syncState();assert.equal(c.markers.left.measured.visible,false);assert.equal(c.markers.left.goal.visible,false);assert.equal(c.markers.right.measured.visible,true);
 c.controllerStatus.left.grip=false;c.state.grabs.left=true;c.syncState();assert.equal(c.markers.left.measured.visible,false);
 c.state.grabs.left=false;c.syncState();assert.equal(c.markers.left.measured.visible,true);assert.equal(c.markers.left.goal.visible,true);
 c.wasGrip.left=false;c.captureAcquiredMs=1000;c.syncState();assert.equal(c.colors.right,'#65dddb');assert.match(c.controlHint(),/FRESH VIDEO/);
});


test('new owner clears the previous Stop into Disabled and still requires fresh Y',()=>{
 const c=fixture();c.state.stopped=true;c.localStop=true;c.buttons.noteStop();
 c.observeOwner({owner_id:'second'});
 assert.equal(c.state.stopped,false);assert.equal(c.localStop,false);assert.equal(c.state.enabled,false);assert.equal(c.ownerSession,null);assert.equal(c.localHold,true);assert.equal(c.buttons.yReleased,false);
});

test('display-clock refresh is read-only, owner-bound and does not change control authority',async()=>{
 const c=fixture();Object.assign(c,{clockRefreshPending:false,lastClockRefresh:-Infinity,viewOffsetUpperMs:0,serverOffsetUpperMs:0});
 const start=source.indexOf('async function refreshDisplayClock('),end=source.indexOf('async function connectControls(',start);vm.runInNewContext(source.slice(start,end),c);
 const calls=[];c.api=async path=>{calls.push(path);return {owner_id:'first',server_ms:2050};};
 await c.refreshDisplayClock();assert.equal(c.serverOffsetUpperMs,50);assert.equal(c.ownerSession,'old-session');assert.equal(c.state.enabled,false);assert.deepEqual(calls,['/api/config']);
});


test('simple simulation Y enables on press and holding it cannot release torque',()=>{
 const b=new Buttons(),released={y:false,b:false,x:false,a:false};
 b.sample(released,0,{simpleY:true});
 const on=b.sample({...released,y:true},20,{simpleY:true});assert.equal(on.enable,true);assert.equal(on.recoveryHeld,false);
 const held=b.sample({...released,y:true},4000,{simpleY:true,enabled:true});assert.equal(held.torqueRelease,false);assert.equal(held.disable,false);
 b.sample(released,4020,{simpleY:true,enabled:true});
 assert.equal(b.sample({...released,y:true},4040,{simpleY:true,enabled:true}).disable,true);
 b.noteStop();assert.equal(b.sample({...released,y:true},4060,{simpleY:true}).enable,false);
});

test('production XR sampling sees left Y on press and sends direct grip with right controller absent',()=>{
 const c=fixture(),gp={mapping:'xr-standard',buttons:Array.from({length:6},()=>({pressed:false,value:0}))};
 Object.assign(c,{empty:()=>({tracked:false,position:[0,0,0],rotation:[1,0,0,0,1,0,0,0,1],grip:false,trigger:0}),sides:['left','right'],config:{simple_y:true,require_both_controllers:false,acquire_from_anywhere:true,acquire_radius_m:.05},
  state:{enabled:false,stopped:false,tools:{left:{position:[0,0,.2],rotation:[1,0,0,0,1,0,0,0,1]},right:{position:[0,0,.2],rotation:[1,0,0,0,1,0,0,0,1]}},grabs:{left:false,right:false}},
  session:{visibilityState:'visible',inputSources:[{handedness:'left',gripSpace:{},gamepad:gp}]},renderer:{xr:{getReferenceSpace:()=>({})}},
  calibration:{rotation:[1,0,0,0,1,0,0,0,1],translation:[0,0,0],referenceHead:[0,0,0]},referenceRing:{},hybridFocusChanged(){},disabledEdit(){},
  controllerStatus:{},panelWasGrip:{},lastHead:null,hybridWanted:false,packets:[],saveLayout(){},enqueue:p=>c.packets.push(p),syncState(){},
  physical:()=>false,active:()=>!!c.state.enabled&&!c.localStop&&!c.localHold});
 Object.assign(c,geometry);
 const start=source.indexOf('function sampleXR('),end=source.indexOf('function saveLayout(',start);vm.runInNewContext(source.slice(start,end),c);
 const pose={emulatedPosition:false,transform:{position:{x:.3,y:.2,z:-.4},orientation:{x:0,y:0,z:0,w:1}}};
 const frame={getViewerPose:()=>({...pose,transform:{...pose.transform,position:{x:0,y:0,z:0}}}),getPose:()=>pose};
 c.sampleXR(frame);assert.equal(c.packets.at(-1).actions.enable,false);
 gp.buttons[5].pressed=true;c.sampleXR(frame);assert.equal(c.packets.at(-1).actions.enable,true);assert.equal(c.packets.at(-1).controllers.left.tracked,true);assert.equal(c.packets.at(-1).controllers.right.tracked,false);
 c.state.enabled=true;gp.buttons[5].pressed=false;c.sampleXR(frame);
 gp.buttons[1].pressed=true;c.sampleXR(frame);assert.equal(c.packets.at(-1).controllers.left.grip,true);assert.ok(c.anchors.left);assert.ok(c.anchors.left.offset_m>.5);
 const right={mapping:'xr-standard',buttons:Array.from({length:6},()=>({pressed:false,value:0}))};
 c.session.inputSources.push({handedness:'right',gripSpace:{},gamepad:right});c.config.task_scenes=true;
 let time=2100;c.performance.now=()=>time;c.sampleXR(frame);
 time+=20;right.buttons[4].pressed=true;c.sampleXR(frame);assert.equal(c.packets.at(-1).actions.advance_task,false);
 time+=20;right.buttons[4].pressed=false;c.sampleXR(frame);assert.equal(c.packets.at(-1).actions.advance_task,true);
 time+=20;right.buttons[4].pressed=true;c.sampleXR(frame);
 let resets=0;for(let i=0;i<110;i++){time+=20;c.sampleXR(frame);resets+=Number(c.packets.at(-1).actions.environment_reset);}
 assert.equal(resets,1);time+=20;right.buttons[4].pressed=false;c.sampleXR(frame);assert.equal(c.packets.at(-1).actions.advance_task,false);

});

function acknowledgmentFixture(profile){
 const c=fixture();Object.assign(c,{config:{tracking_profile:profile},enableRequestedMs:2000,sides:['left','right'],cancellations:0,lossPacket:()=>c.cancellations++});
 const start=source.indexOf('function acceptState('),end=source.indexOf('async function poll(',start);
 vm.runInNewContext(source.slice(start,end),c);return c;
}

test('direct simulation accepts a 350ms Enable acknowledgment without sending Disable',()=>{
 const c=acknowledgmentFixture('direct-clutch-sim');
 c.checkEnableAcknowledgment(2350);assert.equal(c.enablePending,true);assert.equal(c.cancellations,0);assert.equal(c.state.enabled,false);
 c.acceptState({sequence:5,enabled:true,stopped:false,grabs:{}});
 c.checkEnableAcknowledgment(4500);assert.equal(c.enablePending,false);assert.equal(c.cancellations,0);assert.equal(c.state.enabled,true);
});

test('missing Enable acknowledgment still cancels at its bounded profile deadline',()=>{
 for(const [profile,timeout]of [['direct-clutch-sim',2000],['legacy',200]]){
  const c=acknowledgmentFixture(profile);
  c.checkEnableAcknowledgment(2000+timeout);assert.equal(c.cancellations,0);
  c.checkEnableAcknowledgment(2001+timeout);assert.equal(c.cancellations,1);assert.equal(c.enablePending,false);assert.equal(c.localHold,true);
 }
});
