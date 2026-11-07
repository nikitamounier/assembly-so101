// Metric geometry and input/layout rules. No browser or Three dependency.
export const I = [1,0,0,0,1,0,0,0,1];
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const scale=(a,s)=>a.map(v=>v*s);
export const norm=a=>Math.hypot(...a);
export const mv=(r,p)=>[0,1,2].map(i=>r[i*3]*p[0]+r[i*3+1]*p[1]+r[i*3+2]*p[2]);
export const tr=r=>[r[0],r[3],r[6],r[1],r[4],r[7],r[2],r[5],r[8]];
export const mm=(a,b)=>[0,1,2].flatMap(i=>[0,1,2].map(j=>a[i*3]*b[j]+a[i*3+1]*b[j+3]+a[i*3+2]*b[j+6]));
export function quaternionMatrix(q){const {x,y,z,w}=q;return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];}
// Horizontal heading only: never rotate robot gravity to match a pitched phone/head.
export function calibrate(cameraPosition,headPosition,headRotation=I){
  const forward=mv(headRotation,[0,0,-1]);
  const h=Math.hypot(forward[0],forward[2]);
  if(h<0.05)throw new Error('Look horizontally to establish reference heading');
  const yaw=Math.atan2(-forward[0],-forward[2]),c=Math.cos(yaw),s=Math.sin(yaw);
  const rotation=mm([c,0,s,0,1,0,-s,0,c],[0,-1,0,0,0,1,-1,0,0]);
  return {rotation,translation:sub(headPosition,mv(rotation,cameraPosition)),referenceHead:[...headPosition]};
}
export const robotToXR=(c,p)=>add(mv(c.rotation,p),c.translation);
export const xrToRobot=(c,p)=>mv(tr(c.rotation),sub(p,c.translation));
export const xrRotationToRobot=(c,r)=>mm(tr(c.rotation),r);
export function cameraPlane(camera,calibration,depth=1.2){
  const rotation=mm(calibration.rotation,camera.rotation);
  const height=2*depth*Math.tan(camera.fovy_degrees*Math.PI/360);
  return {position:robotToXR(calibration,add(camera.position,mv(camera.rotation,[0,0,-depth]))),rotation,width:height*camera.aspect,height};
}
export function acquire(grip,tool,radius=0.05,anywhere=false){
  if(!anywhere&&norm(sub(grip.position,tool.position))>radius)return null;
  return {grip:structuredClone(grip),tool:structuredClone(tool),offset_m:norm(sub(grip.position,tool.position))};
}
export function clutchGoal(anchor,grip){return {position:add(anchor.tool.position,sub(grip.position,anchor.grip.position)),rotation:mm(mm(grip.rotation,tr(anchor.grip.rotation)),anchor.tool.rotation)};}
export const headOffset=(c,p)=>norm(sub(p,c.referenceHead));
export function defaultPanels(){return {left:{position:[-.32,-.12,-.40],width:.27},right:{position:[.32,-.12,-.40],width:.27}};}
export function validPanels(value){return value&&['left','right'].every(k=>Array.isArray(value[k]?.position)&&value[k].position.length===3&&value[k].position.every(Number.isFinite)&&norm(value[k].position)<3&&Number.isFinite(value[k].width)&&value[k].width>=.18&&value[k].width<=.9);}
export function panelHit(panel,point){const d=sub(point,panel.position),h=panel.width/(16/9);if(Math.abs(d[2])>.10||Math.abs(d[0])>panel.width/2+.025||Math.abs(d[1])>h/2+.025)return null;return d[0]>panel.width/2-.07&&d[1]<-h/2+.07?'resize':'drag';}
export function panelBegin(panel,point,enabled){const kind=enabled?null:panelHit(panel,point);return kind?{kind,panel:structuredClone(panel),point:[...point]}:null;}
export function panelUpdate(edit,point,enabled){if(enabled)return null;const d=sub(point,edit.point);if(edit.kind==='resize')return {...edit.panel,width:Math.max(.18,Math.min(.9,edit.panel.width+2*d[0]))};const position=add(edit.panel.position,d);return norm(position)<3?{...edit.panel,position}:edit.panel;}
// Exact acquisition times survive the single-flight queue. Stop removes older Enable.
export class InputQueue{
 constructor(){this.actions=[];this.latest=null;this.yRelease=null;this.recoveryHeld=false;}
 push(packet){
  const release=this.recoveryHeld&&packet.recovery_held===false;
  this.recoveryHeld=packet.recovery_held===true;
  if(packet.actions.stop){this.actions=[];this.latest=null;this.yRelease=null;}
  else if(release){
   // One revocation slot, with its original time. Discard older authority so
   // delivering this first cannot send a later original source before it.
   this.actions=[];this.latest=null;this.yRelease=packet;return;
  }
  if(Object.values(packet.actions).some(Boolean)){this.actions.push(packet);this.latest=null;}else this.latest=packet;
  this.actions=this.actions.slice(-4);
 }
 take(now){
  if(this.yRelease){const p=this.yRelease;this.yRelease=null;if(now-p.source_ms<200&&now>=p.source_ms)return p;}
  while(this.actions.length){const p=this.actions.shift();if(now-p.source_ms<200&&now>=p.source_ms)return p;}
  const p=this.latest;this.latest=null;return p&&now-p.source_ms<200&&now>=p.source_ms?p:null;
 }
 clear(){this.actions=[];this.latest=null;this.yRelease=null;this.recoveryHeld=false;}
}
// A tap fires on release so a long hold cannot also advance a task.
export class TaskButtons{
 constructor(){this.cancel();}
 cancel(){this.ready=false;this.since=null;this.used=false;this.last=null;}
 sample(a,now,blocked=false){
  let advanceTask=false,environmentReset=false;
  if(blocked||typeof a!=='boolean'||!Number.isFinite(now)){this.cancel();return {advanceTask,environmentReset};}
  if(this.last!==null&&(now<this.last||now-this.last>200)){this.cancel();}
  this.last=now;
  if(!a){advanceTask=this.since!==null&&!this.used;this.since=null;this.used=false;this.ready=true;}
  else if(this.ready){
   if(this.since===null)this.since=now;
   if(now-this.since>=2000&&!this.used){environmentReset=true;this.used=true;}
  }
  return {advanceTask,environmentReset};
 }
}

export function statusLabel({state={},localStop=false,localHold=false,connected=true,pending=false}={}){
 if(localStop||state.stopped||state.faults?.length)return 'STOPPED';
 if(!connected)return 'DISCONNECTED';
 if(pending)return 'ENABLING';
 if(state.torque_enabled===false)return 'TORQUE RELEASED';
 if(localHold&&state.enabled)return 'HELD';
 return state.enabled?'ENABLED':'DISABLED';
}

export class Buttons{
 constructor(){this.tasks=new TaskButtons();this.previous={y:true,x:true,b:true,a:true};this.yReleased=false;this.ySince=null;this.yWasEnabled=false;this.yUsed=false;this.yTorqueEligible=false;this.bSince=null;this.resetSent=false;this.stopCycle=false;this.resetReady=false;}
 disarm(){this.yReleased=false;this.previous.y=true;this.ySince=null;this.yUsed=false;this.yTorqueEligible=false;}
 noteStop(){this.stopCycle=true;this.resetReady=false;this.bSince=null;this.resetSent=false;this.disarm();}
 sample(buttons,now,{enabled=false,gripsReleased=true,freshGrip=false,simpleY=false,taskScenes=false}={}){
  const pressed=k=>buttons[k]===true,edge=k=>pressed(k)&&!this.previous[k];
  const taskActions=taskScenes?this.tasks.sample(buttons.a,now,pressed('b')):{advanceTask:false,environmentReset:false};
  const stop=edge('b');if(stop&&!this.stopCycle)this.noteStop();
  if(typeof buttons.b!=='boolean'){this.bSince=null;this.resetReady=false;}
  else if(!pressed('b')){this.bSince=null;this.resetSent=false;if(this.stopCycle)this.resetReady=true;}
  else if(this.resetReady&&this.bSince===null)this.bSince=now;
  const reset=this.stopCycle&&this.resetReady&&this.bSince!==null&&now-this.bSince>=1000&&!this.resetSent;
  if(reset){this.resetSent=true;this.stopCycle=false;this.resetReady=false;this.bSince=null;this.disarm();}
  if(simpleY){
   const y=this.yReleased&&edge('y')&&!this.stopCycle&&!pressed('b');
   if(!pressed('y'))this.yReleased=true;
   const out={enable:y&&!enabled,disable:y&&enabled,stop,reset,recoveryHeld:false,torqueRelease:false,recenter:edge('x'),save:!taskScenes&&edge('a'),...taskActions};
   this.previous={...buttons};return out;
  }
  let enable=false,disable=false;
  if(!this.stopCycle&&!pressed('b')&&this.yReleased&&edge('y')){
   this.ySince=now;this.yWasEnabled=enabled;this.yUsed=false;this.yTorqueEligible=!enabled&&gripsReleased;
  }
  if(this.ySince!==null&&!gripsReleased)this.yTorqueEligible=false;
  if(this.ySince!==null&&freshGrip)this.yUsed=true;
  const recoveryHeld=!this.stopCycle&&!pressed('b')&&pressed('y')&&this.ySince!==null;
  const torqueRelease=recoveryHeld&&this.yTorqueEligible&&!enabled&&gripsReleased&&now-this.ySince>=3000;
  if(torqueRelease)this.yUsed=true;
  if(!pressed('y')){
   if(this.previous.y&&this.ySince!==null&&!this.yUsed&&!this.stopCycle&&!pressed('b')){
    disable=this.yWasEnabled;enable=!this.yWasEnabled;
   }
   this.ySince=null;this.yReleased=true;this.yTorqueEligible=false;
  }
  const out={enable,disable,stop,reset,recoveryHeld,torqueRelease,recenter:edge('x'),save:!taskScenes&&edge('a'),...taskActions};
  this.previous={...buttons};return out;
 }
}

export function gateGrip(needsRelease,pressed){return {needsRelease:needsRelease&&pressed,grip:pressed&&!needsRelease};}

// Preserve the exact control pose across a change of display/input calibration.
// A held clutch temporarily has a raw-hand offset; release restores direct alignment.
export const rebaseFor=(sent,raw)=>({position:sub(sent.position,raw.position),rotation:mm(sent.rotation,tr(raw.rotation))});
export const applyRebase=(raw,rebase)=>rebase?{position:add(raw.position,rebase.position),rotation:mm(rebase.rotation,raw.rotation)}:raw;
export const rebaseInfo=rebase=>rebase?{translation_m:norm(rebase.position),rotation_degrees:Math.acos(Math.max(-1,Math.min(1,(rebase.rotation[0]+rebase.rotation[4]+rebase.rotation[8]-1)/2)))*180/Math.PI}:null;
export const validGamepad=gp=>!!(gp?.mapping==='xr-standard'&&gp.buttons?.length>=6&&Number.isFinite(gp.buttons[0]?.value)&&gp.buttons[0].value>=0&&gp.buttons[0].value<=1&&[1,4,5].every(i=>typeof gp.buttons[i]?.pressed==='boolean'));

export const validCohort=value=>!!(Number.isInteger(value?.frame_sequence)&&value.frame_sequence>=0&&Number.isFinite(value.acquired_ms)&&['main','left','right'].every(k=>typeof value.frames?.[k]==='string'&&value.frames[k].length>0)&&['left','right'].every(k=>Array.isArray(value.state?.tools?.[k]?.position)&&value.state.tools[k].position.length===3&&value.state.tools[k].position.every(Number.isFinite)));
export class CohortGuard{
 constructor(){this.generation=0;this.sequence=-1;}
 begin(){return ++this.generation;}
 reset(){this.generation++;this.sequence=-1;}
 accept(token,cohort){if(token!==this.generation||!validCohort(cohort)||cohort.frame_sequence<this.sequence)return false;this.sequence=cohort.frame_sequence;return true;}
}

// Readiness concerns both the displayed capture and current owner measurement.
// The caller supplies capture freshness from original acquisition/receipt ages.
export function acquisitionReady({active,grabbed,releaseLatch,gripPressed,fresh,gripPosition,capturedPosition,currentPosition,radius=.05,anywhere=false}){
 if(active!==true||grabbed!==false||releaseLatch!==false||gripPressed!==false||fresh!==true||!Number.isFinite(radius)||radius<=0)return false;
 if(![gripPosition,capturedPosition,currentPosition].every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)))return false;
 return anywhere||(norm(sub(gripPosition,capturedPosition))<=radius&&norm(sub(gripPosition,currentPosition))<=radius);
}

// A restarted owner's counters may be smaller. Retired owners cannot return.
export class OwnerEpoch {
 constructor(){this.id=null;this.generation=0;this.retired=new Set();}
 observe(id){
  if(typeof id!=='string'||!id)return 'same';
  if(id===this.id)return 'same';
  if(this.retired.has(id))return 'stale';
  const changed=this.id!==null;
  if(changed)this.retired.add(this.id);
  this.id=id;this.generation++;
  return changed?'changed':'initial';
 }
}
