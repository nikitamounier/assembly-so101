import * as THREE from '/vendor/three.module.js';
import {ImmersiveScene} from './immersive.mjs';
import {I,add,sub,mv,tr,mm,quaternionMatrix,calibrate,robotToXR,xrToRobot,xrRotationToRobot,cameraPlane,acquire,headOffset,defaultPanels,validPanels,panelBegin,panelUpdate,InputQueue,Buttons,statusLabel,gateGrip,rebaseFor,applyRebase,rebaseInfo,validGamepad,validCohort,CohortGuard,acquisitionReady,OwnerEpoch} from './preview.mjs';
import {defaultAlignment,validAlignment,editAlignment,containImage,phoneFresh,nativeCaptureFresh,hybridViewState,PhoneGuard} from './hybrid.mjs';
const $=id=>document.getElementById(id), sides=['left','right'];
const empty=()=>({tracked:false,position:[0,0,0],rotation:[...I],grip:false,trigger:0});
let showWristViews=false;
let config=null,state={enabled:false,stopped:false,tools:{},goals:{},grabs:{}},calibration=null,session=null,ownerSession=null,syncReceiveMs=null,sequence=0,inflight=false,synthetic=false,enablePending=false,enablePendingSequence=null,enableRequestedMs=0,serverOffsetUpperMs=null;
let latestGrips={},rebases={},anchors={},edits={},wasGrip={},panelWasGrip={},requireGripRelease={left:true,right:true},buttons=new Buttons(),queue=new InputQueue(),lastHead=null,localStop=false,localHold=false,frameSequence={},frameErrors={},frameReceived={},lastPoll=0,pollBusy=false,noticeUntil=0,lastHoldReason=null;
let panels=defaultPanels();try{const saved=JSON.parse(localStorage.getItem('so101-wrist-layout-v1'));if(validPanels(saved))panels=saved;}catch{}
const ownerEpoch=new OwnerEpoch();let controlBinding=false,canReconnect=true,lastBindingAttempt=-Infinity,lastClockRefresh=-Infinity,clockRefreshPending=false,controllerStatus={},readiness={};
let captureState=null,captureSequence=null,captureAcquiredMs=null,viewsBusy=false;const viewGuard=new CohortGuard();
let viewOffsetUpperMs=null,hybridWanted=false,phoneSample=null,phoneReceivedMs=0,phoneRoundtripMs=0,phoneTexture=null,phoneBusy=false,lastPhonePoll=0,phoneProblem='No live camera selected. Simulation only.';
let hybridSuspended=false,hybridRefreshUntil=null,hybridMotionHeld=false;
const phoneGuard=new PhoneGuard();let phoneAlignment=defaultAlignment();try{const saved=JSON.parse(localStorage.getItem('so101-phone-alignment-v1'));if(validAlignment(saved))phoneAlignment=saved;}catch{}
const scene=new THREE.Scene();scene.background=new THREE.Color('#10151b');
const immersiveView=new ImmersiveScene(scene);let sceneLoading=null,sceneLoadGeneration=0;
const camera=new THREE.PerspectiveCamera(64,innerWidth/innerHeight,.01,30);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);renderer.xr.enabled=true;renderer.xr.setReferenceSpaceType('local');renderer.outputColorSpace=THREE.SRGBColorSpace;$('viewport').appendChild(renderer.domElement);
const matRot=r=>new THREE.Matrix4().set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
const orient=(obj,r)=>obj.quaternion.setFromRotationMatrix(matRot(r));
const point=(obj,p)=>obj.position.set(...p);
function message(text){$('notice').textContent=text;$('notice').style.display='block';noticeUntil=performance.now()+7000;}
function label(text,width=1){const c=document.createElement('canvas');c.width=1024;c.height=112;const x=c.getContext('2d');x.fillStyle='#17242fee';x.fillRect(0,0,1024,112);x.fillStyle='#d5e4ea';x.font='500 32px system-ui';x.fillText(text,26,67);const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;const m=new THREE.Mesh(new THREE.PlaneGeometry(width,width*112/1024),new THREE.MeshBasicMaterial({map:t,transparent:true,depthTest:false}));m.renderOrder=9;return m;}
const viewMeshes={},viewGroups={},textures={},labels={},panelHandles={};
for(const key of ['main',...sides]){const g=new THREE.Group(),m=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({color:'#31404c',side:THREE.DoubleSide,depthWrite:false,depthTest:false}));m.renderOrder=key==='main'?0:2;g.add(m);const edge=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1,1)),new THREE.LineBasicMaterial({color:key==='main'?'#5c8790':'#94c4c4'}));edge.position.z=.002;g.add(edge);const l=label(key==='main'?'WORKSPACE / FIXED MONOCULAR REFERENCE':key.toUpperCase()+' WRIST / MEASURED SIMULATION',1);g.add(l);labels[key]=l;viewMeshes[key]=m;viewGroups[key]=g;g.userData={key,edge};scene.add(g);if(key!=='main'){const h=new THREE.Mesh(new THREE.PlaneGeometry(.035,.035),new THREE.MeshBasicMaterial({color:'#a9dcd5',side:THREE.DoubleSide}));h.position.z=.006;g.add(h);panelHandles[key]=h;}}
const realBackground=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({depthTest:false,depthWrite:false,side:THREE.DoubleSide}));
const nativeOverlay=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({transparent:true,opacity:phoneAlignment.opacity,depthTest:false,depthWrite:false,side:THREE.DoubleSide}));
realBackground.renderOrder=0;nativeOverlay.renderOrder=1;realBackground.visible=nativeOverlay.visible=false;viewGroups.main.add(realBackground,nativeOverlay);
const markers={};for(const side of sides){markers[side]={};for(const kind of ['measured','goal']){const mesh=new THREE.Mesh(new THREE.SphereGeometry(kind==='measured'?.020:.011,16,12),new THREE.MeshBasicMaterial({color:kind==='measured'?'#65dddb':'#e8ba72',depthTest:false,transparent:true,opacity:.94}));mesh.renderOrder=12;mesh.visible=false;scene.add(mesh);markers[side][kind]=mesh;}}
const referenceRing=new THREE.Mesh(new THREE.RingGeometry(.07,.073,40),new THREE.MeshBasicMaterial({color:'#8cb4b8',side:THREE.DoubleSide,transparent:true,opacity:.65,depthTest:false}));referenceRing.visible=false;scene.add(referenceRing);
const vrStatus=label('DISABLED',.28);scene.add(camera);camera.add(vrStatus);vrStatus.position.set(-.39,.30,-.8);let vrStatusText='';
function updateVRStatus(text){if(text===vrStatusText)return;vrStatusText=text;const c=vrStatus.material.map.image,x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#17242fee';x.fillRect(0,0,c.width,c.height);x.fillStyle=localStop?'#ffaaa4':'#d5e4ea';x.font='500 52px system-ui';x.fillText(text,24,75);vrStatus.material.map.needsUpdate=true;}
let displayRotation=I;
function displayToXR(p){return add(calibration.referenceHead,mv(displayRotation,p));}
function xrToDisplay(p){return mv(tr(displayRotation),sub(p,calibration.referenceHead));}
function physical(){return config?.execution==='physical_measured';}
function workingDepth(){return config?.working_depth_m??config?.reference_depth_m??.5;}
function layout(){if(!calibration||!config)return;loadImmersiveScene();immersiveView.align(calibration);const p=cameraPlane(config.camera,calibration,workingDepth());point(viewGroups.main,p.position);orient(viewGroups.main,p.rotation);sizeView('main',p.width,p.height);layoutHybrid(p.width,p.height);viewGroups.main.visible=!immersiveView.ready;displayRotation=mm(calibration.rotation,config.camera.rotation);for(const side of sides){viewGroups[side].visible=!immersiveView.ready||showWristViews;point(viewGroups[side],displayToXR(panels[side].position));orient(viewGroups[side],displayRotation);sizeView(side,panels[side].width,panels[side].width/(16/9));panelHandles[side].visible=!panelsLocked();}vrStatus.visible=!!session;point(referenceRing,add(calibration.referenceHead,mv(displayRotation,[0,0,-.6])));orient(referenceRing,displayRotation);}
function sizeView(key,w,h){viewMeshes[key].scale.set(w,h,1);viewGroups[key].userData.edge.scale.set(w,h,1);labels[key].scale.set(w,w,1);labels[key].position.set(0,h/2+w*.06,.005);if(panelHandles[key])panelHandles[key].position.set(w/2-.017,-h/2+.017,.006);}
function active(){return !!state.enabled&&!localStop&&!localHold;}
function recoveryHeld(){return !!buttons.yReleased&&buttons.previous.y===true&&!buttons.stopCycle&&!localStop&&!localHold;}
function panelsLocked(){return !!state.enabled||enablePending||recoveryHeld();}
function observeOwner(value){
 if(value?.owner_id&&value?.state?.owner_id&&value.owner_id!==value.state.owner_id)return false;
 const outcome=ownerEpoch.observe(value?.owner_id??value?.state?.owner_id);
 if(outcome==='stale')return false;
 if(outcome==='changed'){
  ownerSession=null;syncReceiveMs=null;serverOffsetUpperMs=null;sequence=0;canReconnect=true;lastBindingAttempt=-Infinity;lastClockRefresh=-Infinity;
  localStop=false;buttons=new Buttons();state={...state,enabled:false,stopped:false,grabs:{},goals:{}};
  immersiveView.reset();sceneLoadGeneration++;sceneLoading=null;viewGuard.reset();captureSequence=null;captureState=null;captureAcquiredMs=null;frameSequence={};frameErrors={};frameReceived={};calibration=null;
  releaseLocal('Simulation restarted. Reconnecting controls; release Y and grips, then tap Y.');
  if(!session)refreshDesktopConfig();
 }
 return true;
}
async function refreshDesktopConfig(){
 const generation=ownerEpoch.generation,sent=performance.now();
 try{
  const next=await api('/api/config');
  if(session||generation!==ownerEpoch.generation||next.owner_id!==ownerEpoch.id)return;
  config=next;viewOffsetUpperMs=next.server_ms-sent;setDesktopReference();
 }catch(e){message('Waiting for the updated scene: '+e.message);}
}
async function refreshDisplayClock(){
 if(clockRefreshPending||synthetic||performance.now()-lastClockRefresh<5000)return;
 clockRefreshPending=true;lastClockRefresh=performance.now();const generation=ownerEpoch.generation,sent=performance.now();
 try{
  const next=await api('/api/config');
  if(generation!==ownerEpoch.generation||next.owner_id!==ownerEpoch.id)return;
  viewOffsetUpperMs=next.server_ms-sent;
  if(ownerSession)serverOffsetUpperMs=viewOffsetUpperMs;
 }catch{}finally{clockRefreshPending=false;}
}
async function connectControls(){
 if(!session||ownerSession||controlBinding||!canReconnect||synthetic||session.visibilityState!=='visible'||document.visibilityState!=='visible')return;
 if(performance.now()-lastBindingAttempt<1000)return;
 lastBindingAttempt=performance.now();controlBinding=true;const xr=session;
 try{
  const sent=performance.now(),next=await api('/api/config');
  if(session!==xr||!observeOwner(next))return;
  config=next;viewOffsetUpperMs=next.server_ms-sent;const generation=ownerEpoch.generation;
  const stamp=performance.now(),auth=await api('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_ms:stamp})});
  if(session!==xr||generation!==ownerEpoch.generation)return;
  if(auth.owner_id&&auth.owner_id!==ownerEpoch.id){observeOwner(auth);return;}
  syncReceiveMs=performance.now();serverOffsetUpperMs=auth.server_ms-stamp;ownerSession=auth.session;sequence=0;
  buttons=new Buttons();localStop=localStop||!!state.stopped;if(localStop)buttons.noteStop();calibration=null;
  releaseLocal('Controls connected. Release Y and grips, press Y, then hold a grip anywhere to move that arm.');
 }catch(e){message('Controls reconnecting: '+e.message);}
 finally{controlBinding=false;}
}
async function loadImmersiveScene(){
 if(!config?.immersive_view||immersiveView.owner===config.owner_id||sceneLoading===config.owner_id)return;
 const identity=config.owner_id,generation=sceneLoadGeneration;sceneLoading=identity;
 try{
  const value=await api('/api/scene');
  if(generation!==sceneLoadGeneration||identity!==config.owner_id||value.owner_id!==identity)return;
  immersiveView.load(value);immersiveView.align(calibration);immersiveView.accept(state);syncState();
 }catch(e){if(generation===sceneLoadGeneration)message('3D view unavailable: '+e.message);}
 finally{if(generation===sceneLoadGeneration)sceneLoading=null;}
}
function controlHint(){
 if(!session)return 'ENTER QUEST VR TO CONTROL ARMS';
 if(!ownerSession)return canReconnect?'RECONNECTING CONTROLS · KEEP GRIPS RELEASED':'CONTROL SESSION ENDED · EXIT AND RE-ENTER VR';
 if(state.reason==='modeled_collision_hold')return 'CONTACT HOLD · HOLD Y + FRESH GRIP AT TOOL TO ESCAPE';
 if(!active()&&!recoveryHeld()){if((config.require_both_controllers===false?sides.every(s=>!controllerStatus[s]?.tracked):sides.some(s=>!controllerStatus[s]?.tracked)))return 'WAKE CONTROLLER · KEEP IT IN VIEW';return 'RELEASE GRIPS · PRESS Y TO ENABLE';}
 const now=performance.now()+serverOffsetUpperMs;
 if(serverOffsetUpperMs===null||(!immersiveView.ready&&(captureAcquiredMs===null||now<captureAcquiredMs||now-captureAcquiredMs>=500))||now<state.server_ms||now-state.server_ms>=200)return 'WAIT FOR FRESH VIDEO / CONTROL';
 return sides.map(side=>{
  const name=side==='left'?'L':'R';
  if(!controllerStatus[side]?.tracked)return name+': CONTROLLER NOT TRACKED';
  if(state.grabs?.[side])return name+': HOLDING';
  if(requireGripRelease[side]||wasGrip[side])return name+': RELEASE GRIP FIRST';
  if(readiness[side])return name+(config.acquire_from_anywhere?': GRIP ANYWHERE TO MOVE':': GREEN · SQUEEZE GRIP');
  const p=latestGrips[side]?.position,t=state.tools?.[side]?.position;
  return p&&t?name+': MOVE '+Math.ceil(Math.hypot(...sub(p,t))*100)+' CM TO SPHERE':name+': WAIT FOR TRACKING';
 }).join(' / ');
}
function syncState(){syncHybridControls();const enabled=active();const status=statusLabel({state,localStop,localHold,connected:!!config&&(!session||!!ownerSession),pending:enablePending});$('mode').textContent=status+(state.task?' · '+state.task.replaceAll('_',' ').toUpperCase():'');$('mode').style.borderColor=enabled?'#80c3ac':localStop||state.stopped?'#cf7e78':'#426367';$('save').disabled=panelsLocked();$('reset-panels').disabled=panelsLocked();$('disable').disabled=!session&&!state.enabled;layout();if(!calibration)return;for(const side of sides){const measured=captureState?.tools?.[side],hideMarker=!!controllerStatus[side]?.grip||!!state.grabs?.[side];markers[side].measured.visible=!!measured&&!hideMarker;if(measured)point(markers[side].measured,robotToXR(calibration,measured.position));const grip=latestGrips[side];const serverNow=performance.now()+serverOffsetUpperMs;const fresh=!!ownerSession&&serverOffsetUpperMs!==null&&(immersiveView.ready||serverNow>=captureAcquiredMs&&serverNow-captureAcquiredMs<500)&&serverNow>=state.server_ms&&serverNow-state.server_ms<200;const ready=acquisitionReady({active:active()||(recoveryHeld()&&state.recovery_held===true),grabbed:!!state.grabs?.[side],releaseLatch:requireGripRelease[side],gripPressed:!!wasGrip[side],fresh,gripPosition:grip?.position,capturedPosition:measured?.position,currentPosition:state.tools?.[side]?.position,radius:config.acquire_radius_m,anywhere:config.acquire_from_anywhere===true});readiness[side]=ready;markers[side].measured.material.color.set(ready?'#72e6a0':'#65dddb');let goal=grip;if(state.grabs?.[side]&&state.goals?.[side]&&active())goal=state.goals[side];markers[side].goal.visible=!!goal&&!hideMarker;if(goal)point(markers[side].goal,robotToXR(calibration,goal.position));}updateVRStatus($('mode').textContent);const offsets=sides.filter(s=>anchors[s]).map(s=>s+' '+(anchors[s].offset_m*1000).toFixed(1)+' mm').join(' · ');$('detail').textContent=(immersiveView.ready?'Live stereo geometry at metric scale. Lean to inspect the assembly; wrist dots hide while gripping.':'Loading native 3D geometry; camera preview shown meanwhile.')+(offsets?' Client clutch-offset estimate: '+offsets+'; goal begins at measured tool.':(config.acquire_from_anywhere?' Hold a grip anywhere; initial hand/tool offset is retained.':' Acquire within 50 mm; initial grip/tool offset is retained.'))+(sides.some(s=>rebases[s])?' Recenter adds a temporary raw-hand offset; release/regrab restores direct alignment.':'');physicalPresentation();}
function physicalPresentation(){
 if(!physical())return;
 document.querySelector('.keys').textContent='Grip: acquire measured sphere · Trigger: jaw · Tap Y: Enable / measured Hold · B: Stop · release B/grips, hold B 1s: reset · X: recenter. Hardware has no Y collision bypass or torque-release gesture.';
 for(const side of sides)viewGroups[side].visible=false;
 viewMeshes.main.visible=false;nativeOverlay.visible=false;
 $('robot-opacity').disabled=true;$('opacity-value').textContent='OFF';
 $('guide').textContent='PHYSICAL ARMS · actual iPhone view required. Wrist renders and robot overlay are hidden. FK markers use provisional alignment; no real obstacle detection.';
 $('detail').textContent='Physical measured joints; modeled FK/contact checks only. Camera/base/optical alignment remains unqualified. No physical wrist-camera feed.';
 $('connection').textContent='LOCAL HARDWARE · '+(state.reason||'passive');
 caption('main','LIVE IPHONE / MEASURED FK MARKERS · ALIGNMENT PROVISIONAL');
 updateVRStatus($('mode').textContent);
}
function releaseLocal(reason,preserveTaskGesture=false){if(!preserveTaskGesture)buttons.tasks?.cancel();lastHoldReason=typeof reason==='string'?reason.slice(0,240):null;buttons.disarm();enablePending=false;enablePendingSequence=null;requireGripRelease={left:true,right:true};rebases={};anchors={};edits={};wasGrip={};panelWasGrip={};latestGrips={};queue.clear();localHold=true;syncState();if(reason)message(reason);}
async function api(path,options){const r=await fetch(path,{cache:'no-store',...options});if(!r.ok)throw new Error(path+' '+r.status);return r.json();}
async function sendQueued(){
 if(inflight||!ownerSession||synthetic)return;const packet=queue.take(performance.now());if(!packet)return;
 inflight=true;const identity=ownerSession,generation=ownerEpoch.generation;
 try{
  const result=await api('/api/input',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json'},body:JSON.stringify({...packet,live_view_selected:physical()?hybridWanted:undefined,session:identity,sync_receive_ms:syncReceiveMs,sequence:++sequence})});
  if(identity!==ownerSession||generation!==ownerEpoch.generation||!observeOwner(result))return;
  if(identity!==ownerSession||generation!==ownerEpoch.generation)return;
  if(result.tools)state=result;if(packet.actions.enable)enablePendingSequence=sequence;
  if(result.accepted===false){
   if(result.reason==='unknown_or_superseded_session'){ownerSession=null;canReconnect=false;}
   releaseLocal('Control held: '+(result.reason||'input rejected'));
  }
  syncState();
 }catch(e){if(identity===ownerSession){releaseLocal('Connection interrupted. Motion held; fresh Y required after recovery.');$('connection').textContent=e.message;}}
 finally{inflight=false;sendQueued();}
}
function enqueue(packet){queue.push(packet);sendQueued();}
function lossPacket(action='disable'){const actions={enable:false,disable:false,stop:false,reset:false};actions[action]=true;enqueue({source_ms:performance.now(),focus:false,controllers:{left:empty(),right:empty()},actions,client_hold_reason:lastHoldReason});}
function disabledEdit(side,p,grip){if(panelsLocked()||(immersiveView.ready&&!showWristViews))return;if(grip&&!panelWasGrip[side]){for(const key of sides){if(Object.values(edits).some(e=>e?.key===key))continue;const edit=panelBegin(panels[key],xrToDisplay(p),false);if(edit){edits[side]={...edit,key};break;}}}if(grip&&edits[side]){const e=edits[side],updated=panelUpdate(e,xrToDisplay(p),panelsLocked());if(updated)panels[e.key]=updated;layout();}if(!grip)delete edits[side];}
function sampleXR(frame){const reference=renderer.xr.getReferenceSpace(),viewer=frame.getViewerPose(reference),source_ms=performance.now();const focused=session.visibilityState==='visible'&&document.visibilityState==='visible'&&!!viewer&&!viewer.emulatedPosition;
 if(!viewer||!focused){controllerStatus={};hybridFocusChanged(false);if(!localHold){releaseLocal('XR focus/tracking lost. Fresh Y required after recovery.');lossPacket();}return;}
 hybridFocusChanged(true);
 const head=[viewer.transform.position.x,viewer.transform.position.y,viewer.transform.position.z];lastHead=head;
 if(!calibration){calibration=calibrate(config.camera.position,head,quaternionMatrix(viewer.transform.orientation));layout();}
 const distance=headOffset(calibration,head);$('alignment').textContent=immersiveView.ready?'STEREO 3D · lean and look around · X recenters':'Loading 3D geometry';referenceRing.visible=!immersiveView.ready&&distance>.03;
 let freshGrip=false;const inputs={},rawXR={},physicalGrips={},b={y:false,x:false,b:false,a:false},sources=[...session.inputSources],counts=Object.fromEntries(sides.map(s=>[s,sources.filter(i=>i.handedness===s).length]));
 for(const input of sources){const side=input.handedness;if(!sides.includes(side))continue;const pose=input.gripSpace&&frame.getPose(input.gripSpace,reference),gp=input.gamepad;if(!pose||pose.emulatedPosition||counts[side]!==1||!validGamepad(gp)){inputs[side]=empty();controllerStatus[side]={tracked:false};requireGripRelease[side]=true;delete latestGrips[side];delete rebases[side];delete anchors[side];delete edits[side];wasGrip[side]=false;panelWasGrip[side]=false;continue;}const pos=[pose.transform.position.x,pose.transform.position.y,pose.transform.position.z],rot=quaternionMatrix(pose.transform.orientation),grip=!!gp?.buttons[1]?.pressed;
 if(side==='left'){b.y=!!gp?.buttons[5]?.pressed;b.x=!!gp?.buttons[4]?.pressed;}else{b.b=!!gp?.buttons[5]?.pressed;b.a=!!gp?.buttons[4]?.pressed;}
 if(grip&&!panelWasGrip[side])freshGrip=true;physicalGrips[side]=grip;rawXR[side]={position:pos,rotation:rot};if(!grip)delete rebases[side];const raw={position:xrToRobot(calibration,pos),rotation:xrRotationToRobot(calibration,rot)},mapped=applyRebase(raw,rebases[side]);latestGrips[side]=mapped;disabledEdit(side,pos,grip);panelWasGrip[side]=grip;
 const gated=gateGrip(requireGripRelease[side],grip);requireGripRelease[side]=gated.needsRelease;const armGrip=gated.grip;
 if(active()&&armGrip&&!wasGrip[side]&&state.tools?.[side])anchors[side]=acquire(mapped,state.tools[side],config.acquire_radius_m,config.acquire_from_anywhere===true);if(!grip)delete anchors[side];wasGrip[side]=armGrip;
 controllerStatus[side]={tracked:true,grip};inputs[side]={tracked:true,...mapped,grip:armGrip,trigger:Math.min(1,Math.max(0,gp?.buttons[0]?.value||0))};}
 for(const side of sides)if(!inputs[side]){inputs[side]=empty();controllerStatus[side]={tracked:false};requireGripRelease[side]=true;delete latestGrips[side];delete rebases[side];delete anchors[side];delete edits[side];wasGrip[side]=false;panelWasGrip[side]=false;}
 if(!inputs.left.tracked){buttons.disarm();b.y=true;b.x=buttons.previous.x;}
 if(!inputs.right.tracked){delete b.b;delete b.a;}
 const actions=buttons.sample(b,source_ms,{enabled:active(),simpleY:config.simple_y===true,taskScenes:config.task_scenes===true,gripsReleased:sides.every(s=>config.require_both_controllers===false&&!inputs[s].tracked||inputs[s].tracked&&physicalGrips[s]===false),freshGrip});if(physical()){actions.recoveryHeld=false;actions.torqueRelease=false;}if((physical()&&!hybridWanted)||(hybridWanted&&hybridDisplayState(source_ms).hold)){actions.enable=false;actions.recoveryHeld=false;actions.torqueRelease=false;buttons.disarm();}if(actions.recenter){const old=calibration;try{const next=calibrate(config.camera.position,head,quaternionMatrix(viewer.transform.orientation));for(const side of sides){if(!rawXR[side])continue;const raw={position:xrToRobot(next,rawXR[side].position),rotation:xrRotationToRobot(next,rawXR[side].rotation)};if(active()&&inputs[side].grip&&(anchors[side]||state.grabs?.[side]))rebases[side]=rebaseFor(inputs[side],raw);else delete rebases[side];const mapped=applyRebase(raw,rebases[side]);inputs[side]={...inputs[side],...mapped};latestGrips[side]=mapped;}calibration=next;layout();message('Reference recentered. Held arm goals are preserved with a temporary grip-space offset. Release / regrab to restore direct alignment. The 3D workspace follows the new reference.');}catch(e){calibration=old;message(e.message);}} if(actions.save&&!panelsLocked())saveLayout();
 if(actions.environmentReset||actions.advanceTask){releaseLocal(undefined,true);enablePending=false;localStop=false;}if(actions.stop){localStop=true;releaseLocal('Stopped. Release B, hold B for 1 second to reset, then use fresh Y.');}if(actions.reset){localStop=false;localHold=true;}if(actions.recoveryHeld&&!localStop)localHold=false;if(actions.disable){releaseLocal(physical()?'Disabled: measured position Hold; physical torque release is a separate attended operation.':(config.simple_y?'Disabled. Press Y to enable again.':'Disabled: position Hold, servo effort retained. Hold Y 3s with both grips released for torque release.'));}if(actions.enable&&!localStop){localHold=false;enablePending=true;enableRequestedMs=source_ms;enablePendingSequence=null;requireGripRelease={left:true,right:true};rebases={};anchors={};edits={};}
 enqueue({source_ms,focus:focused,recovery_held:actions.recoveryHeld&&!localHold,controllers:inputs,actions:{enable:actions.enable&&!localStop,disable:actions.disable,stop:actions.stop,reset:actions.reset,torque_release:actions.torqueRelease,environment_reset:actions.environmentReset,advance_task:actions.advanceTask}});syncState();}
function saveLayout(){if(panelsLocked())return;try{localStorage.setItem('so101-wrist-layout-v1',JSON.stringify(panels));message('Wrist panel layout saved locally. Main reference view remains fixed.');}catch{message('Local layout storage unavailable.');}}
$('show-wrists').onchange=()=>{showWristViews=$('show-wrists').checked;layout();};
$('reset-panels').onclick=()=>{if(panelsLocked())return;panels=defaultPanels();edits={};desktopEdit=null;try{localStorage.removeItem('so101-wrist-layout-v1');}catch{}layout();message('Wrist panels restored within nominal grip reach. Display layout only.');};$('save').onclick=saveLayout;$('disable').onclick=()=>{releaseLocal('Disabled. Grip wrist panels to drag; grip their lower-right corner to resize.');lossPacket();};$('stop').onclick=()=>{buttons.noteStop();localStop=true;releaseLocal('Stop requested. Release B, then hold B for 1 second to reset; fresh Y required.');lossPacket('stop');};
$('enter').onclick=async()=>{if(session){await session.end();return;}if(!config){message('Simulation config is unavailable.');return;}try{if(!navigator.xr||!await navigator.xr.isSessionSupported('immersive-vr'))throw new Error('WebXR immersive VR unavailable. Use Quest HTTPS / secure localhost. Desktop remains a read-only preview.');const xr=await navigator.xr.requestSession('immersive-vr');session=xr;document.body.classList.add('xr-active');calibration=null;ownerSession=null;sequence=0;buttons=new Buttons();localStop=!!state.stopped;if(localStop)buttons.noteStop();releaseLocal();synthetic=false;canReconnect=true;lastBindingAttempt=-Infinity;await renderer.xr.setSession(xr);await connectControls();$('enter').textContent='Exit VR';xr.addEventListener('end',()=>{lossPacket();session=null;document.body.classList.remove('xr-active');ownerSession=null;releaseLocal();$('enter').textContent='Enter Quest VR';setDesktopReference();});xr.addEventListener('visibilitychange',()=>{hybridFocusChanged(xr.visibilityState==='visible'&&document.visibilityState==='visible');if(xr.visibilityState!=='visible'){releaseLocal('XR session hidden. Fresh Y required.');lossPacket();}});message(physical()?'Physical arms start Disabled. Publish and select the live iPhone, release Y, then tap Y. B Stops; no collision bypass or torque-release gesture.':'Starts Disabled. Press Y, then hold either grip to move its arm from your current hand position. Trigger closes the jaw; B Stops.');}catch(e){if(session)try{await session.end();}catch{}session=null;ownerSession=null;message(e.message);}};
window.addEventListener('blur',()=>{hybridFocusChanged(false);if(session){releaseLocal('Window focus lost. Fresh Y required.');lossPacket();}});
window.addEventListener('focus',()=>hybridFocusChanged(document.visibilityState==='visible'&&(!session||session.visibilityState==='visible')));
document.addEventListener('visibilitychange',()=>hybridFocusChanged(document.visibilityState==='visible'&&(!session||session.visibilityState==='visible')));
window.addEventListener('pagehide',()=>{if(ownerSession){releaseLocal();lossPacket();}});
function setDesktopReference(){if(!config)return;calibration=calibrate(config.camera.position,[0,1.55,0],I);layout();point(camera,calibration.referenceHead);orient(camera,mm(calibration.rotation,config.camera.rotation));camera.fov=90;camera.updateProjectionMatrix();syncState();}
function caption(key,text){const c=labels[key].material.map.image,x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#17242fee';x.fillRect(0,0,c.width,c.height);x.fillStyle='#d5e4ea';x.font='500 27px system-ui';x.fillText((key==='main'?'FIXED MONO':key.toUpperCase()+' WRIST')+' / '+text,22,66);labels[key].material.map.needsUpdate=true;}
async function commitViews(cohort,token){
 if(!validCohort(cohort))throw new Error('Invalid native camera cohort');
 const keys=['main',...sides];if(cohort.robot_overlay)keys.push('robot_overlay');const decoded=await Promise.allSettled(keys.map(async key=>{const r=await fetch(key==='robot_overlay'?cohort.robot_overlay:cohort.frames[key]);if(!r.ok)throw new Error('Image decode fetch '+key);return createImageBitmap(await r.blob(),{imageOrientation:'flipY',premultiplyAlpha:'none'});}));
 const failed=decoded.some(r=>r.status==='rejected');
 if(failed||!viewGuard.accept(token,cohort)){for(const r of decoded)if(r.status==='fulfilled')r.value.close();if(failed)throw new Error('Camera cohort decode failed; previous cohort retained');return false;}
 // No await in this commit: all three native images and measured tools change together.
 const nextTextures=decoded.map(r=>{const t=new THREE.Texture(r.value);t.flipY=false;t.colorSpace=THREE.SRGBColorSpace;t.needsUpdate=true;return t;});
 const selected=structuredClone(cohort.state);
 for(const [i,key] of keys.entries()){
  const old=textures[key];textures[key]=nextTextures[i];const mesh=key==='robot_overlay'?nativeOverlay:viewMeshes[key];mesh.material.map=nextTextures[i];mesh.material.color.set('#ffffff');mesh.material.needsUpdate=true;
  frameSequence[key]=cohort.frame_sequence;frameReceived[key]=performance.now();delete frameErrors[key];if(key!=='robot_overlay')caption(key,'SIM COHORT '+cohort.frame_sequence);if(old){old.image?.close?.();old.dispose();}
 }
 if(!cohort.robot_overlay){const old=textures.robot_overlay;nativeOverlay.material.map=null;delete textures.robot_overlay;if(old){old.image?.close?.();old.dispose();}}
 captureState=selected;captureSequence=cohort.frame_sequence;captureAcquiredMs=cohort.acquired_ms;syncState();return true;
}
async function pollViews(){if(viewsBusy||synthetic)return;viewsBusy=true;const token=viewGuard.begin();try{const cohort=await api('/api/views');if(synthetic||!observeOwner(cohort)||token!==viewGuard.generation)return;if(cohort.frame_sequence!==captureSequence)await commitViews(cohort,token);}catch(e){if(token===viewGuard.generation){for(const key of ['main',...sides]){frameErrors[key]=e.message;caption(key,'UNAVAILABLE / LAST '+(captureSequence??'NONE'));}}}finally{viewsBusy=false;}}

function acceptState(next){
 if(Number.isInteger(next.scene_generation)&&next.scene_generation!==(state.scene_generation??0)){
  localStop=false;buttons=new Buttons();releaseLocal();
 }

 if(next.reason==='original_recovery_input_expired'){buttons.disarm();message('Recovery expired. Release Y and grips, then hold Y and freshly grip at the tool.');}
 if(next.stopped&&!localStop){buttons.noteStop();localStop=true;releaseLocal('Stopped by local owner. Release B, then hold B 1s to reset.');}
 for(const side of sides)if(state.grabs?.[side]&&!next.grabs?.[side]){requireGripRelease[side]=true;delete rebases[side];delete anchors[side];wasGrip[side]=false;}
 state=next;immersiveView.accept(next);if(enablePending&&enablePendingSequence!==null&&next.sequence>=enablePendingSequence){enablePending=false;enablePendingSequence=null;}
}
function checkEnableAcknowledgment(now){
 const timeout=config?.tracking_profile==='direct-clutch-sim'?2000:200;
 if(enablePending&&now-enableRequestedMs>timeout){releaseLocal('Enable request expired. Release Y, then press it again.');lossPacket();}
}
async function poll(now){if(pollBusy||synthetic||now-lastPoll<40)return;lastPoll=now;pollBusy=true;try{const next=await api('/api/state');if(synthetic||!observeOwner(next))return;acceptState(next);refreshDisplayClock();if(session&&!ownerSession)connectControls();$('connection').textContent='LOCAL SIM · '+(next.reason||'ready')+' · '+(next.simulation_s||0).toFixed(1)+'s';syncState();}catch(e){$('connection').textContent='Simulation unavailable';if(session&&!localHold){releaseLocal('Simulation connection interrupted. Motion held; fresh Y required.');}}finally{pollBusy=false;}}
let lastFrames=0;renderer.setAnimationLoop((time,frame)=>{hybridFocusChanged(document.visibilityState==='visible'&&(!session||session.visibilityState==='visible'));checkHybrid(time);if(!synthetic&&time-lastPhonePoll>100&&($('hybrid-controls').open||hybridWanted)){lastPhonePoll=time;pollPhone();}checkEnableAcknowledgment(performance.now());if(config&&frame&&session)try{sampleXR(frame);}catch(e){releaseLocal(e.message);lossPacket();}if(!session&&performance.now()>noticeUntil)$('notice').style.display='none';poll(time);if(!synthetic&&time-lastFrames>40){lastFrames=time;pollViews();}for(const key of ['main',...sides])if(frameReceived[key]&&performance.now()-frameReceived[key]>1500)caption(key,'LAST SIM FRAME '+frameSequence[key]+' / STALE');immersiveView.frame(time);renderer.render(scene,camera);});
window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
// Desktop panel editing affects only saved display geometry, never arm/session inputs.
const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let desktopEdit=null;function pointerOnPanel(e,key){pointer.set(e.clientX/innerWidth*2-1,-e.clientY/innerHeight*2+1);raycaster.setFromCamera(pointer,camera);const plane=new THREE.Plane().setFromNormalAndCoplanarPoint(new THREE.Vector3(0,0,1).applyQuaternion(viewGroups[key].quaternion),viewGroups[key].position),target=new THREE.Vector3();return raycaster.ray.intersectPlane(plane,target)?xrToDisplay(target.toArray()):null;}
renderer.domElement.addEventListener('pointerdown',e=>{if(session||panelsLocked()||!calibration)return;for(const key of sides){if(!viewGroups[key].visible)continue;const p=pointerOnPanel(e,key),edit=p&&panelBegin(panels[key],p,false);if(edit){desktopEdit={...edit,key};renderer.domElement.setPointerCapture(e.pointerId);break;}}});renderer.domElement.addEventListener('pointermove',e=>{if(!desktopEdit||panelsLocked())return;const p=pointerOnPanel(e,desktopEdit.key);if(p){panels[desktopEdit.key]=panelUpdate(desktopEdit,p,false);layout();}});renderer.domElement.addEventListener('pointerup',()=>desktopEdit=null);
window.SO101Preview={hybridSnapshot(){return {selected:hybridWanted,visible:realBackground.visible,overlay_visible:nativeOverlay.visible,physical:physical(),modeled_wrist_visible:sides.some(s=>viewGroups[s].visible),modeled_main_visible:viewMeshes.main.visible,refresh_pending:hybridSuspended||hybridRefreshUntil!==null,local_motion_held:localHold,last_hold_reason:lastHoldReason,native_original_fresh:nativeCaptureFresh(captureAcquiredMs,serverOffsetUpperMs??viewOffsetUpperMs,performance.now()),phone:phoneSample?{generation:phoneSample.generation,sequence:phoneSample.sequence,age_upper_ms:phoneSample.age_upper_ms,decoded_server_ms:phoneSample.decoded_server_ms}:null,alignment:{...phoneAlignment},overlay_sequence:frameSequence.robot_overlay??null,problem:phoneProblem};},setState(next){synthetic=true;viewGuard.reset();state=structuredClone(next);localHold=false;syncState();},async setViews(cohort){synthetic=true;viewGuard.reset();return commitViews(cohort,viewGuard.begin());},async setFrames(frames,measuredState=state){synthetic=true;viewGuard.reset();return commitViews({frame_sequence:(captureSequence??0)+1,acquired_ms:measuredState.server_ms??0,state:measuredState,frames},viewGuard.begin());},setCamera(value){if(value.position)point(camera,value.position);if(value.rotation)orient(camera,value.rotation);if(value.fov){camera.fov=value.fov;camera.updateProjectionMatrix();}},setConfig(value){config=structuredClone(value);setDesktopReference();},setGrips(value){synthetic=true;viewGuard.reset();latestGrips=structuredClone(value);syncState();},snapshot(){return {synthetic,headset:!!session,immersion:immersiveView.snapshot(),render_stats:{triangles:renderer.info.render.triangles,calls:renderer.info.render.calls},hud:{text:vrStatusText,position:vrStatus.position.toArray(),head_locked:vrStatus.parent===camera},marker_visibility:Object.fromEntries(sides.map(s=>[s,{measured:markers[s].measured.visible,goal:markers[s].goal.visible}])),controls:{bound:!!ownerSession,binding:controlBinding,canReconnect,owner_id:ownerEpoch.id,generation:ownerEpoch.generation,last_hold_reason:lastHoldReason,tracking:structuredClone(controllerStatus),readiness:{...readiness},hint:controlHint(),mode:config?.tracking_profile},working_depth_m:workingDepth(),camera:structuredClone(config?.camera),calibration:structuredClone(calibration),panels:structuredClone(panels),state:structuredClone(state),capture:{frame_sequence:captureSequence,acquired_ms:captureAcquiredMs,state:structuredClone(captureState)},frameSequence:{...frameSequence},frameErrors:{...frameErrors},rebases:Object.fromEntries(sides.map(s=>[s,rebaseInfo(rebases[s])])),clientClutchOffsetEstimates:Object.fromEntries(sides.map(s=>[s,anchors[s]?.offset_m??null]))};}};

function hybridDisplayState(now){const result=hybridViewState({selected:hybridWanted,phoneReady:!!phoneTexture&&phoneFresh(phoneSample,phoneReceivedMs,phoneRoundtripMs,now),nativeReady:(physical()||!!nativeOverlay.material.map)&&nativeCaptureFresh(captureAcquiredMs,serverOffsetUpperMs??viewOffsetUpperMs,now),suspended:hybridSuspended,refreshUntil:hybridRefreshUntil,now});if(physical())result.overlay=false;return result;}
function hybridFocusChanged(visible){
 if(!hybridWanted)return;
 if(!visible){if(hybridSuspended)return;hybridSuspended=true;hybridRefreshUntil=null;phoneGuard.invalidate();releaseLocal('XR focus lost. Motion held; fresh Y required.');if(ownerSession)lossPacket();return;}
 if(hybridSuspended){hybridSuspended=false;hybridRefreshUntil=performance.now()+1000;phoneGuard.invalidate();pollPhone();pollViews();}
}
function layoutHybrid(width,height){
 const display=hybridDisplayState(performance.now());viewMeshes.main.visible=!physical()&&!display.background;realBackground.visible=display.background;nativeOverlay.visible=display.overlay;nativeOverlay.scale.set(width,height,1);nativeOverlay.material.opacity=phoneAlignment.opacity;
 if(phoneSample){const fit=containImage(phoneSample.width,phoneSample.height,width,height);realBackground.scale.set(fit.width*phoneAlignment.scale,fit.height*phoneAlignment.scale,1);realBackground.position.set(phoneAlignment.x*width,phoneAlignment.y*height,0);}
}
function syncHybridControls(){
 const locked=panelsLocked();for(const id of ['hybrid-mode','robot-opacity','phone-scale','phone-x','phone-y','reset-alignment'])$(id).disabled=locked;
 $('hybrid-mode').checked=hybridWanted;$('robot-opacity').value=phoneAlignment.opacity*100;$('opacity-value').textContent=Math.round(phoneAlignment.opacity*100)+'%';$('phone-scale').value=phoneAlignment.scale*100;$('phone-x').value=phoneAlignment.x*100;$('phone-y').value=phoneAlignment.y*100;
 $('phone-status').textContent=phoneProblem;
}
function leaveHybrid(reason){
 const selected=hybridWanted;hybridWanted=false;hybridSuspended=false;hybridRefreshUntil=null;hybridMotionHeld=false;phoneGuard.invalidate();phoneSample=null;const old=phoneTexture;phoneTexture=null;realBackground.material.map=null;if(old){old.image?.close?.();old.dispose();}phoneProblem=reason;layout();syncHybridControls();
 if(selected){releaseLocal(reason+' Release Y, then Enable again.');if(ownerSession)lossPacket();}
}
function checkHybrid(now){
 if(!hybridWanted)return;const display=hybridDisplayState(now);
 if(display.drop){leaveHybrid('Live background expired. Simulation-only fallback.');return;}
 realBackground.visible=display.background;nativeOverlay.visible=display.overlay;viewMeshes.main.visible=!physical()&&!display.background;
 if(display.hold){if(!hybridMotionHeld||!localHold||enablePending){hybridMotionHeld=true;releaseLocal('Live view refreshing / native overlay unavailable. Motion held; fresh Y required.');if(ownerSession)lossPacket();}}
 else hybridMotionHeld=false;
}
async function pollPhone(){
 if(phoneBusy)return;phoneBusy=true;const token=phoneGuard.begin(),sent=performance.now();
 try{
  const value=await api('/api/phone'),received=performance.now(),rtt=received-sent;if(token!==phoneGuard.token)return;
  if(!phoneFresh(value,received,rtt,received)){leaveHybrid('Live background unavailable or expired. Simulation only.');return;}
  if(phoneSample&&value.generation!==phoneSample.generation&&hybridWanted)leaveHybrid('Camera source changed. Simulation-only fallback.');
  if(phoneSample?.generation===value.generation&&phoneSample?.sequence===value.sequence)return;
  const response=await fetch(value.jpeg),bitmap=await createImageBitmap(await response.blob(),{imageOrientation:'flipY',premultiplyAlpha:'none'});
  if(!phoneGuard.accept(token,value,received,rtt,performance.now())){bitmap.close();return;}
  const texture=new THREE.Texture(bitmap);texture.flipY=false;texture.colorSpace=THREE.SRGBColorSpace;texture.needsUpdate=true;const old=phoneTexture;
  phoneTexture=texture;realBackground.material.map=texture;realBackground.material.needsUpdate=true;phoneSample=value;hybridRefreshUntil=null;phoneReceivedMs=received;phoneRoundtripMs=rtt;
  if(old){old.image?.close?.();old.dispose();}
  const mismatch=Math.abs((value.width/value.height)/config.camera.aspect-1)>.02;
  phoneProblem='Live camera '+value.width+'×'+value.height+' · sensor age/lens unknown'+(mismatch?' · aspect differs; image fitted without stretch':'')+' · asynchronous visual rehearsal';layout();syncHybridControls();
 }catch(e){if(token===phoneGuard.token)leaveHybrid('Live camera link lost. Simulation-only fallback.');}
 finally{phoneBusy=false;}
}
$('hybrid-mode').onchange=()=>{
 if(panelsLocked())return syncHybridControls();
 if(!$('hybrid-mode').checked){hybridWanted=false;layout();syncHybridControls();return;}
 if(!phoneTexture||!phoneFresh(phoneSample,phoneReceivedMs,phoneRoundtripMs,performance.now())){phoneProblem='Fresh live camera must be ready before selection.';syncHybridControls();return;}
 hybridWanted=true;hybridSuspended=false;hybridRefreshUntil=null;checkHybrid(performance.now());layout();syncState();message(physical()?'Actual iPhone view selected. Robot overlay and modeled wrist views stay hidden. Fresh Y is required.':'Live background with translucent simulated arms. Adjust while Disabled. Real obstacles are not detected.');
};
for(const [id,key,factor]of [['robot-opacity','opacity',.01],['phone-scale','scale',.01],['phone-x','x',.01],['phone-y','y',.01]])$(id).oninput=()=>{phoneAlignment=editAlignment(phoneAlignment,{[key]:Number($(id).value)*factor},panelsLocked());try{localStorage.setItem('so101-phone-alignment-v1',JSON.stringify(phoneAlignment));}catch{}layout();syncHybridControls();};
$('reset-alignment').onclick=()=>{if(panelsLocked())return;phoneAlignment=defaultAlignment();try{localStorage.removeItem('so101-phone-alignment-v1');}catch{}layout();syncHybridControls();};
try{const sent=performance.now();config=await api('/api/config');observeOwner(config);if(Number.isFinite(config.server_ms))viewOffsetUpperMs=config.server_ms-sent;state.tools=config.tools;setDesktopReference();poll(performance.now());pollViews();$('guide').textContent='Close-up stereo 3D workspace. Enter VR, press Y, then hold a grip to move its arm. Lean to inspect the parts; X recenters.';}catch(e){message('Cannot load simulation config: '+e.message);$('connection').textContent='Simulation unavailable';}
