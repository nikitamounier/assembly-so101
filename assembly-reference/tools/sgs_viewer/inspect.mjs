import * as THREE from './vendor/three.module.js';
import {ImmersiveScene} from './native-scene.mjs';

export class SnapshotInspector{
 constructor(container){
  this.container=container;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#e5ebe6');
  this.camera=new THREE.PerspectiveCamera(45,1,.001,20);this.camera.up.set(0,0,1);
  this.renderer=new THREE.WebGLRenderer({antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.outputColorSpace=THREE.SRGBColorSpace;container.append(this.renderer.domElement);
  this.native=new ImmersiveScene(this.scene);this.target=new THREE.Vector3(0,.09,.11);this.distance=.9;this.azimuth=135*Math.PI/180;this.elevation=22*Math.PI/180;this.sequence=0;this.drag=null;
  container.addEventListener('contextmenu',e=>e.preventDefault());
  container.addEventListener('pointerdown',e=>{this.drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};container.setPointerCapture(e.pointerId);});
  container.addEventListener('pointerup',()=>this.drag=null);container.addEventListener('pointercancel',()=>this.drag=null);
  container.addEventListener('pointermove',e=>{if(!this.drag)return;const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;this.drag.x=e.clientX;this.drag.y=e.clientY;if(this.drag.pan){const right=new THREE.Vector3().setFromMatrixColumn(this.camera.matrix,0),up=new THREE.Vector3().setFromMatrixColumn(this.camera.matrix,1);this.target.addScaledVector(right,-dx*this.distance*.0018).addScaledVector(up,dy*this.distance*.0018);}else{this.azimuth-=dx*.008;this.elevation=Math.max(-1.5,Math.min(1.55,this.elevation+dy*.008));}this.draw();});
  container.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.min(3,Math.max(.025,this.distance*Math.exp(e.deltaY*.001)));this.draw();},{passive:false});
  this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(container);
 }
 async load(path){
  const response=await fetch(path);if(!response.ok)throw Error(`Geometry: HTTP ${response.status}`);
  const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
  const geometry=JSON.parse(await new Response(stream).text());this.owner=geometry.owner_id;this.native.load(geometry);
 }
 show(detail){
  this.detail=detail;
  if(!this.native.accept({owner_id:this.owner,server_ms:++this.sequence,scene_generation:detail.id,body_poses:detail.body_poses}))throw Error('Native pose snapshot rejected');
  // A new scene_generation copies poses immediately. Do not call frame(), which
  // interpolates live teleop updates: unrelated SGS snapshots are not trajectories.
  this.draw();
 }
 focus(){if(!this.detail)return;this.target.fromArray(this.detail.body_poses[this.detail.active_body].slice(0,3));this.distance=.24;this.elevation=35*Math.PI/180;this.draw();}
 reset(){this.target.set(0,.09,.11);this.distance=.9;this.azimuth=135*Math.PI/180;this.elevation=22*Math.PI/180;this.draw();}
 draw(){const w=this.container.clientWidth,h=this.container.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();const r=this.distance,c=Math.cos(this.elevation);this.camera.position.set(this.target.x+r*c*Math.cos(this.azimuth),this.target.y+r*c*Math.sin(this.azimuth),this.target.z+r*Math.sin(this.elevation));this.camera.lookAt(this.target);this.renderer.render(this.scene,this.camera);}
}
