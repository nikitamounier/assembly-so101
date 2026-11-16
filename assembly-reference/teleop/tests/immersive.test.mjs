import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as THREE from '../web/vendor/three.module.js';
const source=readFileSync(new URL('../web/immersive.mjs',import.meta.url),'utf8').replace("'/vendor/three.module.js'",JSON.stringify(new URL('../web/vendor/three.module.js',import.meta.url).href));
const {ImmersiveScene}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const assets=owner=>({owner_id:owner,body_count:2,meshes:{},geoms:[{id:1,name:'motor',type:'box',mesh:null,body:1,position:[.01,0,0],quaternion:[1,0,0,0],size:[.02,.02,.02],rgba:[.5,.5,.5,1]}]});
const state=(owner,ms,x=.1)=>({owner_id:owner,server_ms:ms,body_poses:[[0,0,0,1,0,0,0],[x,.2,.3,1,0,0,0]]});

test('HOTO native gloss reaches the headset while legacy materials remain supported',()=>{
 const input=assets('material');input.geoms[0].surface={specular:.85,shininess:.65};
 const view=new ImmersiveScene(new THREE.Scene());view.load(input);
 const material=view.bodies[1].children[0].material;
 assert.equal(material.type,'MeshPhongMaterial');assert.equal(material.shininess,83.2);
 assert.ok(Math.abs(material.specular.r-.85)<1e-12);
 view.load(assets('legacy'));assert.equal(view.bodies[1].children[0].material.type,'MeshStandardMaterial');
});

test('native body and geom transforms remain metric in the world under stereo/head motion',()=>{
 const scene=new THREE.Scene(),view=new ImmersiveScene(scene);view.load(assets('first'));
 view.align({rotation:[0,-1,0,0,0,1,-1,0,0],translation:[1,2,3]});assert.ok(view.accept(state('first',100)));
 scene.updateMatrixWorld(true);const point=new THREE.Vector3();view.bodies[1].children[0].getWorldPosition(point);
 assert.ok(point.distanceTo(new THREE.Vector3(.8,2.3,2.89))<1e-12);
 const camera=new THREE.PerspectiveCamera(70,1,.01,20);camera.position.set(.8,2.3,3.4);camera.updateMatrixWorld();
 const first=point.clone().project(camera);camera.position.x+=.064;camera.updateMatrixWorld();const second=point.clone().project(camera);
 assert.notEqual(first.x,second.x);view.bodies[1].children[0].getWorldPosition(point);assert.ok(point.distanceTo(new THREE.Vector3(.8,2.3,2.89))<1e-12);
 assert.equal(view.root.parent,scene);assert.equal(view.ready,true);
});

test('geometry rejects old owners, regressing poses and malformed transforms, then resets on restart',()=>{
 const view=new ImmersiveScene(new THREE.Scene());view.load(assets('first'));assert.ok(view.accept(state('first',100)));
 assert.equal(view.accept(state('old',200)),false);assert.equal(view.accept(state('first',90)),false);
 const bad=state('first',200);bad.body_poses[1][0]=NaN;assert.equal(view.accept(bad),false);
 view.load(assets('second'));assert.equal(view.ready,false);assert.equal(view.root.visible,false);
 assert.equal(view.accept(state('first',300)),false);assert.ok(view.accept(state('second',1)));
});

test('display interpolation approaches measured poses without extrapolating them',()=>{
 const view=new ImmersiveScene(new THREE.Scene());view.load(assets('first'));view.accept(state('first',1));view.frame(0);
 view.accept(state('first',2,.2));view.frame(20);assert.ok(view.bodies[1].position.x>.1&&view.bodies[1].position.x<.2);
 for(let t=40;t<=500;t+=20)view.frame(t);assert.ok(Math.abs(view.bodies[1].position.x-.2)<1e-8);
});


test('explicit task reset snaps measured geometry instead of interpolating an assembly',()=>{
 const view=new ImmersiveScene(new THREE.Scene());view.load(assets('first'));
 view.accept({...state('first',1),scene_generation:0});view.frame(0);
 view.accept({...state('first',2,.4),scene_generation:1});
 assert.equal(view.bodies[1].position.x,.4);
});
