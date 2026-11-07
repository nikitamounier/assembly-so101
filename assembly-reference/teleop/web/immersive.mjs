import * as THREE from '/vendor/three.module.js';

function unpack(text,Type){const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));return new Type(bytes.buffer);}
const quaternion=q=>new THREE.Quaternion(q[1],q[2],q[3],q[0]);

export class ImmersiveScene{
 constructor(scene){
  this.root=new THREE.Group();this.root.visible=false;scene.add(this.root);
  scene.add(new THREE.HemisphereLight(0xe7f3ff,0x535360,2.1));
  const light=new THREE.DirectionalLight(0xfff3df,2.3);light.position.set(-1,3,1);scene.add(light);
  this.reset();
 }
 reset(){
  const geometries=new Set(),materials=new Set();
  this.root.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});
  for(const g of geometries)g.dispose();for(const m of materials)m.dispose();
  this.root.clear();this.root.visible=false;this.owner=null;this.bodies=[];this.target=null;this.lastServerMs=-Infinity;this.lastFrame=null;this.ready=false;this.sceneGeneration=null;
 }
 load(value){
  this.reset();this.owner=value.owner_id;
  for(let i=0;i<value.body_count;i++){const body=new THREE.Group();this.root.add(body);this.bodies.push(body);}
  const meshes={};
  for(const [key,m]of Object.entries(value.meshes)){
   const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(unpack(m.position,Float32Array),3));
   geometry.setIndex(new THREE.BufferAttribute(unpack(m.index,Uint32Array),1));geometry.computeVertexNormals();geometry.computeBoundingSphere();meshes[key]=geometry;
  }
  this.triangles=0;
  for(const g of value.geoms){
   let geometry;
   if(g.mesh!==null)geometry=meshes[g.mesh];
   else if(g.type==='box')geometry=new THREE.BoxGeometry(...g.size.map(x=>2*x));
   else if(g.type==='plane')geometry=new THREE.PlaneGeometry(2*g.size[0],2*g.size[1]);
   else if(g.type==='cylinder'){geometry=new THREE.CylinderGeometry(g.size[0],g.size[0],2*g.size[1],24);geometry.rotateX(Math.PI/2);}
   else if(g.type==='capsule'){geometry=new THREE.CapsuleGeometry(g.size[0],2*g.size[1],4,12);geometry.rotateX(Math.PI/2);}
   else if(g.type==='sphere')geometry=new THREE.SphereGeometry(g.size[0],16,12);
   else throw Error('Unsupported native visual: '+g.type);
   const color=new THREE.Color(...g.rgba.slice(0,3));
   const material=g.surface?new THREE.MeshPhongMaterial({color,specular:new THREE.Color().setScalar(g.surface.specular),shininess:128*g.surface.shininess,side:THREE.DoubleSide}):new THREE.MeshStandardMaterial({color,roughness:.65,metalness:g.name?.includes('screw')?.5:.12,side:THREE.DoubleSide});
   const mesh=new THREE.Mesh(geometry,material);mesh.name=g.name||'native-'+g.id;mesh.position.fromArray(g.position);mesh.quaternion.copy(quaternion(g.quaternion));
   this.bodies[g.body].add(mesh);this.triangles+=(geometry.index?.count??geometry.attributes.position.count)/3;
  }
  // A subtle metric table grid supplies depth cues without covering the parts.
  const grid=new THREE.GridHelper(1,20,0x536571,0x35434e);grid.rotation.x=Math.PI/2;grid.position.z=.0002;this.bodies[0].add(grid);
 }
 align(calibration){
  if(!calibration)return;
  const r=calibration.rotation;
  this.root.position.fromArray(calibration.translation);
  this.root.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1));
 }
 accept(state){
  if(state?.owner_id!==this.owner||!Number.isFinite(state.server_ms)||state.server_ms<=this.lastServerMs)return false;
  if(state.body_poses?.length!==this.bodies.length||!state.body_poses.every(p=>p.length===7&&p.every(Number.isFinite)))return false;
  this.target=state.body_poses.map(p=>({position:new THREE.Vector3(...p.slice(0,3)),quaternion:quaternion(p.slice(3))}));
  if(!this.ready||state.scene_generation!==this.sceneGeneration)for(let i=0;i<this.bodies.length;i++){this.bodies[i].position.copy(this.target[i].position);this.bodies[i].quaternion.copy(this.target[i].quaternion);}
  this.sceneGeneration=state.scene_generation;this.lastServerMs=state.server_ms;this.ready=true;this.root.visible=true;return true;
 }
 frame(now){
  const dt=this.lastFrame===null?0:Math.max(0,(now-this.lastFrame)/1000);this.lastFrame=now;
  if(!this.target)return;const alpha=1-Math.exp(-Math.min(dt,.1)/.025);
  for(let i=0;i<this.bodies.length;i++){this.bodies[i].position.lerp(this.target[i].position,alpha);this.bodies[i].quaternion.slerp(this.target[i].quaternion,alpha);}
 }
 snapshot(){return {ready:this.ready,owner_id:this.owner,bodies:this.bodies.length,triangles:this.triangles??0,server_ms:Number.isFinite(this.lastServerMs)?this.lastServerMs:null,visible:this.root.visible};}
}
