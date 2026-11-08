import * as THREE from "three";

/** Exact prototype collision primitives; instanced to keep detailed shells fast. */
export class CollisionBoxes {
  constructor(scene, objects, payload) {
    const anchors = new Map(objects.map((object) => [object.name, object]));
    if (payload.schema !== 1 || !payload.boxes?.length || payload.boxes.some((box) => !anchors.has(box.anchor))) {
      throw new Error("Collision geometry does not match this scene");
    }
    this.root = new THREE.Group();
    this.root.name = "bare-collision-geometry";
    this.resources = [];
    this.meshes = [];
    this.entries = [];
    this.fills = [];
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    this.resources.push(geometry);
    for (const [role, color] of Object.entries({ robot: 0x0084dc, finger: 0x00a56a, part: 0xf27718 })) {
      const boxes = payload.boxes.filter((box) => box.role === role);
      if (!boxes.length) continue;
      const fill = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.09, depthWrite: false });
      const wire = new THREE.MeshBasicMaterial({ color, wireframe: true, depthTest: false, transparent: true, opacity: 0.6 });
      this.resources.push(fill, wire);
      this.fills.push(fill);
      const solidMesh = new THREE.InstancedMesh(geometry, fill, boxes.length);
      const wireMesh = new THREE.InstancedMesh(geometry, wire, boxes.length);
      wireMesh.renderOrder = 8;
      solidMesh.frustumCulled = wireMesh.frustumCulled = false;
      this.meshes.push(solidMesh, wireMesh);
      this.root.add(solidMesh, wireMesh);
      boxes.forEach((box, index) => {
        if (box.half_size.length !== 3 || box.half_size.some((v) => !Number.isFinite(v) || v <= 0)) throw new Error("Invalid collision primitive");
        const [w, x, y, z] = box.quaternion;
        const local = new THREE.Matrix4().compose(
          new THREE.Vector3().fromArray(box.position),
          new THREE.Quaternion(x, y, z, w),
          new THREE.Vector3(...box.half_size.map((v) => 2 * v)),
        );
        this.entries.push({ anchor: anchors.get(box.anchor), local, index, solidMesh, wireMesh });
      });
    }
    scene.add(this.root);
    this.update();
  }

  update() {
    const transform = new THREE.Matrix4();
    const unit = new THREE.Vector3(1, 1, 1);
    for (const { anchor, local, index, solidMesh, wireMesh } of this.entries) {
      transform.compose(anchor.position, anchor.quaternion, unit).multiply(local);
      solidMesh.setMatrixAt(index, transform);
      wireMesh.setMatrixAt(index, transform);
    }
    for (const mesh of this.meshes) mesh.instanceMatrix.needsUpdate = true;
  }

  setVisible(visible) { this.root.visible = visible; }

  setCadVisible(visible) {
    for (const fill of this.fills) fill.opacity = visible ? 0.09 : 0.7;
  }

  dispose() {
    this.root.removeFromParent();
    for (const mesh of this.meshes) mesh.dispose();
    for (const resource of this.resources) resource.dispose();
  }
}
