import * as THREE from "three";

export class RewardOverlay {
  constructor(scene) { this.scene = scene; this.visible = false; }
  clear() {
    if (!this.group) return;
    this.scene.remove(this.group);
    this.group.traverse((object) => { object.geometry?.dispose(); object.material?.dispose(); });
    this.group = null;
  }
  setSnapshot(region) {
    this.clear();
    this.region = region;
    if (!region?.available) return;
    this.group = new THREE.Group();
    this.group.name = "reward-region";
    for (const [box, color, opacity] of [[region.target_box, 0xd99b25, .035], [region.noise_box, 0x22b876, .3]]) {
      const geometry = new THREE.BoxGeometry(...box.size);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({color, depthTest: false, transparent: true, opacity: 1}));
      const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color, transparent: true, opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide}));
      const r = box.rotation_row_major;
      const rotation = new THREE.Matrix4().set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
      for (const object of [edges, fill]) {
        object.position.fromArray(box.center);
        object.quaternion.setFromRotationMatrix(rotation);
        object.renderOrder = 20;
        this.group.add(object);
      }
    }
    this.group.visible = this.visible;
    this.scene.add(this.group);
  }
  setVisible(visible) { this.visible = visible; if (this.group) this.group.visible = visible; }
}

export class RewardSection {
  constructor(view, onTolerance) {
    this.view = view;
    this.button = document.querySelector("#toggle-reward");
    this.status = document.querySelector("#reward-status");
    this.position = document.querySelector("#reward-position");
    this.angle = document.querySelector("#reward-angle");
    this.visible = false;
    this.button.onclick = () => {
      this.visible = !this.visible;
      this.button.setAttribute("aria-pressed", String(this.visible));
      this.view.setRewardVisible(this.visible);
    };
    this.apply = document.querySelector("#reward-apply");
    this.apply.onclick = async () => {
      if (!this.position.reportValidity() || !this.angle.reportValidity()) return;
      this.position.disabled = this.angle.disabled = true;
      try { await onTolerance({position_mm: Number(this.position.value), orientation_degrees: Number(this.angle.value)}); }
      catch (error) { this.status.textContent = error.message; }
      finally { this.position.disabled = this.angle.disabled = false; }
    };
  }
  clear() {
    this.button.disabled = this.apply.disabled = true;
    this.position.disabled = this.angle.disabled = true;
    this.status.textContent = "Select an assembly task to see its seating target.";
  }
  setSnapshot(snapshot) {
    const region = snapshot.reward_region;
    if (!region?.available) { this.clear(); return; }
    this.button.disabled = this.apply.disabled = this.position.disabled = this.angle.disabled = false;
    this.position.value = region.position_tolerance_mm;
    this.angle.value = region.orientation_tolerance_degrees;
    const active = region.checks.at(-1);
    const error = Math.max(...active.position_error_m.map(Math.abs)) * 1000;
    this.status.textContent = `Seating reward: ${region.reward} / 1 · position error ${error.toFixed(2)} mm · angle ${active.orientation_error_degrees.toFixed(2)}° · includes preceding insertions`;
    this.status.title = region.definition;
  }
}
