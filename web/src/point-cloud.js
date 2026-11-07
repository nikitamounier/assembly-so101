import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/** Draw the API's complete ordered float32 tensor without sampling or normalization. */
export class PointCloudSection {
  constructor() {
    this.host = document.querySelector("#point-cloud-view");
    this.status = document.querySelector("#point-cloud-status");
    this.meta = document.querySelector("#point-cloud-meta");
    this.empty = document.querySelector("#point-cloud-empty");
    this.legend = document.querySelector("#point-cloud-legend");
    this.fitButton = document.querySelector("#point-cloud-fit");
    this.downloadButton = document.querySelector("#point-cloud-download");
    this.fitButton.addEventListener("click", () => this.fit());
    this.downloadButton.addEventListener("click", () => this.download());
  }

  initialize() {
    if (this.renderer) return;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#edf1f0");
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.setAttribute("aria-label", "All policy input points in the left fixed-finger frame");
    this.host.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.resize = new ResizeObserver(() => this.resizeView());
    this.resize.observe(this.host);
    this.renderer.setAnimationLoop(() => {
      if (!this.host.hidden) {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      }
    });
  }

  resizeView() {
    const { clientWidth: width, clientHeight: height } = this.host;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  clear(message = "No assembly state selected.") {
    this.cloud = null;
    this.host.hidden = true;
    this.empty.hidden = false;
    this.empty.textContent = message;
    this.status.textContent = "No policy point cloud";
    this.meta.textContent = "";
    this.legend.replaceChildren();
    this.fitButton.disabled = this.downloadButton.disabled = true;
  }

  setSnapshot(snapshot) {
    const cloud = snapshot.point_cloud;
    if (!cloud?.available) {
      this.clear(cloud?.message || "No policy observation configured for this scene.");
      return;
    }
    this.cloud = cloud;
    this.host.hidden = false;
    this.empty.hidden = true;
    this.status.textContent = cloud.status;
    this.status.classList.toggle("preview", cloud.preview);
    this.meta.textContent = `${cloud.count} points · ${cloud.shape.join(" × ")} · ${cloud.dtype} · ${cloud.frame} frame · ${cloud.units}`;
    this.meta.title = `Observation: ${cloud.observation_signature}\nTensor: ${cloud.tensor_sha256}\nState: ${cloud.state_id}`;
    this.initialize();
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(cloud.flat, 3));
    const colors = new Float32Array(cloud.count * 3);
    for (const group of cloud.groups) {
      const color = new THREE.Color(group.color);
      for (let i = group.start; i < group.start + group.count; i++) color.toArray(colors, i * 3);
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true }));
    this.scene.add(this.points);
    this.legend.replaceChildren(...cloud.groups.map((group) => {
      const label = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = group.color;
      label.append(swatch, `${group.name} · ${group.count}`);
      return label;
    }));
    this.fitButton.disabled = this.downloadButton.disabled = false;
    this.resizeView();
    this.fit();
  }

  fit() {
    if (!this.points) return;
    this.points.geometry.computeBoundingSphere();
    const { center, radius } = this.points.geometry.boundingSphere;
    const distance = Math.max(0.05, radius) / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.25 / Math.min(1, this.camera.aspect);
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1, -1.4, 0.8).normalize().multiplyScalar(distance));
    this.controls.minDistance = Math.max(radius * 0.02, 0.001);
    this.controls.maxDistance = distance * 8;
    this.controls.update();
  }

  download() {
    if (!this.cloud) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(this.cloud, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `point-cloud-${this.cloud.state_id}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
