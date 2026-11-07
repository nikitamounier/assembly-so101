import * as THREE from "three";
import { RewardOverlay } from "./reward.js";
import { pointCloudWorldPositions } from "./point-cloud-data.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";


const Z = new THREE.Vector3(0, 0, 1);

export class ThreeSceneRenderer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.rewardOverlay = new RewardOverlay(this.scene);
    this.scene.background = new THREE.Color("#e8eceb");
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.001, 20);
    this.camera.up.copy(Z);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 3D assembly reset",
    );
    container.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.025;
    this.controls.maxDistance = 3;
    this.controls.maxPolarAngle = Math.PI * 0.51;
    this.controls.addEventListener("start", () => {
      this.activePreset = null;
    });
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x627577, 2.4));
    const light = new THREE.DirectionalLight(0xffffff, 3.2);
    light.position.set(-0.5, -0.5, 1.5);
    this.scene.add(light);
    const fill = new THREE.DirectionalLight(0xe0eaff, 1.2);
    fill.position.set(1, 0.5, 0.8);
    this.scene.add(fill);
    const grid = new THREE.GridHelper(1.2, 40, 0xaab5b4, 0xc8d0cf);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.0003;
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
    this.scene.add(grid);
    this.geometryVisible = this.pointsVisible = true;
    this.objects = [];
    this.resources = new Set();
    this.resize = new ResizeObserver(() => this.resizeView());
    this.resize.observe(container);
    this.preset("perspective");
    this.renderer.setAnimationLoop(() => {
      if (!this.container.hidden) {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      }
    });
  }

  resizeView() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  setScene(scene) {
    this.cubeScene =
      scene.geoms.some((geom) => geom.name === "cube_geom") &&
      scene.geoms.some((geom) => geom.name.startsWith("nominal_path_"));
    this.fittedCubeScene = false;
    for (const object of this.objects) this.scene.remove(object);
    for (const resource of this.resources) resource.dispose();
    this.resources.clear();
    const meshes = new Map();
    for (const [key, mesh] of Object.entries(scene.meshes)) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(mesh.vertices, 3),
      );
      geometry.setIndex(mesh.faces);
      geometry.computeVertexNormals();
      meshes.set(key, geometry);
      this.resources.add(geometry);
    }
    this.objects = scene.geoms.map((geom) => {
      const [x, y, z] = geom.size;
      let geometry;
      switch (geom.type) {
        case 0:
          geometry = new THREE.PlaneGeometry(1.3, 1);
          break;
        case 2:
          geometry = new THREE.SphereGeometry(x, 24, 16);
          break;
        case 3:
          geometry = new THREE.CapsuleGeometry(x, 2 * y, 8, 24);
          geometry.rotateX(Math.PI / 2);
          break;
        case 4:
          geometry = new THREE.SphereGeometry(1, 24, 16);
          geometry.scale(x, y, z);
          break;
        case 5:
          geometry = new THREE.CylinderGeometry(x, x, 2 * y, 32);
          geometry.rotateX(Math.PI / 2);
          break;
        case 6:
          geometry = new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
          break;
        case 7:
          geometry = meshes.get(geom.mesh);
          break;
        default:
          throw new Error(`Unsupported visual geometry: ${geom.type}`);
      }
      this.resources.add(geometry);
      const [r, g, b, a] = geom.rgba;
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        metalness: 0.16,
        roughness: 0.62,
        transparent: a < 1,
        opacity: a,
        side: THREE.DoubleSide,
      });
      this.resources.add(material);
      const object = new THREE.Mesh(geometry, material);
      object.name = geom.name;
      object.visible = this.geometryVisible;
      this.scene.add(object);
      return object;
    });
  }

  setSnapshot(snapshot) {
    this.setPointCloud(snapshot.point_cloud);
    this.rewardOverlay.setSnapshot(snapshot.reward_region);
    snapshot.poses.forEach(([x, y, z, w, qx, qy, qz], index) => {
      this.objects[index].position.set(x, y, z);
      this.objects[index].quaternion.set(qx, qy, qz, w);
    });
    if (snapshot.focus) {
      this.reviewFocus = snapshot.focus;
      this.preset("detail");
    }
    if (this.cubeScene) {
      this.cubePosition = this.objects
        .find((object) => object.name === "cube_geom")
        .position.clone();
      this.physicalBounds = new THREE.Box3();
      for (const object of this.objects) {
        if (object.geometry.type === "PlaneGeometry") continue;
        object.updateMatrixWorld(true);
        if (!object.geometry.boundingBox) object.geometry.computeBoundingBox();
        this.physicalBounds.union(
          object.geometry.boundingBox.clone().applyMatrix4(object.matrixWorld),
        );
      }
      if (!this.fittedCubeScene || this.activePreset === "detail") {
        this.preset(this.activePreset || "perspective");
        this.fittedCubeScene = true;
      }
    }
  }

  clearPointCloud() {
    if (!this.pointCloud) return;
    this.scene.remove(this.pointCloud);
    this.pointCloud.geometry.dispose();
    this.pointCloud.material.dispose();
    this.pointCloud = null;
  }

  setPointCloud(cloud) {
    this.clearPointCloud();
    if (!cloud?.available) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pointCloudWorldPositions(cloud), 3));
    const colors = new Float32Array(cloud.count * 3);
    for (const group of cloud.groups) {
      const color = new THREE.Color(group.color);
      for (let i = group.start; i < group.start + group.count; i++) color.toArray(colors, i * 3);
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true, depthTest: false, depthWrite: false, toneMapped: false }));
    this.pointCloud.name = "policy-input-points";
    this.pointCloud.renderOrder = 10;
    this.pointCloud.visible = this.pointsVisible;
    this.scene.add(this.pointCloud);
  }

  setLayers({ geometry, points }) {
    this.geometryVisible = geometry;
    this.pointsVisible = points;
    for (const object of this.objects) object.visible = geometry;
    if (this.pointCloud) this.pointCloud.visible = points;
  }

  preset(name) {
    this.activePreset = name;
    if (name === "detail" && this.rewardOverlay.region?.available) {
      const box = this.rewardOverlay.region.target_box;
      const target = new THREE.Vector3().fromArray(box.center);
      const radius = new THREE.Vector3().fromArray(box.size).length() / 2;
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.6 / Math.min(1, this.camera.aspect);
      this.camera.position.copy(target).add(new THREE.Vector3(1, -1, 1).normalize().multiplyScalar(distance));
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    if (this.reviewFocus) {
      const target = new THREE.Vector3().fromArray(this.reviewFocus.target);
      const direction = name === "top" ? [0, -.001, 1] : name === "front" ? [0, -1, .12] : name === "perspective" ? [1, -1, 1] : [1, -1, 1];
      const offset = new THREE.Vector3(...direction).normalize().multiplyScalar(this.reviewFocus.radius * 3);
      this.camera.position.copy(target).add(offset);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    if (
      this.cubeScene &&
      this.physicalBounds &&
      !this.physicalBounds.isEmpty()
    ) {
      const center = this.physicalBounds.getCenter(new THREE.Vector3());
      const size = this.physicalBounds.getSize(new THREE.Vector3());
      const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
      let target = center;
      let offset;
      if (name === "detail") {
        target = this.cubePosition;
        offset = new THREE.Vector3(0.24, -0.4, 0.24);
      } else if (name === "front") {
        const distance =
          1.15 *
            Math.max(
              size.x / (2 * Math.tan(halfFov) * this.camera.aspect),
              size.z / (2 * Math.tan(halfFov)),
            ) +
          size.y / 2;
        offset = new THREE.Vector3(0, -distance, 0);
      } else if (name === "top") {
        const distance =
          1.15 *
            Math.max(
              size.x / (2 * Math.tan(halfFov) * this.camera.aspect),
              size.y / (2 * Math.tan(halfFov)),
            ) +
          size.z / 2;
        offset = new THREE.Vector3(0, -0.001, distance);
      } else {
        const distance = (size.length() / (2 * Math.sin(halfFov))) * 1.1;
        offset = new THREE.Vector3(0.68, -0.88, 0.6)
          .normalize()
          .multiplyScalar(distance);
      }
      this.camera.position.copy(target).add(offset);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    const presets = {
      perspective: [
        [0.8, -0.775, 0.92],
        [0, 0.025, 0.12],
      ],
      top: [
        [0, 0.02, 1.18],
        [0, 0.025, 0],
      ],
      front: [
        [0, -1.05, 0.35],
        [0, 0.025, 0.12],
      ],
      detail: [
        [0.24, -0.33, 0.32],
        [0, 0.025, 0.035],
      ],
    };
    const [position, target] = presets[name] || presets.perspective;
    this.camera.position.fromArray(position);
    this.controls.target.fromArray(target);
    this.controls.update();
  }

  dispose() {
    this.clearPointCloud();
    this.rewardOverlay.clear();
    this.resize.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    for (const resource of this.resources) resource.dispose();
    this.renderer.dispose();
    this.container.replaceChildren();
  }
}

export class MujocoSceneRenderer {
  constructor(container) {
    this.frame = document.createElement("iframe");
    this.frame.title = "Official MuJoCo WebViewer · selected reset";
    this.frame.allow = "cross-origin-isolated";
    this.frame.hidden = true;
    container.append(this.frame);
  }

  async activate() {
    if (!this.frame.src) {
      const response = await fetch("/api/studio", { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      this.frame.src = (await response.json()).url;
    }
    this.frame.hidden = false;
  }

  hide() {
    this.frame.hidden = true;
  }
  dispose() {
    this.frame.remove();
  }
}

/** Reusable viewport: one scene/state provider, interchangeable renderers. */
export class SceneView {
  constructor(container) {
    this.container = container;
    this.threeHost = document.createElement("div");
    this.threeHost.className = "three-host";
    container.append(this.threeHost);
    this.three = new ThreeSceneRenderer(this.threeHost);
    this.mujoco = new MujocoSceneRenderer(container);
    this.mode = "three";
  }

  setScene(scene) {
    this.three.setScene(scene);
  }
  setSnapshot(snapshot) {
    this.three.setSnapshot(snapshot);
    this.threeHost.hidden = this.mode !== "three";
    if (this.mode === "mujoco") this.mujoco.frame.hidden = false;
  }
  setRewardVisible(visible) { this.three.rewardOverlay.setVisible(visible); }
  setLayers(layers) {
    this.three.setLayers(layers);
  }
  clear() {
    this.three.clearPointCloud();
    this.three.rewardOverlay.clear();
    this.threeHost.hidden = true;
    this.mujoco.hide();
  }
  preset(name) {
    this.three.preset(name);
  }
  async setMode(mode) {
    if (mode === "mujoco") await this.mujoco.activate();
    else this.mujoco.hide();
    this.threeHost.hidden = mode !== "three";
    this.mode = mode;
    if (mode === "three") this.three.resizeView();
  }
  dispose() {
    this.three.dispose();
    this.mujoco.dispose();
  }
}
