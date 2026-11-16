/** Point-cloud metadata and independent layers in the shared scene viewport. */
export class PointCloudSection {
  constructor(view) {
    this.view = view;
    this.status = document.querySelector("#point-cloud-status");
    this.meta = document.querySelector("#point-cloud-meta");
    this.empty = document.querySelector("#point-cloud-empty");
    this.legend = document.querySelector("#point-cloud-legend");
    this.fitButton = document.querySelector("#point-cloud-fit");
    this.downloadButton = document.querySelector("#point-cloud-download");
    this.geometryButton = document.querySelector("#toggle-geometry");
    this.pointsButton = document.querySelector("#toggle-points");
    this.geometryVisible = this.pointsVisible = true;
    this.fitButton.addEventListener("click", () => this.view.preset("perspective"));
    this.downloadButton.addEventListener("click", () => this.download());
    this.geometryButton.addEventListener("click", () => {
      this.geometryVisible = !this.geometryVisible;
      this.updateLayers();
    });
    this.pointsButton.addEventListener("click", () => {
      this.pointsVisible = !this.pointsVisible;
      this.updateLayers();
    });
  }

  updateLayers() {
    this.geometryButton.setAttribute("aria-pressed", String(this.geometryVisible));
    this.pointsButton.setAttribute("aria-pressed", String(this.pointsVisible && !!this.cloud));
    this.pointsButton.disabled = !this.cloud;
    this.view.setLayers({ geometry: this.geometryVisible, points: this.pointsVisible });
  }

  clear(message = "No assembly state selected.") {
    this.cloud = null;
    this.empty.hidden = false;
    this.empty.textContent = message;
    this.status.textContent = "No policy point cloud";
    this.meta.textContent = "";
    this.legend.replaceChildren();
    this.fitButton.disabled = this.downloadButton.disabled = true;
    this.updateLayers();
  }

  setSnapshot(snapshot) {
    const cloud = snapshot.point_cloud;
    if (!cloud?.available) {
      this.clear(cloud?.message || "No policy observation configured for this scene.");
      return;
    }
    this.cloud = cloud;
    this.empty.hidden = true;
    this.status.textContent = cloud.status;
    this.status.classList.toggle("preview", cloud.preview);
    const budget = cloud.sampling_budget;
    this.meta.textContent = `${cloud.count} points · ${cloud.shape.join(" × ")} · ${cloud.dtype} · World-aligned overlay${budget ? ` · Receiver: ${budget.feature} feature / ${budget.nearby} nearby / ${budget.context} context` : ""}`;
    this.meta.title = `Input: ${cloud.frame} frame · ${cloud.units}\nObservation: ${cloud.observation_signature}\nTensor: ${cloud.tensor_sha256}\nState: ${cloud.state_id}`;
    this.legend.replaceChildren(...cloud.groups.map((group) => {
      const label = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = group.color;
      label.append(swatch, `${group.name} · ${group.count}`);
      return label;
    }));
    this.fitButton.disabled = this.downloadButton.disabled = false;
    this.updateLayers();
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
