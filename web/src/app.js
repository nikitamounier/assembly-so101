import "./style.css";
import { filterEntries, selectionAfterFilter } from "./catalog.js";
import { SceneView } from "./viewer.js";

const $ = (selector) => document.querySelector(selector);
const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const filters = { bank: "", task: "", split: "", family: "", search: "" };
const PAGE_SIZE = 18;
let catalog,
  filtered = [],
  selected = null,
  page = 0,
  sequence = 0,
  view;
let selectionRequest = Promise.resolve();

function fail(error) {
  $("#error").textContent = error.message || String(error);
  $("#error").hidden = false;
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function option(value, label) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = label;
  return element;
}

function drawList() {
  const list = $("#entries");
  list.replaceChildren();
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  page = Math.min(page, pages - 1);
  for (const entry of filtered.slice(
    page * PAGE_SIZE,
    (page + 1) * PAGE_SIZE,
  )) {
    const button = document.createElement("button");
    button.className =
      "reset-entry" + (entry.id === selected ? " selected" : "");
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(entry.id === selected));
    const icon = document.createElement("span");
    icon.className = "reset-glyph";
    icon.textContent = "◇";
    const text = document.createElement("span");
    text.className = "reset-entry-text";
    const title = document.createElement("strong");
    title.textContent = `${capitalize(entry.task)} · ${entry.seed}`;
    const description = document.createElement("small");
    description.textContent = `${entry.family_name} / ${entry.collection.includes("family_eval/") ? "Family evaluation" : "Reset " + String(entry.row + 1).padStart(3, "0")}`;
    text.append(title, description);
    const split = document.createElement("span");
    split.className = `list-split ${entry.split}`;
    split.textContent = entry.split.toUpperCase();
    button.append(icon, text, split);
    button.addEventListener("click", () => select(entry.id));
    list.append(button);
  }
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = catalog.entries.length
      ? "No resets match these filters."
      : "No compatible reset banks found. Generate banks or add a bank root with --banks.";
    list.append(empty);
  }
  $("#filtered-count").textContent =
    `${filtered.length.toLocaleString()} resets`;
  $("#page-label").textContent = `${page + 1} / ${pages}`;
  $("#page-prev").disabled = page === 0;
  $("#page-next").disabled = page >= pages - 1;
  const index = filtered.findIndex((entry) => entry.id === selected);
  $("#position").textContent =
    index < 0 ? "0 / 0" : `${index + 1} / ${filtered.length}`;
  $("#previous").disabled = index <= 0;
  $("#next").disabled = index < 0 || index >= filtered.length - 1;
}

function details(entry, snapshot) {
  $("#selection-title").textContent =
    `${capitalize(entry.task)} / Seed ${entry.seed}`;
  $("#selection-split").textContent = entry.split.toUpperCase();
  $("#selection-split").className = `pill ${entry.split}`;
  $("#state-time").textContent = `t = ${snapshot.time.toFixed(3)} s`;
  const bank = catalog.banks.find((bank) => bank.id === entry.bank);
  const fields = [
    ["TASK", capitalize(entry.task)],
    ["SEED", String(entry.seed)],
    ["FAMILY", entry.family_name],
    ["BANK", bank.name],
  ];
  $("#details").replaceChildren(
    ...fields.map(([name, value]) => {
      const field = document.createElement("div");
      const label = document.createElement("span");
      const content = document.createElement("strong");
      label.textContent = name;
      content.textContent = value;
      content.title = value;
      field.append(label, content);
      return field;
    }),
  );
  const hash = new URLSearchParams({ reset: entry.id, mode: view.mode });
  history.replaceState(null, "", `#${hash}`);
}

async function select(id) {
  const ticket = ++sequence;
  selected = id;
  page = Math.max(
    0,
    Math.floor(filtered.findIndex((entry) => entry.id === id) / PAGE_SIZE),
  );
  drawList();
  if (!id) {
    view.clear();
    $("#selection-title").textContent = "No reset selected";
    $("#selection-split").textContent = "";
    $("#state-time").textContent = "—";
    for (const field of document.querySelectorAll("#details strong"))
      field.textContent = "—";
    $("#view-message").textContent = "No resets match these filters.";
    $("#view-message").hidden = false;
    $("#three-mode").disabled = $("#mujoco-mode").disabled = true;
    return;
  }
  try {
    selectionRequest = selectionRequest
      .catch(() => {})
      .then(() =>
        request("/api/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        }),
      );
    const snapshot = await selectionRequest;
    if (ticket !== sequence) return;
    view.setSnapshot(snapshot);
    $("#view-message").hidden = true;
    $("#three-mode").disabled = $("#mujoco-mode").disabled = false;
    details(snapshot.entry, snapshot);
    $("#error").hidden = true;
  } catch (error) {
    if (ticket === sequence) fail(error);
  }
}

function applyFilters() {
  filtered = filterEntries(catalog.entries, filters);
  const next = selectionAfterFilter(filtered, selected);
  page = 0;
  drawList();
  if (next !== selected) select(next);
}

function move(delta) {
  const index = filtered.findIndex((entry) => entry.id === selected);
  const next = filtered[index + delta];
  if (next) select(next.id);
}

async function setMode(mode) {
  const message = $("#view-message");
  const three = $("#three-mode"),
    mujoco = $("#mujoco-mode");
  three.disabled = mujoco.disabled = true;
  message.textContent = "Starting MuJoCo WebViewer…";
  message.hidden = mode !== "mujoco";
  try {
    await view.setMode(mode);
    message.hidden = true;
    for (const [button, name] of [
      [three, "three"],
      [mujoco, "mujoco"],
    ]) {
      button.classList.toggle("active", mode === name);
      button.setAttribute("aria-pressed", String(mode === name));
    }
    $("#view-label").textContent =
      mode === "three" ? "3D SCENE" : "MUJOCO WEBVIEWER";
    $(".camera-presets").hidden = mode === "mujoco";
    $("#home-camera").hidden = mode === "mujoco";
    $("#orbit-hint").textContent =
      mode === "three"
        ? "Drag to orbit · Scroll to zoom · Shift-drag to pan"
        : "Official MuJoCo renderer · Native reset poses";
    const hash = new URLSearchParams(location.hash.slice(1));
    hash.set("mode", mode);
    history.replaceState(null, "", `#${hash}`);
  } catch (error) {
    message.hidden = true;
    fail(error);
  } finally {
    three.disabled = mujoco.disabled = false;
  }
}

async function main() {
  const [data, scene] = await Promise.all([
    request("/api/catalog"),
    request("/api/scene"),
  ]);
  catalog = data;
  view = new SceneView($("#scene-view"));
  view.setScene(scene);
  $("#total").textContent = catalog.entries.length.toLocaleString();
  for (const bank of catalog.banks) {
    const element = option(
      bank.id,
      `${bank.name} · ${bank.compatible ? bank.count : "incompatible"}`,
    );
    element.disabled = !bank.compatible;
    $("#bank").append(element);
  }
  const problems = catalog.banks.flatMap((bank) =>
    bank.errors.map((error) => `${bank.name}: ${error}`),
  );
  $("#bank-note").textContent = problems.length
    ? `${problems.length} bank issue${problems.length > 1 ? "s" : ""} · hover for details`
    : `${catalog.banks.length} bank${catalog.banks.length === 1 ? "" : "s"} discovered`;
  $("#bank-note").title = problems.join("\n");
  for (const task of [...new Set(catalog.entries.map((entry) => entry.task))])
    $("#task").append(option(task, capitalize(task)));
  const families = new Map(
    catalog.entries.map((entry) => [entry.family, entry.family_name]),
  );
  for (const [id, name] of [...families].sort(([a], [b]) => a - b))
    $("#family").append(option(String(id), name));
  for (const key of ["bank", "task", "family", "search"]) {
    $(`#${key}`).addEventListener(
      key === "search" ? "input" : "change",
      (event) => {
        filters[key] = event.target.value;
        applyFilters();
      },
    );
  }
  for (const button of document.querySelectorAll("#split button"))
    button.addEventListener("click", () => {
      filters.split = button.dataset.value;
      for (const item of document.querySelectorAll("#split button"))
        item.classList.toggle("active", item === button);
      applyFilters();
    });
  $("#previous").onclick = () => move(-1);
  $("#next").onclick = () => move(1);
  $("#page-prev").onclick = () => {
    page--;
    drawList();
  };
  $("#page-next").onclick = () => {
    page++;
    drawList();
  };
  $("#three-mode").onclick = () => setMode("three");
  $("#mujoco-mode").onclick = () => setMode("mujoco");
  $("#home-camera").onclick = () => view.preset("perspective");
  for (const button of document.querySelectorAll("[data-camera]"))
    button.onclick = () => {
      view.preset(button.dataset.camera);
      for (const item of document.querySelectorAll("[data-camera]"))
        item.classList.toggle("active", item === button);
    };
  document.addEventListener("keydown", (event) => {
    if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      move(1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      move(-1);
    }
    if (event.key.toLowerCase() === "f" && view.mode === "three")
      view.preset("perspective");
    if (event.key === "/") {
      event.preventDefault();
      $("#search").focus();
    }
  });
  window.addEventListener("pagehide", () => view.dispose(), { once: true });
  const hash = new URLSearchParams(location.hash.slice(1));
  filtered = filterEntries(catalog.entries, filters);
  await select(selectionAfterFilter(filtered, hash.get("reset")));
  if (hash.get("mode") === "mujoco" && selected) await setMode("mujoco");
}

main().catch(fail);
