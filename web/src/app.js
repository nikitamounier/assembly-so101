import "./style.css";
import { RewardSection } from "./reward.js";
import {
  filterEntries,
  selectionAfterFilter,
  resetIdentity,
} from "./catalog.js";
import { stateCatalog, stateVariations, admissionKind } from "./states.js";
import { SceneView } from "./viewer.js";
import { PointCloudSection } from "./point-cloud.js";
import {
  variationFields,
  variationStatus,
  progressFields,
} from "./variation.js";

const $ = (selector) => document.querySelector(selector);
const capitalize = (text) =>
  text.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
const filters = {
  bank: "",
  task: "",
  split: "",
  family: "",
  search: "",
  kind: "",
};
const PAGE_SIZE = 18;
let catalog,
  filtered = [],
  selected = null,
  page = 0,
  sequence = 0,
  view;
let pointCloud, rewardSection;
let selectionRequest = Promise.resolve();
let stateNavigation = false,
  activeTask = "",
  browseKind = "nominal";
let variationSplit = "",
  variationStatusFilter = "",
  variationPage = 0;
const rememberedVariations = new Map();
const VARIATION_PAGE_SIZE = 8;

function states() {
  return stateCatalog(catalog.entries, filters.bank);
}

function currentState() {
  return states().find((state) => state.task === activeTask);
}

function navigationEntries() {
  return browseKind === "nominal"
    ? states()
        .map((state) => state.reference)
        .filter(Boolean)
    : stateVariations(currentState(), variationSplit, variationStatusFilter);
}

function drawStateNavigation() {
  const all = states();
  const query = filters.search.toLowerCase().trim();
  const visible = all.filter((state) =>
    `${state.task} ${state.label} ${state.group}`.toLowerCase().includes(query),
  );
  const list = $("#state-entries");
  list.replaceChildren();
  let group;
  for (const state of visible) {
    if (state.group !== group) {
      group = state.group;
      const heading = document.createElement("h3");
      heading.textContent = group;
      if (group === "Final module · Left arm") {
        const videoButton = document.createElement("button");
        videoButton.className = "reference-video-button";
        videoButton.type = "button";
        videoButton.textContent = "▶";
        videoButton.title = "Compare real assembly and simulation";
        videoButton.setAttribute("aria-label", "Watch final module reference video");
        videoButton.onclick = () => {
          const dialog = $("#reference-video-dialog");
          chooseAssemblyVideo("simulation");
          dialog.showModal();
        };
        heading.append(videoButton);
      }
      list.append(heading);
    }
    const button = document.createElement("button");
    button.className =
      "stage-entry" + (state.task === activeTask ? " selected" : "");
    button.setAttribute(
      "aria-current",
      state.task === activeTask ? "step" : "false",
    );
    button.setAttribute(
      "aria-label",
      `${state.task.replaceAll("_", " ")} · ${state.variations.length} variations`,
    );
    const number = document.createElement("span");
    number.className = "stage-id";
    number.textContent = state.task.split("_")[0];
    const label = document.createElement("strong");
    label.textContent = state.label;
    const count = document.createElement("small");
    count.textContent = `${state.variations.length} var`;
    button.append(number, label, count);
    button.onclick = () => openState(state.task, "nominal");
    list.append(button);
  }
  const activeButton = list.querySelector('[aria-current="step"]');
  if (activeButton) {
    const relativeTop =
      activeButton.getBoundingClientRect().top -
      list.getBoundingClientRect().top;
    if (relativeTop < 0) list.scrollTop += relativeTop;
    else if (relativeTop + activeButton.offsetHeight > list.clientHeight)
      list.scrollTop +=
        relativeTop + activeButton.offsetHeight - list.clientHeight;
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No states match. Try a state number or part name.";
    list.append(empty);
  }
  $("#filtered-count").textContent = `${visible.length} assembly states`;
  $("#total").textContent = all.length;
  const state = currentState();
  $("#stage-number").textContent = state
    ? `STATE ${state.task.split("_")[0]} · ${state.label}`
    : "Choose a state";
  $("#stage-group").textContent = state?.group || "";
  $("#variation-count").textContent = state?.variations.length || 0;
  for (const [id, kind] of [
    ["reference-tab", "nominal"],
    ["variations-tab", "variation"],
  ]) {
    const button = $(`#${id}`);
    button.setAttribute("aria-pressed", String(browseKind === kind));
    button.classList.toggle("active", browseKind === kind);
    button.disabled =
      !state ||
      (kind === "nominal" && !state.reference) ||
      (kind === "variation" && !state.variations.length);
  }
  $("#variation-picker").hidden = browseKind !== "variation";
  const variations = stateVariations(
    state,
    variationSplit,
    variationStatusFilter,
  );
  const pageCount = Math.max(
    1,
    Math.ceil(variations.length / VARIATION_PAGE_SIZE),
  );
  variationPage = Math.min(variationPage, pageCount - 1);
  const picker = $("#variation-entries");
  picker.replaceChildren();
  for (const entry of variations.slice(
    variationPage * VARIATION_PAGE_SIZE,
    (variationPage + 1) * VARIATION_PAGE_SIZE,
  )) {
    const button = document.createElement("button");
    const index =
      stateVariations(state).findIndex((item) => item.id === entry.id) + 1;
    const status = admissionKind(entry);
    button.className = `variation-chip ${status}`;
    button.setAttribute("aria-pressed", String(entry.id === selected));
    button.textContent = `Variation ${index} · ${entry.split.toUpperCase()} · ${capitalize(status)}`;
    button.onclick = () => select(entry.id);
    picker.append(button);
  }
  if (!variations.length)
    picker.textContent =
      "No variations match these filters. Try All splits / All statuses.";
  $("#variation-page").textContent =
    `${variations.length} variations${pageCount > 1 ? ` · Page ${variationPage + 1} of ${pageCount}` : ""}`;
  $("#variation-prev").disabled = variationPage === 0;
  $("#variation-next").disabled = variationPage === pageCount - 1;
  $(".variation-pagination").hidden = pageCount === 1;
  const index = filtered.findIndex((entry) => entry.id === selected);
  $("#position").textContent =
    `${browseKind === "nominal" ? "State" : "Variation"} ${index < 0 ? 0 : index + 1} / ${filtered.length}`;
  $("#previous").disabled = index <= 0;
  $("#next").disabled = index < 0 || index >= filtered.length - 1;
  for (const [id, name] of [
    ["previous", "Previous"],
    ["next", "Next"],
  ]) {
    $(`#${id}`).setAttribute(
      "aria-label",
      `${name} ${browseKind === "nominal" ? "state" : "variation"}`,
    );
    $(`#${id}`).title =
      `${name} ${browseKind === "nominal" ? "state" : "variation"}`;
  }
}

async function openState(task, kind) {
  activeTask = task;
  browseKind = kind;
  filtered = navigationEntries();
  variationPage = 0;
  const next =
    kind === "nominal"
      ? currentState()?.reference?.id
      : selectionAfterFilter(filtered, rememberedVariations.get(task));
  await select(next || null);
}

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
  if (stateNavigation) return drawStateNavigation();
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
    const identity = resetIdentity(entry);
    title.textContent = `${capitalize(entry.task)} · ${identity.reference ? "Reference" : entry.seed}`;
    const description = document.createElement("small");
    description.textContent = entry.variation
      ? variationStatus(entry.variation)
      : `${entry.setup || entry.family_name} / Reset ${String(entry.row + 1).padStart(3, "0")}`;
    text.append(title, description);
    const split = document.createElement("span");
    split.className = `list-split ${identity.reference ? "reference" : entry.split}`;
    split.textContent = identity.reference
      ? "REF"
      : entry.variation
        ? entry.split.toUpperCase()
        : ["wowrobo-components", "wowrobo-layout"].includes(catalog.scene)
          ? "CAD"
          : entry.split.toUpperCase();
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
  const variation = snapshot.variation || entry.variation;
  $("#selection-status").hidden = !variation;
  $("#selection-status").textContent = variation
    ? variationStatus(variation)
    : "";
  $("#selection-status").classList.toggle(
    "rejected",
    Boolean(variation?.candidate_rejected || variation?.rejected),
  );
  if (
    !["cube-trial", "wowrobo-components", "wowrobo-layout"].includes(
      catalog.scene,
    )
  ) {
    $(".page-heading p").textContent = entry.distribution?.startsWith("subtask")
      ? "Only the active part is loose; completed prerequisites are assembled."
      : "Full-sequence starts: all assembly parts are loose.";
  }
  const qualification = $("#qualification");
  qualification.hidden = !snapshot.qualification && !variation;
  qualification.querySelector("dl").replaceChildren();
  const showQualification = (metadata, prefix = "") => {
    for (const [key, value] of Object.entries(metadata)) {
      if (["split", "row", "seed", "family"].includes(key)) continue;
      const label = prefix + capitalize(key);
      if (
        value &&
        typeof value === "object" &&
        (!Array.isArray(value) ||
          value.some((item) => item && typeof item === "object"))
      ) {
        showQualification(value, label + " / ");
        continue;
      }
      const term = document.createElement("dt");
      const content = document.createElement("dd");
      term.textContent = label;
      content.textContent = Array.isArray(value)
        ? value
            .map((item) =>
              typeof item === "number"
                ? Number(item.toPrecision(5))
                : String(item),
            )
            .join(", ") || "None"
        : typeof value === "number"
          ? String(Number(value.toPrecision(5)))
          : String(value);
      qualification.querySelector("dl").append(term, content);
    }
  };
  if (snapshot.qualification) showQualification(snapshot.qualification);
  if (variation) showQualification(variation, "Variation / ");
  const identity = resetIdentity(entry);
  $("#selection-title").textContent =
    `${capitalize(entry.task)} / ${identity.suffix}`;
  $("#selection-split").textContent = identity.reference
    ? "REF"
    : entry.variation
      ? entry.split.toUpperCase()
      : ["wowrobo-components", "wowrobo-layout"].includes(catalog.scene)
        ? "CAD"
        : entry.split.toUpperCase();
  $("#selection-split").className =
    `pill ${identity.reference ? "reference" : entry.split}`;
  $("#state-time").textContent = `t = ${snapshot.time.toFixed(3)} s`;
  const bank = catalog.banks.find((bank) => bank.id === entry.bank);
  const fields = [
    ["TASK", capitalize(entry.task)],
    identity.reference
      ? ["REFERENCE", "Reference pose"]
      : ["SEED", String(entry.seed)],
    ["FAMILY", entry.family_name],
    catalog.scene === "cube-trial"
      ? ["BANK", bank.name]
      : [
          "SETUP",
          entry.predecessors?.length
            ? `${entry.predecessors.map(capitalize).join(" + ")} assembled`
            : entry.setup || bank.name,
        ],
  ];
  if (entry.stage) {
    fields.push(
      [
        "LEFT ARM",
        variation?.support_grasp_by_side?.left
          ? `Hold ${capitalize(variation.support_grasp_by_side.left.piece)}`
          : entry.stage.left_role,
      ],
      ["RIGHT ARM", entry.stage.right_role],
      [
        "REFERENCE",
        entry.stage.reference_images.join(", ") || "Inferred intermediate",
      ],
      ["POSE", entry.stage.description],
      [
        "STATUS",
        entry.stage.terminal
          ? "Terminal reference · excluded from training starts"
          : variation
            ? variationStatus(variation)
            : "Review candidate · physical qualification pending",
      ],
    );
    if (entry.stage.boundary_review && entry.review_kind !== "variation") {
      fields.push(["BOUNDARY REVIEW", entry.stage.boundary_review]);
    }
    if (entry.stage.photo_review_note) {
      fields.push(["PHOTO REVIEW", entry.stage.photo_review_note]);
    }
  }
  const variationReview = $("#variation-review");
  variationReview.hidden = !variation && !bank.progress;
  const variationList = variationReview.querySelector("dl");
  variationList.replaceChildren();
  for (const [name, value] of [
    ...variationFields(variation),
    ...progressFields(bank.progress, entry.task),
  ]) {
    const field = document.createElement("div");
    const term = document.createElement("dt");
    const content = document.createElement("dd");
    term.textContent = name;
    content.textContent = value;
    field.append(term, content);
    variationList.append(field);
  }
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
  const hash = new URLSearchParams({
    reset: entry.id,
    mode: view.mode,
    bank: filters.bank || "all",
  });
  if (stateNavigation) {
    hash.set("state", activeTask);
    hash.set("start", browseKind);
    if (variationSplit) hash.set("split", variationSplit);
    if (variationStatusFilter) hash.set("status", variationStatusFilter);
  }
  history.replaceState(null, "", `#${hash}`);
}

async function select(id) {
  const ticket = ++sequence;
  selected = id;
  if (stateNavigation && id) {
    const entry = catalog.entries.find((entry) => entry.id === id);
    activeTask = entry.task;
    browseKind = resetIdentity(entry).reference ? "nominal" : "variation";
    if (browseKind === "variation") rememberedVariations.set(activeTask, id);
    filtered = navigationEntries();
    const index = filtered.findIndex((entry) => entry.id === id);
    variationPage = Math.max(0, Math.floor(index / VARIATION_PAGE_SIZE));
  }
  page = Math.max(
    0,
    Math.floor(filtered.findIndex((entry) => entry.id === id) / PAGE_SIZE),
  );
  drawList();
  if (!id) {
    if (stateNavigation) {
      const hash = new URLSearchParams({
        mode: view.mode,
        bank: filters.bank || "all",
        state: activeTask,
        start: browseKind,
      });
      if (variationSplit) hash.set("split", variationSplit);
      if (variationStatusFilter) hash.set("status", variationStatusFilter);
      history.replaceState(null, "", `#${hash}`);
    }
    $("#selection-status").hidden = true;
    $("#variation-review").hidden = true;
    $("#qualification").hidden = true;
    if (view.mode === "mujoco") await setMode("three");
    view.clear();
    pointCloud.clear();
    rewardSection.clear();
    $("#selection-title").textContent = "No reset selected";
    $("#selection-split").textContent = "";
    $("#state-time").textContent = "—";
    for (const field of document.querySelectorAll("#details strong"))
      field.textContent = "—";
    $("#view-message").textContent = "No resets match these filters.";
    $("#view-message").hidden = false;
    $("#close-inspection").disabled = $("#advanced-inspection").disabled = true;
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
    pointCloud.setSnapshot(snapshot);
    rewardSection.setSnapshot(snapshot);
    $("#view-message").hidden = true;
    $("#close-inspection").disabled = $("#advanced-inspection").disabled = false;
    details(snapshot.entry, snapshot);
    $("#error").hidden = true;
  } catch (error) {
    if (ticket === sequence) fail(error);
  }
}

function applyFilters() {
  if (stateNavigation) {
    const state = currentState() || states()[0];
    return openState(state?.task || "", browseKind);
  }
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
  const three = $("#close-inspection"),
    mujoco = $("#advanced-inspection");
  three.disabled = mujoco.disabled = true;
  message.textContent = "Starting MuJoCo WebViewer…";
  message.hidden = mode !== "mujoco";
  try {
    await view.setMode(mode);
    message.hidden = true;
    three.hidden = mode !== "mujoco";
    mujoco.hidden = mode === "mujoco";
    $("#view-label").textContent =
      mode === "three" ? "3D SCENE" : "MUJOCO WEBVIEWER";
    $(".camera-presets").hidden = mode === "mujoco";
    $("#home-camera").hidden = mode === "mujoco";
    $(".scene-layer-controls").hidden = mode === "mujoco";
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

async function refreshCollections() {
  const button = $("#refresh-collections");
  button.disabled = true;
  button.textContent = "Refreshing…";
  try {
    catalog = await request("/api/refresh", { method: "POST" });
    $("#total").textContent = catalog.entries.length.toLocaleString();
    $("#bank").replaceChildren(option("", "All banks"));
    for (const bank of catalog.banks) {
      const choice = option(
        bank.id,
        `${bank.label || bank.name} · ${bank.compatible ? bank.count : "incompatible"}`,
      );
      choice.disabled = !bank.compatible;
      $("#bank").append(choice);
    }
    const tasks = [...new Set(catalog.entries.map((entry) => entry.task))];
    $("#task").replaceChildren(
      option("", "All tasks"),
      ...tasks.map((task) => option(task, capitalize(task))),
    );
    const families = new Map(
      catalog.entries.map((entry) => [String(entry.family), entry.family_name]),
    );
    $("#family").replaceChildren(
      option("", "All families"),
      ...[...families].map(([id, name]) => option(id, name)),
    );
    if (
      filters.bank &&
      !catalog.banks.some((bank) => bank.id === filters.bank && bank.compatible)
    )
      filters.bank = catalog.default_bank || "";
    if (!tasks.includes(filters.task)) filters.task = "";
    if (!families.has(filters.family)) filters.family = "";
    for (const key of ["bank", "task", "family"])
      $(`#${key}`).value = filters[key];
    const hasVariations = catalog.banks.some(
      (bank) => bank.distribution === "photo_variations",
    );
    $("#start-kind-section").hidden = !hasVariations;
    if (hasVariations) $("#split").closest(".sidebar-section").hidden = false;
    const problems = catalog.banks.flatMap((bank) =>
      bank.errors.map((error) => `${bank.name}: ${error}`),
    );
    $("#bank-note").textContent = problems.length
      ? `${problems.length} bank issues · hover for details`
      : `${catalog.banks.length} collections · updated just now`;
    $("#bank-note").title = problems.join("\n");
    if (stateNavigation) {
      $("#start-kind-section").hidden = true;
      $("#split").closest(".sidebar-section").hidden = true;
      filtered = navigationEntries();
      const next = selectionAfterFilter(filtered, selected);
      if (next) await select(next);
      else
        await openState(
          (currentState() || states()[0])?.task || "",
          browseKind,
        );
    } else {
      filtered = filterEntries(catalog.entries, filters);
      page = 0;
      await select(selectionAfterFilter(filtered, selected));
    }
  } catch (error) {
    fail(error);
  } finally {
    button.disabled = false;
    button.textContent = "Refresh collections";
  }
}

async function main() {
  const [data, scene] = await Promise.all([
    request("/api/catalog"),
    request("/api/scene"),
  ]);
  catalog = data;
  if (["wowrobo-components", "wowrobo-layout"].includes(catalog.scene)) {
    document.title = "SO-101 · WowRobo components";
    $(".page-heading h1").textContent = "WowRobo components.";
    $(".page-heading p").textContent =
      "Corrected leader pieces with camera-equipped follower helpers. Select a piece to inspect it.";
    $("label[for=task]").textContent = "COMPONENT";
    $("label[for=bank]").textContent = "COLLECTION";
    $("#split").closest(".sidebar-section").hidden = true;
    $(".total span").textContent = "REVIEW VIEWS";
    $("[data-camera=detail]").textContent = "Component";
  }
  if (catalog.scene === "wowrobo-layout") {
    document.title = "SO-101 · Initial workcell";
    $(".page-heading h1").textContent = "Initial workcell.";
    $(".page-heading p").textContent =
      "72 cm between base rear edges · assembly base at 30 cm · grip spans stay tangential. Blue: left pickup; orange: right pickup.";
    $("label[for=task]").textContent = "SCENE";
    $(".total span").textContent = "LAYOUTS";
  }
  if (catalog.banks.some((bank) => bank.distribution === "photo_review")) {
    document.title = "SO-101 · Photo reset sequence";
    $(".page-heading h1").textContent = "Photo reset sequence.";
    $(".page-heading p").textContent =
      "Both arm roles follow the reference photos. Kinematic candidates: contacts, supporting grasps and rewards await physical qualification.";
    $("label[for=task]").textContent = "STAGE";
    $(".total span").textContent = "REVIEW STATES";
  }
  if (catalog.banks.some((bank) => bank.distribution === "photo_variations")) {
    document.title = "SO-101 · Reset variations";
    $(".page-heading h1").textContent = "Reset variations.";
    $(".page-heading p").textContent =
      "Compare approved nominal starts and generated variations. Physical validation status is recorded separately for every sample.";
    $("label[for=task]").textContent = "STAGE";
    $("label[for=bank]").textContent = "COLLECTION";
    $(".total span").textContent = "REVIEW STARTS";
    $("#start-kind-section").hidden = false;
    $("#split").closest(".sidebar-section").hidden = false;
  }
  if (catalog.scene === "cube-trial") {
    document.title = "SO-101 · Cube handoff resets";
    $(".page-heading p").textContent =
      "Explore the starting conditions for the cube handoff trial.";
    $("label[for=task]").textContent = "TRIAL TASK";
    $("[data-camera=detail]").textContent = "Handoff";
  }
  stateNavigation = catalog.entries.some(
    (entry) => resetIdentity(entry).reference,
  );
  if (stateNavigation) {
    $("#app").classList.add("state-navigation");
    document.title = "SO-101 · Assembly states";
    $(".page-heading h1").textContent = "Assembly states";
    $(".page-heading p").textContent =
      "Choose a state, then inspect its reference pose or explore variations.";
    $(".total span").textContent = "ASSEMBLY STATES";
    $("#browser-options").open = false;
    for (const id of ["task", "family", "kind", "split"])
      $(`#${id}`).closest(".sidebar-section").hidden = true;
    $("#state-context").hidden = false;
    $("#state-entries").hidden = false;
    $("#entries").hidden = true;
    $(".pagination").hidden = true;
    $("#list-order").textContent = "ASSEMBLY ORDER";
    $("#search").placeholder = "Find state or part…";
    $("#search").setAttribute("aria-label", "Find assembly state");
    $("#reference-tab").onclick = () => openState(activeTask, "nominal");
    $("#variations-tab").onclick = () => openState(activeTask, "variation");
    $("#variation-split").onchange = (event) => {
      variationSplit = event.target.value;
      openState(activeTask, "variation");
    };
    $("#variation-status").onchange = (event) => {
      variationStatusFilter = event.target.value;
      openState(activeTask, "variation");
    };
    $("#variation-prev").onclick = () => {
      variationPage--;
      drawStateNavigation();
    };
    $("#variation-next").onclick = () => {
      variationPage++;
      drawStateNavigation();
    };
  }
  view = new SceneView($("#scene-view"));
  pointCloud = new PointCloudSection(view);
  rewardSection = new RewardSection(view, async (values) => {
    await request("/api/reward-tolerance", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(values)});
    if (selected) await select(selected);
  });
  view.setScene(scene);
  if (catalog.scene === "wowrobo-layout") {
    try {
      const response = await fetch("/collision-boxes.json", { cache: "no-cache" });
      if (response.ok) {
        const boxes = await response.json();
        view.setCollisionBoxes(boxes);
        const button = $("#toggle-boxes");
        let visible = false;
        button.hidden = false;
        button.setAttribute("aria-pressed", "false");
        view.setCollisionBoxesVisible(false);
        button.title = `Actual bare MuJoCo collision geometry: ${boxes.boxes.length} box geoms · Blue: robot · Green: finger pads · Orange: assembly parts · 3D toggles the visual CAD`;
        pointCloud.geometryVisible = true;
        pointCloud.pointsVisible = false;
        pointCloud.updateLayers();
        $("#view-label").textContent = "3D SCENE";
        button.onclick = () => {
          visible = !visible;
          view.setCollisionBoxesVisible(visible);
          button.setAttribute("aria-pressed", String(visible));
        };
      }
    } catch (error) {
      console.warn("Collision overlay unavailable:", error.message);
    }
  }
  $("#total").textContent = catalog.entries.length.toLocaleString();
  for (const bank of catalog.banks) {
    const element = option(
      bank.id,
      `${bank.label || bank.name} · ${bank.compatible ? bank.count : "incompatible"}`,
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
  for (const key of ["bank", "task", "family", "search", "kind"]) {
    $(`#${key}`).addEventListener(
      key === "search" ? "input" : "change",
      (event) => {
        filters[key] = event.target.value;
        if (stateNavigation && key === "search") drawStateNavigation();
        else applyFilters();
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
  $("#refresh-collections").onclick = refreshCollections;
  $("#next").onclick = () => move(1);
  $("#page-prev").onclick = () => {
    page--;
    drawList();
  };
  $("#page-next").onclick = () => {
    page++;
    drawList();
  };
  window.addEventListener("pagehide", () => {
    if (view.mode === "mujoco") navigator.sendBeacon("/api/studio/stop");
  });
  $("#close-inspection").onclick = () => setMode("three");
  $("#advanced-inspection").onclick = () => setMode("mujoco");
  $("#home-camera").onclick = () => view.preset("perspective");
  for (const button of document.querySelectorAll("[data-camera]"))
    button.onclick = () => {
      view.preset(button.dataset.camera);
      for (const item of document.querySelectorAll("[data-camera]"))
        item.classList.toggle("active", item === button);
    };
  document.addEventListener("keydown", (event) => {
    if (referenceDialog.open || /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
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
  filters.bank =
    hash.get("bank") === "all"
      ? ""
      : hash.get("bank") || catalog.default_bank || "";
  if (
    filters.bank &&
    !catalog.banks.some((bank) => bank.id === filters.bank && bank.compatible)
  )
    filters.bank = catalog.default_bank || "";
  $("#bank").value = filters.bank;
  if (stateNavigation) {
    const linked = catalog.entries.find(
      (entry) =>
        entry.id === hash.get("reset") &&
        (!filters.bank || entry.bank === filters.bank),
    );
    activeTask =
      linked?.task ||
      (
        states().find((state) => state.task === hash.get("state")) ||
        states()[0]
      )?.task ||
      "";
    browseKind = linked
      ? resetIdentity(linked).reference
        ? "nominal"
        : "variation"
      : hash.get("start") === "variation"
        ? "variation"
        : "nominal";
    variationSplit = ["train", "eval"].includes(hash.get("split"))
      ? hash.get("split")
      : "";
    variationStatusFilter = ["qualified", "pending", "rejected"].includes(
      hash.get("status"),
    )
      ? hash.get("status")
      : "";
    $("#variation-split").value = variationSplit;
    $("#variation-status").value = variationStatusFilter;
    filtered = navigationEntries();
    const next =
      browseKind === "nominal"
        ? currentState()?.reference?.id
        : selectionAfterFilter(filtered, linked?.id);
    await select(next || null);
  } else {
    filtered = filterEntries(catalog.entries, filters);
    await select(selectionAfterFilter(filtered, hash.get("reset")));
  }

}

main().catch(fail);

const referenceDialog = $("#reference-video-dialog");
$("#close-reference-video").onclick = () => referenceDialog.close();
referenceDialog.addEventListener("close", () => {
  const video = $("#reference-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
});

function chooseAssemblyVideo(source) {
  const video = $("#reference-video");
  video.pause();
  video.src = `/reference/final-module-${source}.mp4`;
  $("#video-real").setAttribute("aria-pressed", String(source === "real"));
  $("#video-simulation").setAttribute("aria-pressed", String(source === "simulation"));
  $("#reference-video-caption").textContent = source === "real"
    ? "Real assembly reference. Intended robot motion: hold the motor, let the bracket fold, then lower it so the bracket slides and opens."
    : "Earlier diagnostic before housing centering: force-held motor, passive hinge, fixed receiver, no hand. Bracket folds ~10° then opens to ~0.6°; motor reaches within 0.17 mm of the seating reference under applied force. In a separate 1 s release check it drifted 0.13 mm. Fixed-receiver diagnostic, not training qualification.";
}
$("#video-real").onclick = () => chooseAssemblyVideo("real");
$("#video-simulation").onclick = () => chooseAssemblyVideo("simulation");
