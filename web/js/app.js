import { buildSystem, estimate, parallelLevels, ramBandwidthGbps } from "./estimator.js";
import {
  LANGS, lang, loadLocale, numberLocale, pickLang, saveLang, savedLang, t, translateDom,
} from "./i18n.js";
import { Schedule } from "./mock.js";
import {
  BACKEND_NAMES, buildReport, cpuChecks, gpuChecks, markdown, parallelChecks, timingInfo, warnings,
} from "./report.js";
import { newestFirst, searchParts } from "./search.js";
import { decodeState, encodeState } from "./state.js";
import { charsPerToken, chunkText, guessLang } from "./tokenize.js";

const RAM_SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536];
const RAM_KINDS = ["ddr4-2666", "ddr4-3200", "ddr4-3600", "ddr5-4800", "ddr5-5600",
                   "ddr5-6000", "ddr5-6400", "ddr5-8000"];
const RAM_CHANNELS = [1, 2, 4, 8, 12];

// Common kinds plus every CPU preset's default memory, sorted by generation and speed.
function ramKinds() {
  const kinds = new Set([...RAM_KINDS, ...P.cpus.map((c) => c.mem_kind).filter(Boolean)]);
  const key = (k) => { const [g, mt] = k.split("-"); return Number(g.replace(/\D/g, "")) * 1e5 + Number(mt); };
  return [...kinds].sort((a, b) => key(a) - key(b));
}
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
const KV_BITS = [16, 8, 4];
// Concurrent request slots (llama-server --parallel).
const PARALLEL = [1, 2, 4, 8, 16, 32];
// Tokens already in the KV cache when playback starts (llama-bench -d).
const DEPTHS = [0, 4096, 8192, 16384, 32768, 65536, 131072];
const MAX_CONFIGS = 2;
const LETTERS = "AB";

const DEFAULT_CONFIG = {
  platform: "pc",
  gpus: ["rtx3090-24gb"],
  cpu: "ryzen7-5800x",
  ramGb: 64,
  ramKind: "ddr4-3200",
  ramChannels: 2,
  chip: "apple-m4-max-40c",
  soc: "",
  memoryGb: 64,
  backend: "",
  showDatacenter: false,
  model: "qwen3-32b",
  quant: "Q4_K_M",
  context: 8192,
  kvBits: 16,
  parallel: 1,
};
// Defaults when a configuration moves to the server tab.
const SERVER_DEFAULT = { gpu: "h100-sxm-80gb", cpu: "epyc-9654", ramGb: 512 };
const DEFAULT_STATE = {
  configs: [DEFAULT_CONFIG],
  prompt: 1000,
  maxTokens: 300,
  depth: 0,
};

let P; // presets.json
let TEXTS; // data/texts/manifest.json
let CREDITS; // data/texts/credits.json
let state;
let playback = null;

// ---------- helpers ----------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("data-") || k.startsWith("aria-") || k === "for") el.setAttribute(k, v);
    else el[k] = v;
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function select(key, options, value, extra = {}) {
  return h("select", { "data-k": key, ...extra },
    options.map(([v, label]) => h("option", { value: String(v), selected: String(v) === String(value) }, label)));
}

function field(label, control) {
  return h("label", {}, label, control);
}

const fmtInt = (n) => Math.round(n).toLocaleString(numberLocale());

export function fmtTps(x) {
  if (!Number.isFinite(x)) return "–";
  if (x >= 1000) return fmtInt(x);
  if (x >= 100) return x.toFixed(0);
  if (x >= 10) return x.toFixed(1);
  return x.toFixed(2);
}

export function fmtTime(s) {
  if (!Number.isFinite(s)) return "–";
  if (s < 1) return t("time.ms", { n: Math.round(s * 1000) });
  if (s < 60) return t("time.s", { n: s.toFixed(1) });
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m < 60 ? t("time.minSec", { m, s: String(sec).padStart(2, "0") })
    : t("time.hourMin", { h: Math.floor(m / 60), m: m % 60 });
}

const gb = (x) => `${x.toFixed(1)} GB`;

function modelLabel(m) {
  const size = m.active_params_b ? t("option.moe", { total: m.params_b, active: m.active_params_b }) : `${m.params_b}B`;
  return `${m.name} (${size})`;
}

// Backends llama.cpp offers per GPU vendor; the first is the default.
const VENDOR_BACKENDS = { NVIDIA: ["cuda", "vulkan"], AMD: ["rocm", "vulkan"], Intel: ["sycl", "vulkan"], Apple: ["metal"] };

const chipName = (a) => a.label || a.name;

function chipLabel(a) {
  return t("option.chip", { name: chipName(a), cores: a.gpu_cores, bw: a.bandwidth_gbps });
}

function socLabel(a) {
  return t("option.soc", { name: chipName(a), bw: a.bandwidth_gbps });
}

const gpuName = (name) => P.gpus.find((g) => g.name === name)?.label || name;

function gpuLabel(g) {
  return `${g.label || g.name} · ${g.vram_gb} GB`;
}

// Options grouped as "<vendor> · <segment>" for GPUs and CPUs, "<vendor>" for SoCs. Groups
// keep the preset order (NVIDIA, AMD, Intel); within a group newer parts come first.
function groupedSelect(key, items, value, labelOf, groupOf, extra) {
  const groups = new Map();
  for (const x of items) {
    const g = groupOf(x);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(x);
  }
  for (const [g, xs] of groups) groups.set(g, newestFirst(xs));
  const option = (x) => h("option", { value: x.name, selected: x.name === value }, labelOf(x));
  return h("select", { "data-k": key, ...extra },
    [...groups].map(([label, xs]) => (label ? h("optgroup", { label }, xs.map(option)) : xs.map(option))));
}

const segmentGroup = (prefix) => (x) => (x.vendor ? `${x.vendor} · ${t(`${prefix}.${x.segment}`)}` : "");
const appleChips = () => P.unified.filter((a) => a.vendor === "Apple");
const otherChips = () => P.unified.filter((a) => a.vendor !== "Apple");
// PC and server configurations share the same form; the server tab lists data-center GPUs
// and server CPUs, the PC tab everything else.
const isPcLike = (cfg) => cfg.platform === "pc" || cfg.platform === "server";
// The PC tab can opt in to data-center cards (e.g. an A100 PCIe in a workstation).
const gpusFor = (platform, withDatacenter = false) => P.gpus.filter((g) => (platform === "server"
  ? g.segment === "datacenter" : withDatacenter || g.segment !== "datacenter"));
const cpusFor = (platform) => P.cpus.filter((c) => (c.segment === "server") === (platform === "server"));

// Select a CPU together with the memory it supports; the user can still change the memory.
function setCpu(cfg, name) {
  const cpu = P.cpus.find((c) => c.name === name);
  cfg.cpu = name;
  if (cpu?.mem_kind) cfg.ramKind = cpu.mem_kind;
  if (cpu?.mem_channels) cfg.ramChannels = cpu.mem_channels;
}

function switchPlatform(cfg, platform) {
  const was = cfg.platform;
  cfg.platform = platform;
  if (!isPcLike(cfg)) return;
  const allowed = gpusFor(platform, cfg.showDatacenter).map((g) => g.name);
  cfg.gpus = cfg.gpus.filter((g) => allowed.includes(g));
  if (!cfg.gpus.length) cfg.gpus = [platform === "server" ? SERVER_DEFAULT.gpu : DEFAULT_CONFIG.gpus[0]];
  if (!cpusFor(platform).some((c) => c.name === cfg.cpu)) {
    setCpu(cfg, platform === "server" ? SERVER_DEFAULT.cpu : DEFAULT_CONFIG.cpu);
  }
  if (platform === "server" && was !== "server") cfg.ramGb = Math.max(cfg.ramGb, SERVER_DEFAULT.ramGb);
}

function activeChip(cfg) {
  if (cfg.platform === "apple") return P.unified.find((a) => a.name === cfg.chip);
  if (cfg.platform === "unified") return P.unified.find((a) => a.name === cfg.soc);
  return null;
}

// Backend choices for a configuration: the vendors' defaults plus shared ones (Vulkan).
function backendChoices(cfg) {
  const vendors = isPcLike(cfg)
    ? [...new Set(cfg.gpus.map((g) => P.gpus.find((x) => x.name === g)?.vendor).filter(Boolean))]
    : [activeChip(cfg)?.vendor].filter(Boolean);
  if (!vendors.length) return [];
  const lists = vendors.map((v) => VENDOR_BACKENDS[v] || ["vulkan"]);
  const shared = lists[0].filter((b) => lists.every((l) => l.includes(b)) && b !== lists[0][0]);
  return ["", ...shared];
}

// ---------- state ----------

function sanitizeConfig(c) {
  const cfg = { ...DEFAULT_CONFIG, ...(c && typeof c === "object" ? c : {}) };
  const has = (list, name) => list.some((x) => x.name === name);
  cfg.platform = ["apple", "unified", "server"].includes(cfg.platform) ? cfg.platform : "pc";
  const pcLike = isPcLike(cfg) ? cfg.platform : "pc";
  cfg.showDatacenter = cfg.showDatacenter === true;
  cfg.gpus = Array.isArray(cfg.gpus)
    ? cfg.gpus.filter((g) => has(gpusFor(pcLike, cfg.showDatacenter), g)).slice(0, 8) : [];
  if (!has(cpusFor(pcLike), cfg.cpu)) cfg.cpu = pcLike === "server" ? SERVER_DEFAULT.cpu : DEFAULT_CONFIG.cpu;
  if (!RAM_SIZES.includes(Number(cfg.ramGb))) cfg.ramGb = DEFAULT_CONFIG.ramGb;
  if (!ramKinds().includes(cfg.ramKind)) cfg.ramKind = DEFAULT_CONFIG.ramKind;
  if (!RAM_CHANNELS.includes(Number(cfg.ramChannels))) cfg.ramChannels = DEFAULT_CONFIG.ramChannels;
  if (!has(appleChips(), cfg.chip)) cfg.chip = DEFAULT_CONFIG.chip;
  if (!has(otherChips(), cfg.soc)) cfg.soc = otherChips()[0]?.name || "";
  if (cfg.platform === "unified" && !cfg.soc) cfg.platform = "pc";
  const chip = activeChip(cfg);
  if (chip && !chip.memory_options_gb.includes(Number(cfg.memoryGb))) {
    cfg.memoryGb = Math.max(...chip.memory_options_gb);
  }
  if (!backendChoices(cfg).includes(cfg.backend)) cfg.backend = "";
  if (!has(P.models, cfg.model)) cfg.model = DEFAULT_CONFIG.model;
  if (!(cfg.quant in P.quant_bpw)) cfg.quant = DEFAULT_CONFIG.quant;
  if (!CONTEXTS.includes(Number(cfg.context))) cfg.context = DEFAULT_CONFIG.context;
  if (!KV_BITS.includes(Number(cfg.kvBits))) cfg.kvBits = DEFAULT_CONFIG.kvBits;
  if (!PARALLEL.includes(Number(cfg.parallel))) cfg.parallel = DEFAULT_CONFIG.parallel;
  for (const k of ["ramGb", "ramChannels", "memoryGb", "context", "kvBits", "parallel"]) cfg[k] = Number(cfg[k]);
  return cfg;
}

function loadState() {
  const saved = location.hash.length > 1 ? decodeState(location.hash.slice(1)) : null;
  const src = saved || DEFAULT_STATE;
  const configs = (Array.isArray(src.configs) ? src.configs : DEFAULT_STATE.configs)
    .slice(0, MAX_CONFIGS).map(sanitizeConfig);
  const clamp = (v, lo, hi, d) => (Number.isFinite(Number(v)) ? Math.min(hi, Math.max(lo, Math.round(v))) : d);
  return {
    configs: configs.length ? configs : [sanitizeConfig(DEFAULT_CONFIG)],
    prompt: clamp(src.prompt, 1, 200000, DEFAULT_STATE.prompt),
    maxTokens: clamp(src.maxTokens, 1, 4000, DEFAULT_STATE.maxTokens),
    depth: DEPTHS.includes(src.depth) ? src.depth : DEFAULT_STATE.depth,
    text: TEXTS.texts.some((x) => x.id === src.text) ? src.text : defaultTextId(),
  };
}

// Custom text never goes into the URL: a shared link falls back to the default text.
function saveState() {
  const shared = state.text === CUSTOM ? { ...state, text: undefined } : state;
  history.replaceState(null, "", `#${encodeState(shared)}`);
}

// ---------- estimate ----------

function systemSpec(cfg) {
  const backend = cfg.backend || undefined;
  if (!isPcLike(cfg)) return { kind: "unified", chip: activeChip(cfg).name, memory_gb: cfg.memoryGb, backend };
  return {
    kind: "pc", cpu: cfg.cpu, ram_gb: cfg.ramGb,
    ram_bandwidth_gbps: ramBandwidthGbps(cfg.ramKind, cfg.ramChannels), gpus: cfg.gpus, backend,
  };
}

function compute(cfg) {
  const system = buildSystem(systemSpec(cfg), P);
  const m = P.models.find((x) => x.name === cfg.model);
  const est = estimate(system, m, cfg.quant, cfg.context, cfg.kvBits, P.efficiency, P.quant_bpw, cfg.parallel);
  return { system, est };
}

// ---------- rendering: configuration ----------

function renderConfigs() {
  const root = document.getElementById("configs");
  const focused = document.activeElement?.dataset?.focusId;
  root.classList.toggle("single", state.configs.length === 1);
  root.replaceChildren(...state.configs.map((cfg, i) => renderConfig(cfg, i)));
  root.append(h("div", { class: "compare-bar" },
    state.configs.length < MAX_CONFIGS
      ? h("button", { type: "button", "data-act": "add-config" }, t("config.compare")) : null,
    h("span", { class: "report-links", title: t("report.hint") }, `${t("report.label")} `,
      h("button", { type: "button", "data-act": "report", "data-v": "json" }, t("report.json")),
      " · ",
      h("button", { type: "button", "data-act": "report", "data-v": "markdown" }, t("report.markdown")))));
  if (focused) root.querySelector(`[data-focus-id="${focused}"]`)?.focus();
  renderStreams();
}

function renderConfig(cfg, i) {
  const single = state.configs.length === 1;
  const panel = h("div", { class: "panel config", "data-i": i });
  panel.append(h("div", { class: "config-head" },
    single ? h("h2", {}, t("config.single"))
      : h("h2", { "aria-label": t("config.title", { letter: LETTERS[i] }) }, h("span", { class: "tag" }, `[${LETTERS[i]}]`), t("config.single")),
    h("div", { class: "segmented", role: "group", "aria-label": t("config.platform") },
      h("button", { type: "button", "data-act": "platform", "data-v": "pc", "aria-pressed": String(cfg.platform === "pc") }, t("platform.pc")),
      h("button", { type: "button", "data-act": "platform", "data-v": "apple", "aria-pressed": String(cfg.platform === "apple") }, t("platform.mac")),
      otherChips().length ? h("button", { type: "button", "data-act": "platform", "data-v": "unified", "aria-pressed": String(cfg.platform === "unified") }, t("platform.unified")) : null,
      h("button", { type: "button", "data-act": "platform", "data-v": "server", "aria-pressed": String(cfg.platform === "server") }, t("platform.server"))),
    state.configs.length > 1 ? h("button", { type: "button", class: "link", "data-act": "remove-config" }, t("config.remove")) : null));

  const settings = h("div", { class: "config-settings" });
  settings.append(h("h3", {}, t("section.hardware")));
  if (!isPcLike(cfg)) {
    const chip = activeChip(cfg);
    const chipSelect = cfg.platform === "apple"
      ? select("chip", newestFirst(appleChips()).map((a) => [a.name, chipLabel(a)]), cfg.chip, { "data-focus-id": `${i}-chip` })
      : groupedSelect("soc", otherChips(), cfg.soc, socLabel, (a) => a.vendor, { "data-focus-id": `${i}-soc` });
    settings.append(h("div", { class: "fields" },
      field(t(cfg.platform === "apple" ? "field.chip" : "field.system"), chipSelect)));
    settings.append(h("div", { class: "fields" },
      field(t("field.unifiedMemory"), select("memoryGb", chip.memory_options_gb.map((m) => [m, `${m} GB`]), cfg.memoryGb, { "data-focus-id": `${i}-mem` }))));
  } else {
    cfg.gpus.forEach((g, j) => {
      settings.append(h("div", { class: "gpu-row" },
        field(j === 0 ? t("field.gpu") : t("field.gpuN", { n: j + 1 }),
          groupedSelect("gpu", gpusFor(cfg.platform, cfg.showDatacenter), g, gpuLabel, segmentGroup("gpu.segment"), { "data-j": j, "data-focus-id": `${i}-gpu-${j}` })),
        field(t("field.vram"), h("input", { value: `${P.gpus.find((x) => x.name === g).vram_gb} GB`, disabled: true })),
        h("button", { type: "button", "data-act": "remove-gpu", "data-j": j, "aria-label": t("gpu.remove", { n: j + 1 }) }, "✕")));
    });
    settings.append(h("button", { type: "button", class: "link", "data-act": "add-gpu" },
      cfg.gpus.length ? t("gpu.add") : t("gpu.addFirst")));
    if (cfg.platform === "pc") {
      settings.append(h("label", { class: "check" },
        h("input", { type: "checkbox", "data-k": "showDatacenter", checked: cfg.showDatacenter, "data-focus-id": `${i}-dc` }),
        t("gpu.showDatacenter")));
    }
    settings.append(h("div", { class: "fields" },
      field(t("field.cpu"), groupedSelect("cpu", cpusFor(cfg.platform), cfg.cpu, (c) => t("option.cpu", { name: c.label || c.name, cores: c.cores }),
        segmentGroup("cpu.segment"), { "data-focus-id": `${i}-cpu` }))));
    settings.append(h("div", { class: "fields" },
      field(t("field.ram"), select("ramGb", RAM_SIZES.map((s) => [s, `${s} GB`]), cfg.ramGb, { "data-focus-id": `${i}-ram` })),
      field(t("field.ramType"), select("ramKind", ramKinds().map((k) => [k, k.toUpperCase()]), cfg.ramKind, { "data-focus-id": `${i}-kind` })),
      field(t("field.channels"), select("ramChannels", RAM_CHANNELS.map((c) => [c, `${c} (${ramBandwidthGbps(cfg.ramKind, c).toFixed(0)} GB/s)`]), cfg.ramChannels, { "data-focus-id": `${i}-ch` }))));
  }

  const backends = backendChoices(cfg);
  if (backends.length > 1) {
    settings.append(h("div", { class: "fields" }, field(t("field.backend"), select("backend",
      backends.map((b) => [b, b ? BACKEND_NAMES[b] : t("backend.default")]), cfg.backend, { "data-focus-id": `${i}-backend` }))));
  }

  settings.append(h("h3", {}, t("section.model")));
  settings.append(h("div", { class: "fields" },
    field(t("field.model"), select("model", P.models.map((m) => [m.name, modelLabel(m)]), cfg.model, { "data-focus-id": `${i}-model` }))));
  settings.append(h("div", { class: "fields" },
    field(t("field.quant"), select("quant", Object.entries(P.quant_bpw).map(([q, b]) => [q, `${q} (${b} bpw)`]), cfg.quant, { "data-focus-id": `${i}-quant` })),
    field(t("field.context"), select("context", CONTEXTS.map((c) => [c, t("option.context", { k: c / 1024 })]), cfg.context, { "data-focus-id": `${i}-ctx` })),
    field(t("field.kvCache"), select("kvBits", KV_BITS.map((b) => [b, b === 16 ? "F16" : `Q${b}`]), cfg.kvBits, { "data-focus-id": `${i}-kv` }))));
  settings.append(h("div", { class: "fields" },
    field(t("field.parallel"), select("parallel", PARALLEL.map((n) => [n, n === 1 ? t("option.parallelOne") : String(n)]), cfg.parallel, { "data-focus-id": `${i}-parallel` }))));

  panel.append(h("div", { class: "config-body" }, settings, renderResult(cfg)));
  return panel;
}

function renderResult(cfg) {
  let res;
  try {
    res = compute(cfg);
  } catch (e) {
    return h("div", { class: "result error" }, e.message);
  }
  const { system, est } = res;
  const gpuLayers = est.placements.filter((p) => p.is_gpu).reduce((a, p) => a + p.layers, 0);
  const n = est.model.n_layers;
  // Only problems get a status line; the memory bars already show a model that fits.
  let badge = null;
  if (!est.fits) badge = h("p", { class: "badge bad" }, t("badge.noFit"));
  else if (!system.gpus.length) badge = h("p", { class: "badge warn" }, t("badge.cpuOnly"));
  else if (gpuLayers < n) badge = h("p", { class: "badge warn" }, t("badge.layersOnCpu", { cpu: n - gpuLayers, total: n }));

  const timing = timingInfo(est, { prompt_tokens: state.prompt, max_tokens: state.maxTokens, depth: state.depth });
  const slots = cfg.parallel;
  const notes = localizedNotes(cfg, system, est);
  const result = h("div", { class: "result" },
    badge,
    h("div", { class: "headline" },
      h("span", { class: "big" }, fmtTps(est.decode_tps_empty)),
      h("span", { class: "unit" }, t("result.unit"))),
    h("div", { class: "metrics" },
      metric(t("metric.fullContext"), t("result.tps", { v: fmtTps(est.decode_tps_full) })),
      metric(t("metric.prefill"), t("result.tps", { v: fmtTps(est.prefill_tps) })),
      slots > 1 ? metric(t("metric.each", { n: slots }), t("result.tps", { v: fmtTps(est.parallelTps(slots, 0)) })) : null,
      slots > 1 ? metric(t("metric.total", { n: slots }), t("result.tps", { v: fmtTps(slots * est.parallelTps(slots, 0)) })) : null,
      slots > 1
        ? metric(t("metric.firstTokenAll", { n: fmtInt(state.prompt), k: slots }), fmtTime(timing.ttft_all_s))
        : metric(t("metric.firstToken", { n: fmtInt(state.prompt) }), fmtTime(timing.ttft_s)),
      slots > 1
        ? metric(t("metric.replyAll", { n: fmtInt(state.maxTokens), k: slots }), fmtTime(timing.reply_all_s))
        : metric(t("metric.reply", { n: fmtInt(state.maxTokens) }), fmtTime(timing.reply_s))),
    renderBars(system, est),
    renderParallel(cfg, est),
    notes.length ? h("ul", { class: "notes" }, notes.map((note) => h("li", {}, note))) : null,
    renderWarnings(system, est),
    renderMeasured(cfg, system),
    renderParallelMeasured(cfg, system, est));
  return result;
}

// Speeds with 1, 2, 4, ... requests at once, up to the configured slots.
function renderParallel(cfg, est) {
  if (cfg.parallel === 1) return null;
  const rows = parallelLevels(cfg.parallel).map((n) => {
    const each = est.parallelTps(n, 0);
    return h("tr", {}, h("td", {}, String(n)), h("td", {}, fmtTps(each)), h("td", {}, fmtTps(n * each)));
  });
  return h("details", { class: "measured-wrap" },
    h("summary", {}, t("parallel.summary", { n: cfg.parallel })),
    h("div", { class: "measured" },
      h("p", { style: "margin:0 0 6px" }, t("parallel.intro", {
        context: fmtInt(cfg.context), n: cfg.parallel, total: cfg.context * cfg.parallel })),
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, t("parallel.requests")), h("th", {}, t("parallel.each")), h("th", {}, t("parallel.total")))),
        h("tbody", {}, rows))));
}

// llama-batched-bench runs on the selected GPU or chip, measured vs. simulated.
function renderParallelMeasured(cfg, system, est) {
  const check = parallelChecks(P, system, est, cfg.kvBits);
  if (!check) return null;
  const body = check.rows.map((r) => h("tr", {},
    h("td", {}, `${r.model} · ${r.quant}`, h("span", { class: "run-meta" }, `${r.backend} · pp${r.n_prompt} tg${r.n_gen}`)),
    h("td", {}, String(r.requests)), h("td", {}, fmtTps(r.tg_measured)), h("td", {}, fmtTps(r.tg_simulated))));
  return h("details", { class: "measured-wrap" },
    h("summary", {}, t("measured.parallelSummary", { n: check.rows.length })),
    h("div", { class: "measured" },
      h("p", { style: "margin:0 0 6px" }, `${t("measured.parallelIntro")} `, h("a", { href: check.sources[0] }, t("measured.source"))),
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, t("measured.model")), h("th", {}, t("parallel.requests")),
          h("th", {}, t("measured.genReal")), h("th", {}, t("measured.genSim")))),
        h("tbody", {}, body))));
}

// The estimator's notes are English; rebuild them from its numbers instead.
function localizedNotes(cfg, system, est) {
  const notes = [];
  const n = est.model.n_layers;
  const cpu = est.placements.find((p) => !p.is_gpu);
  if (cfg.context > est.model.max_context) {
    notes.push(t("note.context", { context: fmtInt(cfg.context), max: fmtInt(est.model.max_context) }));
  }
  if (cpu && !est.fits) notes.push(t("note.ram", { need: gb(est.ram_needed_gb), free: gb(est.ram_free_gb) }));
  // When the model fits, the status line already says how many layers run on the CPU.
  if (cpu && system.gpus.length && !est.fits) notes.push(t("note.offload", { cpu: cpu.layers, total: n }));
  return notes;
}

// Locale keys for the report's warning codes.
const WARNING_KEYS = {
  prefill_device_uncalibrated: "warn.prefillDevice",
  cpu_device_uncalibrated: "warn.cpuDevice",
  backend_uncalibrated: "warn.uncalibrated",
  decode_uncalibrated: "warn.decodeUncalibrated",
  prefill_uncalibrated: "warn.prefillUncalibrated",
  moe_routing: "warn.moe",
  parallel_uncalibrated: "warn.parallelUncalibrated",
  parallel_device_uncalibrated: "warn.parallelDevice",
  moe_parallel: "warn.moeParallel",
};

// Point out where the numbers rest on uncalibrated assumptions rather than measurements.
function renderWarnings(system, est) {
  const items = warnings(P, system, est).map((w) => t(WARNING_KEYS[w.code], w.params));
  if (!items.length) return null;
  return h("div", { class: "warnings", role: "note" },
    h("strong", {}, t("warn.title")), h("ul", {}, items.map((x) => h("li", {}, x))));
}

function metric(k, v) {
  return h("div", { class: "metric" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v));
}

function bar(label, capacity, segments, over = false) {
  const total = Math.max(capacity, segments.reduce((a, s) => a + s.v, 0));
  const used = segments.reduce((a, s) => a + s.v, 0);
  return h("div", { class: "bar-row" },
    h("div", { class: "bar-label" }, h("span", {}, label), h("span", {}, `${gb(used)} / ${gb(capacity)}`)),
    h("div", { class: `bar${over ? " over" : ""}`, role: "img", "aria-label": t("bar.aria", { label, used: gb(used), capacity: gb(capacity) }) },
      segments.map((s) => h("span", { class: s.cls, style: `width:${(100 * s.v) / total}%` }))));
}

function renderBars(system, est) {
  const eff = P.efficiency;
  const rows = [];
  const cpu = est.placements.find((p) => !p.is_gpu);
  if (system.unified) {
    const g = est.placements.find((p) => p.is_gpu);
    const segs = [];
    if (g) segs.push({ cls: "w", v: g.weights_gb }, { cls: "kv", v: g.kv_gb });
    if (cpu) segs.push({ cls: "cpu", v: cpu.weights_gb + cpu.kv_gb });
    rows.push(bar(t("bar.unified", { limit: gb(system.ram.size_gb * eff.unified_gpu_fraction) }),
                  system.ram.size_gb, segs, !est.fits));
  } else {
    let k = 0;
    system.gpus.forEach((gpu, j) => {
      const p = est.placements[k];
      const segs = [];
      if (p && p.is_gpu && p.device === gpu.name) {
        segs.push({ cls: "w", v: p.weights_gb }, { cls: "kv", v: p.kv_gb });
        k++;
      }
      rows.push(bar(system.gpus.length > 1 ? t("bar.vramN", { n: j + 1 }) : t("bar.vram"), gpu.vram_gb, segs));
    });
    rows.push(bar(t("bar.ram", { reserve: gb(eff.ram_reserve_gb) }), system.ram.size_gb,
                  cpu ? [{ cls: "cpu", v: cpu.weights_gb + cpu.kv_gb }] : [], !est.fits));
  }
  // Only the colors the bars actually use.
  const onGpu = est.placements.some((p) => p.is_gpu);
  return h("div", {},
    h("div", { class: "bars" }, rows),
    h("div", { class: "legend" },
      onGpu ? h("span", {}, h("i", { style: "background:var(--bar-gpu)" }), t("legend.weights")) : null,
      onGpu ? h("span", {}, h("i", { style: "background:var(--bar-kv)" }), t("legend.kv")) : null,
      cpu ? h("span", {}, h("i", { style: "background:var(--bar-cpu)" }), t("legend.cpu")) : null));
}

function renderMeasured(cfg, system) {
  if (isPcLike(cfg) && !cfg.gpus.length) return renderCpuMeasured(cfg, system);
  const check = gpuChecks(P, system, cfg.kvBits);
  if (!check) return null;
  const body = check.rows.map((r) => {
    const fa = r.flash_attn ? " · FA" : "";
    return h("tr", {},
      h("td", {}, `${r.backend} · ${r.quant}${fa}`, r.old_build ? h("span", { class: "old-build" }, ` ${t("measured.oldBuild")}`) : null),
      h("td", {}, fmtTps(r.tg_measured)), h("td", {}, fmtTps(r.tg_simulated)),
      h("td", {}, fmtTps(r.pp_measured)), h("td", {}, fmtTps(r.pp_simulated)));
  });
  return measuredTable(check.rows.length,
    h("p", { style: "margin:0 0 6px" }, `${t("measured.intro")} `, h("a", { href: check.sources[0] }, t("measured.source"))),
    t("measured.quant"), body,
    check.rows.some((r) => r.old_build) ? h("p", { class: "old-build" }, t("measured.oldBuildNote")) : null);
}

function measuredTable(n, intro, firstHead, body, footer) {
  return h("details", { class: "measured-wrap" },
    h("summary", {}, t("measured.summary", { n })),
    h("div", { class: "measured" },
      intro,
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, firstHead), h("th", {}, t("measured.genReal")),
          h("th", {}, t("measured.genSim")), h("th", {}, t("measured.promptReal")), h("th", {}, t("measured.promptSim")))),
        h("tbody", {}, body)),
      footer));
}

// CPU-only runs on the selected CPU. Each row is simulated with the memory that run used,
// not the RAM chosen in the form.
function renderCpuMeasured(cfg, system) {
  const check = cpuChecks(P, system);
  if (!check) return null;
  const body = check.rows.map((r) => {
    const memory = r.dimms && r.mem_mts ? `${r.mem_mts} MT/s × ${r.dimms}` : t("measured.memUnknown");
    const test = [r.n_prompt ? `pp${r.n_prompt}` : null, r.n_gen ? `tg${r.n_gen}` : null].filter(Boolean).join(" ");
    const meta = [memory, r.threads ? t("measured.threads", { n: r.threads }) : null, test,
                  `×${r.runs}`, `[${r.sources.join(",")}]`].filter(Boolean).join(" · ");
    return h("tr", {},
      h("td", {}, `${r.model} · ${r.quant}`, h("span", { class: "run-meta" }, meta)),
      h("td", {}, fmtTps(r.tg_measured)), h("td", {}, fmtTps(r.tg_simulated)),
      h("td", {}, fmtTps(r.pp_measured)), h("td", {}, fmtTps(r.pp_simulated)));
  });
  const key = `${cfg.cpu}|cpu`;
  const fitted = key in P.efficiency.device_bandwidth || key in P.efficiency.device_compute;
  return measuredTable(check.rows.length,
    h("p", { style: "margin:0 0 6px" }, t("measured.cpuIntro"), fitted ? ` ${t("measured.cpuFitted")}` : null),
    t("measured.model"), body,
    h("p", { class: "sources" }, `${t("measured.sources")} `,
      check.sources.flatMap((url, i) => [i ? " " : null, h("a", { href: url }, `[${i + 1}]`)])));
}

// ---------- report export ----------

function downloadReport(format) {
  readMockInputs();
  saveState();
  const timing = { prompt_tokens: state.prompt, max_tokens: state.maxTokens, depth: state.depth };
  const configs = state.configs.map((cfg, i) => ({
    label: state.configs.length > 1 ? LETTERS[i] : null, system: systemSpec(cfg), model: cfg.model,
    quant: cfg.quant, context: cfg.context, kv_bits: cfg.kvBits, parallel: cfg.parallel, timing,
  }));
  const rep = buildReport(P, configs, "web", location.href);
  const [body, type, ext] = format === "json"
    ? [`${JSON.stringify(rep, null, 2)}\n`, "application/json", "json"]
    : [markdown(rep), "text/markdown", "md"];
  const url = URL.createObjectURL(new Blob([body], { type: `${type};charset=utf-8` }));
  const a = h("a", { href: url, download: `llm-sim-bench-report.${ext}` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// ---------- events ----------

function onConfigEvent(e) {
  const panel = e.target.closest(".config");
  const act = e.target.closest("[data-act]");
  if (e.type === "click" && act?.dataset.act === "add-config") {
    state.configs.push({ ...state.configs[state.configs.length - 1] });
    return changed();
  }
  if (e.type === "click" && act?.dataset.act === "report") return downloadReport(act.dataset.v);
  if (!panel) return;
  const i = Number(panel.dataset.i);
  const cfg = state.configs[i];
  if (e.type === "click" && act) {
    const j = Number(act.dataset.j);
    switch (act.dataset.act) {
      case "platform": switchPlatform(cfg, act.dataset.v); break;
      case "add-gpu": cfg.gpus.push(cfg.gpus[cfg.gpus.length - 1] || gpusFor(cfg.platform, cfg.showDatacenter)[0].name); break;
      case "remove-gpu": cfg.gpus.splice(j, 1); break;
      case "remove-config": state.configs.splice(i, 1); break;
      default: return;
    }
    cfg.gpus = [...cfg.gpus];
    return changed();
  }
  if (e.type === "change" && e.target.dataset.k) {
    const k = e.target.dataset.k;
    const v = e.target.value;
    if (k === "showDatacenter") {
      cfg.showDatacenter = e.target.checked;
      switchPlatform(cfg, "pc");  // drops data-center cards when unchecked, keeping one GPU
    } else if (k === "gpu") {
      cfg.gpus = cfg.gpus.map((g, j) => (j === Number(e.target.dataset.j) ? v : g));
    } else if (k === "cpu") {
      setCpu(cfg, v);
    } else {
      cfg[k] = ["ramGb", "ramChannels", "memoryGb", "context", "kvBits"].includes(k) ? Number(v) : v;
    }
    state.configs[i] = sanitizeConfig(cfg);
    changed();
  }
}

function changed() {
  if (document.getElementById("part-search").value) renderSearch();
  stopPlayback();
  state.configs = state.configs.map((c) => sanitizeConfig({ ...c, gpus: [...c.gpus] }));
  saveState();
  renderConfigs();
}

// ---------- mock streams ----------

const textCache = new Map();

// The viewer's own text lives only in this browser (localStorage), never in the URL.
const CUSTOM = "custom";
const CUSTOM_KEY = "llmsim.customText";
let customText = "";

function loadCustomText() {
  try {
    return localStorage.getItem(CUSTOM_KEY) || "";
  } catch {
    return "";
  }
}

function saveCustomText() {
  try {
    if (customText) localStorage.setItem(CUSTOM_KEY, customText);
    else localStorage.removeItem(CUSTOM_KEY);
  } catch { /* storage may be blocked or full; the text then lasts for this page only */ }
}

function textEntry(id = state.text) {
  if (id === CUSTOM) return { id, kind: "answer", origin: "custom", lang: guessLang(customText) || lang };
  return TEXTS.texts.find((x) => x.id === id);
}

function defaultTextId() {
  return textEntry(`${lang}-answer`) ? `${lang}-answer` : TEXTS.texts[0].id;
}

async function loadText(entry) {
  if (entry.id === CUSTOM) return customText;
  if (!textCache.has(entry.id)) {
    const resp = await fetch(`data/texts/${entry.file}`);
    if (!resp.ok) throw new Error(`${entry.file}: HTTP ${resp.status}`);
    textCache.set(entry.id, entry.kind === "chat" ? (await resp.json()).turns : await resp.text());
  }
  return textCache.get(entry.id);
}

function renderTextPicker() {
  const picker = document.getElementById("text");
  picker.replaceChildren(...Object.entries(LANGS).map(([code, name]) =>
    h("optgroup", { label: name },
      TEXTS.texts.filter((x) => x.lang === code).map((x) =>
        h("option", { value: x.id, selected: x.id === state.text },
          x.author ? `${x.title} (${x.author})` : x.title)))),
    h("option", { value: CUSTOM, selected: state.text === CUSTOM }, t("mock.custom.option")));
  renderCredits();
}

function renderCredits() {
  const entry = textEntry();
  const el = document.getElementById("credits");
  document.getElementById("custom-text").hidden = entry.origin !== "custom";
  if (entry.origin === "custom") {
    el.replaceChildren(t("mock.custom.note"));
    return;
  }
  if (entry.origin === "authored") {
    el.replaceChildren(t("mock.authored"));
    return;
  }
  const notes = CREDITS[entry.id]?.notes;
  el.replaceChildren(
    t("mock.credit", { title: entry.title, author: entry.author, year: entry.first_published,
                       died: entry.author_died }), " · ",
    h("a", { href: entry.source.page }, entry.source.name),
    ...(notes ? [h("details", { class: "credit-notes" }, h("summary", {}, t("mock.creditNotes")), notes)] : []));
}

// Custom text borrows the ratio of the sample answer in the language it looks like.
function tokenizerFor(cfg) {
  const m = P.models.find((x) => x.name === cfg.model);
  const ref = state.text === CUSTOM ? textEntry().lang : null;
  const r = charsPerToken(P.tokenizers, ref ? `${ref}-answer` : state.text, m.tokenizer);
  return { family: m.tokenizer, ref, ...r };
}

function tokenizerLabel(tk) {
  const cpt = tk.value.toFixed(2);
  if (!tk.measured) return t("mock.tokenizerUnknown", { cpt });
  return tk.ref ? t("mock.tokenizerCustom", { family: tk.family, cpt, lang: LANGS[tk.ref] })
    : t("mock.tokenizer", { family: tk.family, cpt });
}

function renderStreams() {
  const root = document.getElementById("streams");
  root.replaceChildren(...state.configs.map((cfg, i) => {
    let label = "";
    let tok = "";
    try {
      const m = P.models.find((x) => x.name === cfg.model);
      const hw = !isPcLike(cfg) ? chipName(activeChip(cfg))
        : (cfg.gpus.length ? cfg.gpus.map(gpuName).join(" + ") : t("mock.cpuOnly"));
      label = `${hw} · ${m.name} ${cfg.quant}`
        + (cfg.parallel > 1 ? ` · ${t("mock.requests", { n: cfg.parallel })}` : "");
      tok = tokenizerLabel(tokenizerFor(cfg));
    } catch { /* invalid configs show an error in their panel */ }
    return h("div", { class: "stream", "data-i": i },
      h("div", { class: "stream-head" },
        h("strong", {}, state.configs.length > 1 ? h("span", { class: "tag" }, `[${LETTERS[i]}] `) : null, label),
        h("span", { class: "tok" }, tok),
        h("span", { class: "stats" }),
        h("span", { class: "problem" }, contextProblem(cfg, state.depth + state.prompt + state.maxTokens))),
      h("div", { class: "stream-out waiting", "aria-live": "off" }, t("mock.ready")));
  }));
  document.getElementById("depth-note").hidden = !state.depth;
}

// Only a problem gets a line: a run that needs more context than the configuration has.
function contextProblem(cfg, need) {
  return need > cfg.context
    ? t("mock.overContext", { need: fmtInt(need), context: fmtInt(cfg.context) }) : "";
}

function renderDepthPicker() {
  document.getElementById("depth").replaceChildren(...DEPTHS.map((d) => h("option",
    { value: String(d), selected: d === state.depth }, d ? t("option.context", { k: d / 1024 }) : t("mock.depthNone"))));
}

async function startPlayback() {
  const entry = textEntry();
  const content = await loadText(entry);
  if (entry.id === CUSTOM && !content.trim()) {
    for (const el of document.querySelectorAll(".stream .stats")) el.textContent = t("mock.custom.empty");
    return;
  }
  const streams = [];
  state.configs.forEach((cfg, i) => {
    const el = document.querySelector(`.stream[data-i="${i}"]`);
    try {
      const { est } = compute(cfg);
      const tk = tokenizerFor(cfg);
      const turns = entry.kind === "chat"
        ? content.map((x) => ({ user: chunkText(x.user, tk.value), assistant: chunkText(x.assistant, tk.value) }))
        : [{ user: null, assistant: chunkText(content, tk.value) }];
      // With several slots, play one of n requests that arrive together: their prompts
      // share prompt processing and every decode step serves all of them.
      const n = cfg.parallel;
      const schedule = new Schedule({
        turns, promptTokens: state.prompt, maxTokens: state.maxTokens, cachedTokens: state.depth,
        prefillTps: est.prefill_tps / n, decodeTps: (fill) => est.parallelTps(n, fill), seed: 7 + i });
      streams.push({ el, est, shown: 0, block: null, chat: entry.kind === "chat", schedule });
      // The real text length is known now; a conversation can need more than one reply.
      el.querySelector(".problem").textContent = contextProblem(cfg, schedule.events.at(-1)?.fill ?? 0);
      const out = el.querySelector(".stream-out");
      out.replaceChildren();
      out.classList.remove("waiting");
      if (state.depth) out.append(h("div", { class: "cached" }, t("mock.depthCached", { n: fmtInt(state.depth) })));
    } catch (e) {
      el.querySelector(".stats").textContent = e.message;
    }
  });
  playback = { streams, start: performance.now(), raf: 0 };
  setPlaying(true);
  tick();
}

function stopPlayback() {
  if (!playback) return;
  cancelAnimationFrame(playback.raf);
  playback = null;
  setPlaying(false);
}

function setPlaying(on) {
  const button = document.getElementById("play");
  button.textContent = t(on ? "mock.stop" : "mock.play");
  button.classList.toggle("playing", on);
  document.getElementById("streams").classList.toggle("live", on);
}

// Append events [s.shown, n) to the output, one block per message.
function appendEvents(s, n) {
  const out = s.el.querySelector(".stream-out");
  out.classList.remove("waiting");
  out.querySelector(".pending")?.remove();
  for (const ev of s.schedule.events.slice(s.shown, n)) {
    if (ev.role === "user" || !s.block || s.block.dataset.turn !== String(ev.turn)) {
      const role = ev.role === "user" ? "user" : "assistant";
      s.block = h("div", { class: `msg ${role}`, "data-turn": ev.turn },
        s.chat ? h("span", { class: "who" }, t(role === "user" ? "mock.you" : "mock.ai")) : null,
        h("span", { class: "body" }));
      out.append(s.block);
      if (ev.role === "user") {
        s.block.querySelector(".body").textContent = ev.text;
        s.block = null;
        continue;
      }
    }
    s.block.querySelector(".body").textContent += ev.text;
  }
  s.shown = n;
  out.scrollTop = out.scrollHeight;
}

function tick() {
  if (!playback) return;
  const elapsed = (performance.now() - playback.start) / 1000;
  let running = false;
  for (const s of playback.streams) {
    const sch = s.schedule;
    const n = sch.dueBy(elapsed);
    if (n > s.shown) appendEvents(s, n);
    const k = Math.max(0, sch.turns.findLastIndex((x) => x.start <= elapsed));
    const turn = sch.turns[k];
    const out = s.el.querySelector(".stream-out");
    const stats = s.el.querySelector(".stats");
    const turnLabel = sch.turns.length > 1 ? `${t("mock.turn", { k: k + 1, total: sch.turns.length })} · ` : "";
    if (elapsed < turn.firstAt) {
      let pending = out.querySelector(".pending");
      if (!pending) {
        pending = h("div", { class: "pending" });
        out.append(pending);
      }
      pending.textContent = t("mock.prefilling", {
        n: fmtInt(turn.prefillTokens), left: fmtTime(Math.max(0, turn.firstAt - elapsed)) });
      stats.textContent = turnLabel + t("mock.prefillStat", { v: fmtTps(s.est.prefill_tps / s.est.parallel) });
      running = true;
    } else {
      const now = Math.min(elapsed, turn.end);
      const done = sch.events.slice(0, n).filter((ev) => ev.turn === k && ev.role === "assistant");
      const genElapsed = now - turn.firstAt;
      const rate = done.length > 1 && genElapsed > 0 ? (done.length - 1) / genElapsed : 0;
      const chars = done.slice(1).reduce((a, ev) => a + Array.from(ev.text).length, 0);
      const fill = n ? sch.events[n - 1].fill : state.depth + state.prompt;
      // With a pre-filled context, show how much slower generation is than when empty.
      const p = s.est.parallel;
      const slower = state.depth ? Math.round(100 * (1 - s.est.parallelTps(p, fill) / s.est.parallelTps(p, 0))) : 0;
      stats.textContent = turnLabel + t("mock.stats", {
        ttft: fmtTime(turn.ttft), n: done.length, total: turn.tokens,
        rate: rate ? fmtTps(rate) : "–", cps: rate ? fmtTps(chars / genElapsed) : "–",
        fill: fmtInt(fill),
      }) + (slower > 0 ? ` · ${t("mock.depthSlowdown", { pct: slower })}` : "");
    }
    if (n < sch.events.length) running = true;
  }
  if (running) playback.raf = requestAnimationFrame(tick);
  else stopPlayback();
}

function readMockInputs() {
  const clamp = (el, lo, hi) => Math.min(hi, Math.max(lo, Math.round(Number(el.value) || lo)));
  state.prompt = clamp(document.getElementById("prompt-tokens"), 1, 200000);
  state.maxTokens = clamp(document.getElementById("max-tokens"), 1, 4000);
  state.depth = Number(document.getElementById("depth").value) || 0;
}

// ---------- part search ----------

function partMeta({ kind, item }) {
  if (kind === "gpu") {
    return `${item.vendor} · ${t(`gpu.segment.${item.segment}`)} · ${item.vram_gb} GB · ${item.bandwidth_gbps} GB/s`;
  }
  if (kind === "cpu") {
    return `${item.vendor} · ${t(`cpu.segment.${item.segment}`)} · ${t("search.cores", { n: item.cores })}` +
      ` · ${item.mem_kind.toUpperCase()} × ${item.mem_channels}`;
  }
  return `${item.vendor} · ${item.bandwidth_gbps} GB/s · ${Math.max(...item.memory_options_gb)} GB`;
}

function renderSearch() {
  const input = document.getElementById("part-search");
  const list = document.getElementById("search-results");
  const query = input.value.trim();
  if (!query) {
    list.hidden = true;
    list.replaceChildren();
    return;
  }
  const hits = searchParts(query, P);
  list.hidden = false;
  if (!hits.length) {
    list.replaceChildren(h("li", { class: "search-empty" }, t("search.none")));
    return;
  }
  const targets = state.configs.map((_, i) => i);
  list.replaceChildren(...hits.map((hit) => h("li", {},
    h("div", { class: "search-hit" },
      h("strong", {}, hit.item.label || hit.item.name),
      h("span", { class: "search-meta" }, `${t(`search.kind.${hit.kind}`)} · ${partMeta(hit)}`)),
    h("div", { class: "search-actions" }, targets.map((i) => h("button", {
      type: "button", "data-kind": hit.kind, "data-name": hit.item.name, "data-i": i,
    }, targets.length > 1 ? t("search.applyTo", { letter: LETTERS[i] }) : t("search.apply")))))));
}

// Put a part into configuration i, switching to the tab that lists it.
function applyPart(i, kind, name) {
  const cfg = state.configs[i];
  if (kind === "gpu") {
    const gpu = P.gpus.find((g) => g.name === name);
    const keepPc = cfg.platform === "pc" && cfg.showDatacenter;
    switchPlatform(cfg, gpu.segment === "datacenter" && !keepPc ? "server" : "pc");
    cfg.gpus = [name, ...cfg.gpus.slice(1)];
  } else if (kind === "cpu") {
    const cpu = P.cpus.find((c) => c.name === name);
    switchPlatform(cfg, cpu.segment === "server" ? "server" : "pc");
    setCpu(cfg, name);
  } else {
    const chip = P.unified.find((a) => a.name === name);
    switchPlatform(cfg, chip.vendor === "Apple" ? "apple" : "unified");
    if (chip.vendor === "Apple") cfg.chip = name;
    else cfg.soc = name;
  }
  document.getElementById("part-search").value = "";
  renderSearch();
  changed();
}

// ---------- boot ----------

function renderStatic() {
  document.documentElement.lang = lang;
  translateDom();
  document.getElementById("play").textContent = t(playback ? "mock.stop" : "mock.play");
  const list = ["cuda", "rocm", "vulkan", "sycl", "metal", "cpu"].filter((b) => P.calibration[`${b}/tg`])
    .map((b) => {
      const c = P.calibration[`${b}/tg`];
      return `${BACKEND_NAMES[b]} ${(c.median * 100).toFixed(1)}% (${fmtInt(c.n)})`;
    });
  document.getElementById("calibration").textContent = t("about.calibration.stats", { list: list.join(", ") });
}

async function switchLang(code) {
  stopPlayback();
  saveLang(code);
  const url = new URL(location.href);
  if (url.searchParams.has("lang")) {
    url.searchParams.set("lang", code);
    history.replaceState(null, "", url);
  }
  await loadLocale(code);
  renderStatic();
  renderTextPicker();
  renderDepthPicker();
  renderConfigs();
}

async function main() {
  const code = pickLang(location.search, savedLang(), navigator.languages || [navigator.language]);
  const json = (url) => fetch(url).then((r) => r.json());
  [P, TEXTS, CREDITS] = await Promise.all([
    json("data/presets.json"), json("data/texts/manifest.json"), json("data/texts/credits.json"),
    loadLocale(code)]);
  state = loadState();
  customText = loadCustomText();

  const picker = document.getElementById("lang");
  picker.replaceChildren(...Object.entries(LANGS).map(([c, name]) => h("option", { value: c, selected: c === code }, name)));
  picker.addEventListener("change", () => switchLang(picker.value));
  renderStatic();
  renderTextPicker();
  const search = document.getElementById("part-search");
  search.addEventListener("input", renderSearch);
  search.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      search.value = "";
      renderSearch();
    } else if (e.key === "Enter") {
      document.querySelector("#search-results button")?.click();
    }
  });
  document.getElementById("search-results").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-kind]");
    if (b) applyPart(Number(b.dataset.i), b.dataset.kind, b.dataset.name);
  });
  document.getElementById("text").addEventListener("change", (e) => {
    state.text = e.target.value;
    renderCredits();
    changed();
  });
  const customInput = document.getElementById("custom-input");
  customInput.value = customText;
  customInput.addEventListener("input", () => {
    customText = customInput.value;
    saveCustomText();
    stopPlayback();
    renderStreams();
  });
  document.getElementById("custom-clear").addEventListener("click", () => {
    customInput.value = "";
    customInput.dispatchEvent(new Event("input"));
    customInput.focus();
  });
  document.getElementById("prompt-tokens").value = state.prompt;
  document.getElementById("max-tokens").value = state.maxTokens;

  const configs = document.getElementById("configs");
  configs.addEventListener("click", onConfigEvent);
  configs.addEventListener("change", onConfigEvent);
  renderDepthPicker();
  for (const id of ["depth", "prompt-tokens", "max-tokens"]) {
    document.getElementById(id).addEventListener("change", () => {
      readMockInputs();
      changed();
    });
  }
  document.getElementById("play").addEventListener("click", () => {
    if (playback) return stopPlayback();
    readMockInputs();
    saveState();
    startPlayback().catch((e) => {
      document.querySelector(".stream .stats").textContent = e.message;
    });
  });
  window.addEventListener("hashchange", () => {
    stopPlayback();
    state = loadState();
    renderTextPicker();
    renderDepthPicker();
    renderConfigs();
  });

  renderConfigs();
}

main().catch((e) => {
  const msg = t("app.error", { msg: e.message }).replace(/^app\.error$/,
    `Could not load the app data (${e.message}). Reload the page. When running it from a local copy, serve it over HTTP with: llmsimbench web`);
  document.querySelector("main").replaceChildren(h("p", { class: "panel error" }, msg));
  if (!document.getElementById("lang").options.length) document.querySelector(".lang-picker").style.display = "none";
});
