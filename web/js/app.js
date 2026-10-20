import { buildSystem, estimate, ramBandwidthGbps } from "./estimator.js";
import { MockStream } from "./mock.js";
import { decodeState, encodeState } from "./state.js";

const RAM_SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256];
const RAM_KINDS = ["ddr4-2666", "ddr4-3200", "ddr4-3600", "ddr5-4800", "ddr5-5600",
                   "ddr5-6000", "ddr5-6400", "ddr5-8000"];
const RAM_CHANNELS = [1, 2, 4, 8];
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
const KV_BITS = [16, 8, 4];
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
  memoryGb: 64,
  model: "qwen3-32b",
  quant: "Q4_K_M",
  context: 8192,
  kvBits: 16,
};
const DEFAULT_STATE = {
  configs: [DEFAULT_CONFIG, { ...DEFAULT_CONFIG, platform: "apple" }],
  prompt: 1000,
  maxTokens: 300,
};

let P; // presets.json
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

export function fmtTps(x) {
  if (!Number.isFinite(x)) return "–";
  if (x >= 1000) return Math.round(x).toLocaleString("en-US");
  if (x >= 100) return x.toFixed(0);
  if (x >= 10) return x.toFixed(1);
  return x.toFixed(2);
}

export function fmtTime(s) {
  if (!Number.isFinite(s)) return "–";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m < 60 ? `${m}m ${String(sec).padStart(2, "0")}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

const gb = (x) => `${x.toFixed(1)} GB`;

function modelLabel(m) {
  const size = m.active_params_b ? `${m.params_b}B, ${m.active_params_b}B active` : `${m.params_b}B`;
  return `${m.name} (${size})`;
}

// "apple-m4-max-40c" -> "M4 Max"
function chipName(a) {
  const [gen, ...tier] = a.name.replace(/^apple-/, "").replace(/-\d+c$/, "").split("-");
  return [gen.toUpperCase(), ...tier.map((t) => t[0].toUpperCase() + t.slice(1))].join(" ");
}

function chipLabel(a) {
  return `${chipName(a)} · ${a.gpu_cores}-core GPU · ${a.bandwidth_gbps} GB/s`;
}

// "rtx4070tisuper-16gb" -> "RTX 4070 Ti SUPER"
function gpuName(name) {
  const m = /^rtx(\d+)(ti)?(super)?-/.exec(name);
  return m ? `RTX ${m[1]}${m[2] ? " Ti" : ""}${m[3] ? " SUPER" : ""}` : name;
}

function gpuLabel(g) {
  return `${gpuName(g.name)} · ${g.vram_gb} GB`;
}

// ---------- state ----------

function sanitizeConfig(c) {
  const cfg = { ...DEFAULT_CONFIG, ...(c && typeof c === "object" ? c : {}) };
  const has = (list, name) => list.some((x) => x.name === name);
  cfg.platform = cfg.platform === "apple" ? "apple" : "pc";
  cfg.gpus = Array.isArray(cfg.gpus) ? cfg.gpus.filter((g) => has(P.gpus, g)).slice(0, 8) : [];
  if (!has(P.cpus, cfg.cpu)) cfg.cpu = DEFAULT_CONFIG.cpu;
  if (!RAM_SIZES.includes(Number(cfg.ramGb))) cfg.ramGb = DEFAULT_CONFIG.ramGb;
  if (!RAM_KINDS.includes(cfg.ramKind)) cfg.ramKind = DEFAULT_CONFIG.ramKind;
  if (!RAM_CHANNELS.includes(Number(cfg.ramChannels))) cfg.ramChannels = DEFAULT_CONFIG.ramChannels;
  if (!has(P.apple, cfg.chip)) cfg.chip = DEFAULT_CONFIG.chip;
  const chip = P.apple.find((a) => a.name === cfg.chip);
  if (!chip.memory_options_gb.includes(Number(cfg.memoryGb))) cfg.memoryGb = Math.max(...chip.memory_options_gb);
  if (!has(P.models, cfg.model)) cfg.model = DEFAULT_CONFIG.model;
  if (!(cfg.quant in P.quant_bpw)) cfg.quant = DEFAULT_CONFIG.quant;
  if (!CONTEXTS.includes(Number(cfg.context))) cfg.context = DEFAULT_CONFIG.context;
  if (!KV_BITS.includes(Number(cfg.kvBits))) cfg.kvBits = DEFAULT_CONFIG.kvBits;
  for (const k of ["ramGb", "ramChannels", "memoryGb", "context", "kvBits"]) cfg[k] = Number(cfg[k]);
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
  };
}

function saveState() {
  history.replaceState(null, "", `#${encodeState(state)}`);
}

// ---------- estimate ----------

function systemSpec(cfg) {
  if (cfg.platform === "apple") return { kind: "apple", chip: cfg.chip, memory_gb: cfg.memoryGb };
  return {
    kind: "pc", cpu: cfg.cpu, ram_gb: cfg.ramGb,
    ram_bandwidth_gbps: ramBandwidthGbps(cfg.ramKind, cfg.ramChannels), gpus: cfg.gpus,
  };
}

function compute(cfg, model = cfg.model, quant = cfg.quant, context = cfg.context) {
  const system = buildSystem(systemSpec(cfg), P);
  const m = P.models.find((x) => x.name === model);
  const est = estimate(system, m, quant, context, cfg.kvBits, P.efficiency, P.quant_bpw);
  return { system, est };
}

// ---------- rendering: configuration ----------

function renderConfigs() {
  const root = document.getElementById("configs");
  const focused = document.activeElement?.dataset?.focusId;
  root.replaceChildren(...state.configs.map((cfg, i) => renderConfig(cfg, i)));
  if (state.configs.length < MAX_CONFIGS) {
    root.append(h("div", { class: "panel add-config" },
      h("button", { type: "button", "data-act": "add-config" }, `+ Compare with configuration ${LETTERS[state.configs.length]}`)));
  }
  if (focused) root.querySelector(`[data-focus-id="${focused}"]`)?.focus();
  renderStreams();
}

function renderConfig(cfg, i) {
  const panel = h("div", { class: "panel config", "data-i": i });
  panel.append(h("div", { class: "config-head" },
    h("h2", {}, h("span", { class: "tag", "aria-hidden": "true" }, LETTERS[i]), `Configuration ${LETTERS[i]}`),
    h("div", { class: "segmented", role: "group", "aria-label": "Platform" },
      h("button", { type: "button", "data-act": "platform", "data-v": "pc", "aria-pressed": String(cfg.platform === "pc") }, "PC"),
      h("button", { type: "button", "data-act": "platform", "data-v": "apple", "aria-pressed": String(cfg.platform === "apple") }, "Mac")),
    state.configs.length > 1 ? h("button", { type: "button", class: "link", "data-act": "remove-config" }, "Remove") : null));

  panel.append(h("h3", {}, "Hardware"));
  if (cfg.platform === "apple") {
    const chip = P.apple.find((a) => a.name === cfg.chip);
    panel.append(h("div", { class: "fields" },
      field("Chip", select("chip", P.apple.map((a) => [a.name, chipLabel(a)]), cfg.chip, { "data-focus-id": `${i}-chip` })),
      field("Unified memory", select("memoryGb", chip.memory_options_gb.map((m) => [m, `${m} GB`]), cfg.memoryGb, { "data-focus-id": `${i}-mem` }))));
  } else {
    const gpuOptions = P.gpus.map((g) => [g.name, gpuLabel(g)]);
    cfg.gpus.forEach((g, j) => {
      panel.append(h("div", { class: "gpu-row" },
        field(j === 0 ? "GPU" : `GPU ${j + 1}`, select("gpu", gpuOptions, g, { "data-j": j, "data-focus-id": `${i}-gpu-${j}` })),
        field("VRAM", h("input", { value: `${P.gpus.find((x) => x.name === g).vram_gb} GB`, disabled: true })),
        h("button", { type: "button", "data-act": "remove-gpu", "data-j": j, "aria-label": `Remove GPU ${j + 1}` }, "✕")));
    });
    panel.append(h("button", { type: "button", class: "link", "data-act": "add-gpu" },
      cfg.gpus.length ? "+ Add another GPU" : "+ Add a GPU (currently CPU only)"));
    panel.append(h("div", { class: "fields" },
      field("CPU", select("cpu", P.cpus.map((c) => [c.name, `${c.name} (${c.cores} cores)`]), cfg.cpu, { "data-focus-id": `${i}-cpu` })),
      field("System RAM", select("ramGb", RAM_SIZES.map((s) => [s, `${s} GB`]), cfg.ramGb, { "data-focus-id": `${i}-ram` })),
      field("RAM type", select("ramKind", RAM_KINDS.map((k) => [k, k.toUpperCase()]), cfg.ramKind, { "data-focus-id": `${i}-kind` })),
      field("Channels", select("ramChannels", RAM_CHANNELS.map((c) => [c, `${c} (${ramBandwidthGbps(cfg.ramKind, c).toFixed(0)} GB/s)`]), cfg.ramChannels, { "data-focus-id": `${i}-ch` }))));
  }

  panel.append(h("h3", {}, "Model"));
  panel.append(h("div", { class: "fields" },
    field("Model", select("model", P.models.map((m) => [m.name, modelLabel(m)]), cfg.model, { "data-focus-id": `${i}-model` })),
    field("Quantization", select("quant", Object.entries(P.quant_bpw).map(([q, b]) => [q, `${q} (${b} bpw)`]), cfg.quant, { "data-focus-id": `${i}-quant` })),
    field("Context", select("context", CONTEXTS.map((c) => [c, `${c / 1024}K tokens`]), cfg.context, { "data-focus-id": `${i}-ctx` })),
    field("KV cache", select("kvBits", KV_BITS.map((b) => [b, b === 16 ? "F16" : `Q${b}`]), cfg.kvBits, { "data-focus-id": `${i}-kv` }))));

  panel.append(renderResult(cfg));
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
  let badge;
  if (!est.fits) badge = h("span", { class: "badge bad" }, "Does not fit in memory");
  else if (!system.gpus.length) badge = h("span", { class: "badge warn" }, "CPU only");
  else if (gpuLayers === n) badge = h("span", { class: "badge ok" }, system.unified ? "Fits in GPU memory" : "Fits in VRAM");
  else badge = h("span", { class: "badge warn" }, `${n - gpuLayers}/${n} layers on CPU`);

  const ttft = state.prompt / est.prefill_tps;
  const gen = genTime(est, state.prompt, state.maxTokens);
  const result = h("div", { class: "result" },
    badge,
    h("div", { class: "headline" },
      h("span", { class: "big" }, fmtTps(est.decode_tps_empty)),
      h("span", { class: "unit" }, "tok/s generation (empty context)")),
    h("div", { class: "metrics" },
      metric("At full context", `${fmtTps(est.decode_tps_full)} tok/s`),
      metric("Prompt processing", `${fmtTps(est.prefill_tps)} tok/s`),
      metric(`First token (${state.prompt.toLocaleString("en-US")}-token prompt)`, fmtTime(ttft)),
      metric(`${state.maxTokens}-token reply`, fmtTime(ttft + gen))),
    renderBars(system, est),
    est.notes.length ? h("ul", { class: "notes" }, est.notes.map((t) => h("li", {}, t))) : null,
    renderMeasured(cfg));
  return result;
}

function metric(k, v) {
  return h("div", { class: "metric" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v));
}

function genTime(est, prompt, maxTokens) {
  let t = 0;
  for (let i = 1; i < maxTokens; i++) t += 1 / est.decodeTpsAt(prompt + i);
  return t;
}

function bar(label, capacity, segments, over = false) {
  const total = Math.max(capacity, segments.reduce((a, s) => a + s.v, 0));
  const used = segments.reduce((a, s) => a + s.v, 0);
  return h("div", { class: "bar-row" },
    h("div", { class: "bar-label" }, h("span", {}, label), h("span", {}, `${gb(used)} / ${gb(capacity)}`)),
    h("div", { class: `bar${over ? " over" : ""}`, role: "img", "aria-label": `${label}: ${gb(used)} of ${gb(capacity)}` },
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
    rows.push(bar(`Unified memory (GPU limit ${gb(system.ram.size_gb * eff.unified_gpu_fraction)})`,
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
      rows.push(bar(system.gpus.length > 1 ? `GPU ${j + 1} VRAM` : "VRAM", gpu.vram_gb, segs));
    });
    rows.push(bar(`System RAM (${gb(eff.ram_reserve_gb)} kept for the OS)`, system.ram.size_gb,
                  cpu ? [{ cls: "cpu", v: cpu.weights_gb + cpu.kv_gb }] : [], !est.fits));
  }
  return h("div", {},
    h("div", { class: "bars" }, rows),
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: "background:var(--bar-gpu)" }), "Weights on GPU"),
      h("span", {}, h("i", { style: "background:var(--bar-kv)" }), "KV cache on GPU"),
      h("span", {}, h("i", { style: "background:var(--bar-cpu)" }), "Layers in system RAM")));
}

function renderMeasured(cfg) {
  const preset = cfg.platform === "apple" ? cfg.chip : (cfg.gpus.length === 1 ? cfg.gpus[0] : null);
  const rows = preset ? P.measurements.filter((m) => m.preset === preset) : [];
  if (!rows.length) return null;
  const body = rows.map((m) => {
    const { est } = compute(cfg, m.model, m.quant, m.n_prompt + m.n_gen);
    const tgEst = m.n_gen / genTime(est, 0, m.n_gen + 1);
    const fa = m.backend === "CUDA" ? (m.flash_attn ? " · FA" : "") : "";
    return h("tr", {},
      h("td", {}, `${m.quant}${fa}`),
      h("td", {}, fmtTps(m.tg_tps)), h("td", {}, fmtTps(tgEst)),
      h("td", {}, fmtTps(m.pp_tps)), h("td", {}, fmtTps(est.prefill_tps)));
  });
  return h("details", { class: "measured-wrap" },
    h("summary", {}, `Check: real benchmarks on this hardware (${rows.length})`),
    h("div", { class: "measured" },
      h("p", { style: "margin:0 0 6px" }, "Llama 2 7B with llama.cpp, measured vs. this simulator (tok/s). ",
        h("a", { href: rows[0].source }, "Source")),
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Quant"), h("th", {}, "Gen real"), h("th", {}, "Gen sim"),
          h("th", {}, "Prompt real"), h("th", {}, "Prompt sim"))),
        h("tbody", {}, body))));
}

// ---------- events ----------

function onConfigEvent(e) {
  const panel = e.target.closest(".config");
  const act = e.target.closest("[data-act]");
  if (e.type === "click" && act?.dataset.act === "add-config") {
    state.configs.push({ ...state.configs[state.configs.length - 1] });
    return changed();
  }
  if (!panel) return;
  const i = Number(panel.dataset.i);
  const cfg = state.configs[i];
  if (e.type === "click" && act) {
    const j = Number(act.dataset.j);
    switch (act.dataset.act) {
      case "platform": cfg.platform = act.dataset.v; break;
      case "add-gpu": cfg.gpus.push(cfg.gpus[cfg.gpus.length - 1] || "rtx3060-12gb"); break;
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
    if (k === "gpu") {
      cfg.gpus = cfg.gpus.map((g, j) => (j === Number(e.target.dataset.j) ? v : g));
    } else {
      cfg[k] = ["ramGb", "ramChannels", "memoryGb", "context", "kvBits"].includes(k) ? Number(v) : v;
    }
    state.configs[i] = sanitizeConfig(cfg);
    changed();
  }
}

function changed() {
  stopPlayback();
  state.configs = state.configs.map((c) => sanitizeConfig({ ...c, gpus: [...c.gpus] }));
  saveState();
  renderConfigs();
}

// ---------- mock streams ----------

function renderStreams() {
  const root = document.getElementById("streams");
  root.replaceChildren(...state.configs.map((cfg, i) => {
    let label = "";
    try {
      const m = P.models.find((x) => x.name === cfg.model);
      const hw = cfg.platform === "apple" ? chipName(P.apple.find((a) => a.name === cfg.chip))
        : (cfg.gpus.length ? cfg.gpus.map(gpuName).join(" + ") : "CPU only");
      label = `${hw} · ${m.name} ${cfg.quant}`;
    } catch { /* invalid configs show an error in their panel */ }
    return h("div", { class: "stream", "data-i": i },
      h("div", { class: "stream-head" },
        h("strong", {}, `${LETTERS[i]} · ${label}`),
        h("span", { class: "stats" }, "Press Play")),
      h("div", { class: "stream-out waiting", "aria-live": "off" }, ""));
  }));
}

function startPlayback() {
  const streams = [];
  state.configs.forEach((cfg, i) => {
    const el = document.querySelector(`.stream[data-i="${i}"]`);
    try {
      const { est } = compute(cfg);
      streams.push({
        el, est, shown: 0,
        mock: new MockStream({ promptTokens: state.prompt, maxTokens: state.maxTokens,
                               prefillTps: est.prefill_tps, decodeTps: est.decodeTpsAt, seed: 7 + i }),
      });
      el.querySelector(".stream-out").textContent = "";
    } catch (e) {
      el.querySelector(".stats").textContent = e.message;
    }
  });
  playback = { streams, start: performance.now(), raf: 0 };
  document.getElementById("play").textContent = "Stop";
  tick();
}

function stopPlayback() {
  if (!playback) return;
  cancelAnimationFrame(playback.raf);
  playback = null;
  document.getElementById("play").textContent = "Play";
}

function tick() {
  if (!playback) return;
  const elapsed = (performance.now() - playback.start) / 1000;
  let running = false;
  for (const s of playback.streams) {
    const out = s.el.querySelector(".stream-out");
    const stats = s.el.querySelector(".stats");
    const n = s.mock.tokensDueBy(elapsed);
    const total = s.mock.times.length;
    if (n > s.shown) {
      const words = s.mock.words.slice(s.shown, n).join(" ");
      out.textContent = s.shown ? `${out.textContent} ${words}` : words;
      out.classList.remove("waiting");
      out.scrollTop = out.scrollHeight;
      s.shown = n;
    }
    if (n === 0) {
      out.classList.add("waiting");
      out.textContent = `Processing ${state.prompt.toLocaleString("en-US")} prompt tokens… ` +
        `${fmtTime(Math.max(0, s.mock.ttft - elapsed))} left`;
      stats.textContent = `prefill ${fmtTps(s.est.prefill_tps)} tok/s`;
      running = true;
    } else {
      const genElapsed = Math.min(elapsed, s.mock.totalTime) - s.mock.ttft;
      const rate = n > 1 && genElapsed > 0 ? (n - 1) / genElapsed : 0;
      stats.textContent = `TTFT ${fmtTime(s.mock.ttft)} · ${n}/${total} tokens · ` +
        `${rate ? fmtTps(rate) : "–"} tok/s · ${fmtTime(Math.min(elapsed, s.mock.totalTime))}`;
      if (n < total) running = true;
    }
  }
  if (running) playback.raf = requestAnimationFrame(tick);
  else stopPlayback();
}

function readMockInputs() {
  const clamp = (el, lo, hi) => Math.min(hi, Math.max(lo, Math.round(Number(el.value) || lo)));
  state.prompt = clamp(document.getElementById("prompt-tokens"), 1, 200000);
  state.maxTokens = clamp(document.getElementById("max-tokens"), 1, 4000);
}

// ---------- boot ----------

async function main() {
  const resp = await fetch("data/presets.json");
  P = await resp.json();
  state = loadState();

  document.getElementById("prompt-tokens").value = state.prompt;
  document.getElementById("max-tokens").value = state.maxTokens;
  const cal = P.calibration;
  const pct = (x) => `${(x * 100).toFixed(1)} %`;
  document.getElementById("calibration").textContent =
    `Median error in generation speed: ${pct(cal["cuda/tg"].median)} on ${cal["cuda/tg"].n} NVIDIA results, ` +
    `${pct(cal["metal/tg"].median)} on ${cal["metal/tg"].n} Apple Silicon results.`;

  const configs = document.getElementById("configs");
  configs.addEventListener("click", onConfigEvent);
  configs.addEventListener("change", onConfigEvent);
  for (const id of ["prompt-tokens", "max-tokens"]) {
    document.getElementById(id).addEventListener("change", () => {
      readMockInputs();
      changed();
    });
  }
  document.getElementById("play").addEventListener("click", () => {
    if (playback) return stopPlayback();
    readMockInputs();
    saveState();
    startPlayback();
  });
  window.addEventListener("hashchange", () => {
    stopPlayback();
    state = loadState();
    renderConfigs();
  });

  renderConfigs();
}

main().catch((e) => {
  document.getElementById("configs").replaceChildren(
    h("p", { class: "error" }, `Could not start: ${e.message}. Serve it over HTTP with: llmsim web`));
});
