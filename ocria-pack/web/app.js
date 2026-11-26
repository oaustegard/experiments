// Demo page: OCRia-nano-1 from packed weights on LiteParse's wasm build. Logs go through relay.js (?log=1 on Oskar's devices).
import { loadOcria, LITEPARSE_VERSION } from "./ocria.js";

const BUILD = "2026-11-26-a"; // keep equal to build.json
const $ = (id) => document.getElementById(id);
const relay = (window.Relay || { start() { return this; }, log() {}, warn() {}, error() {} }).start({ page: "ocria-pack", build: BUILD });
$("build").textContent = BUILD;
fetch("build.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((b) => {
  if (!b || !b.build || b.build === BUILD) return;
  relay.warn("stale page", { running: BUILD, current: b.build });
  let tried = null;
  try { tried = sessionStorage.getItem("ocria:reloaded-for"); sessionStorage.setItem("ocria:reloaded-for", b.build); } catch (e) {}
  if (tried === b.build) return;
  const url = new URL(location.href);
  url.searchParams.set("b", b.build);
  location.replace(url.toString());
}).catch(() => {});

const status = (msg, err) => { const s = $("status"); s.textContent = msg; s.className = err ? "err" : "muted"; };
const fmtB = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + " MB" : (n / 1e3).toFixed(0) + " kB");
const ms = (v) => (v == null ? "" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");

const engines = {};   // "layout|scripts" -> loaded reader
let image = null;     // { canvas, name }
let busy = false;

function wire(re) {
  let n = 0;
  for (const e of performance.getEntriesByType("resource")) if (re.test(e.name)) n += e.transferSize || e.encodedBodySize || 0;
  return n;
}

async function engine() {
  const layout = $("layout").value === "on";
  const scripts = $("scripts").value ? $("scripts").value.split(",") : [];
  const key = `${layout}|${scripts.join(",")}`;
  if (!engines[key]) {
    status("Loading OCRia-nano (runtime from jsdelivr, packed weights from this site)...");
    engines[key] = await loadOcria({ layout, scripts, log: (m) => { relay.log("load", { m }); status(m); } });
    relay.log("ready", { key, load_ms: Math.round(engines[key].loadMs), pack_bytes: engines[key].packBytes,
                         runtime_wire: wire(/liteparse-wasm/) });
  }
  return engines[key];
}

async function loadImage(src, name) {
  const img = new Image();
  img.src = src;
  await img.decode();
  const cv = document.createElement("canvas");
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  cv.getContext("2d").drawImage(img, 0, 0);
  image = { canvas: cv, name };
  draw([]);
  relay.log("image", { name, w: cv.width, h: cv.height });
}

function draw(items) {
  const cv = $("cv"), src = image.canvas;
  cv.width = src.width; cv.height = src.height;
  const g = cv.getContext("2d");
  g.drawImage(src, 0, 0);
  g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--box") || "#e0301e";
  g.lineWidth = Math.max(1, src.width / 600);
  for (const it of items) g.strokeRect(it.box.x, it.box.y, it.box.width, it.box.height);
}

async function run() {
  if (busy || !image) return;
  busy = true; $("run").disabled = true;
  try {
    const t0 = performance.now();
    const ocr = await engine();
    const tLoad = performance.now() - t0;
    status(`Reading ${image.name}...`);
    const page = await ocr.read(image.canvas);
    draw(page.items);
    $("md").textContent = page.markdown || "(no text found)";
    $("md").className = "";
    const rows = [
      ["page", `${page.width} x ${page.height} px, ${page.items.length} text items, ${page.blocks.length} blocks`],
      ["read", ms(page.ms)],
      ["load (this run)", ms(tLoad)],
      ["packs on the wire", fmtB(ocr.packBytes)],
      [`LiteParse ${LITEPARSE_VERSION} runtime on the wire`, fmtB(wire(/liteparse-wasm/)) + " (0 when cached)"],
    ];
    $("stats").tBodies[0].innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
    status(`Done: ${image.name} in ${ms(page.ms)}.`);
    relay.log("page", { name: image.name, w: page.width, h: page.height, items: page.items.length, blocks: page.blocks.length,
                        ms: Math.round(page.ms), chars: page.markdown.length });
  } catch (e) {
    relay.error(e);
    status(`Failed: ${(e && e.message) || e}`, true);
  } finally {
    busy = false; $("run").disabled = false;
  }
}

$("run").onclick = run;
$("sample").onchange = () => loadImage($("sample").value, $("sample").selectedOptions[0].text).catch((e) => status(String(e), true));
$("file").onchange = () => {
  const f = $("file").files[0];
  if (f) loadImage(URL.createObjectURL(f), f.name).then(run).catch((e) => status(String(e), true));
};
loadImage($("sample").value, $("sample").selectedOptions[0].text)
  .then(() => status("Ready. Press Run (the first run downloads the runtime and weights)."))
  .catch((e) => status(String(e), true));
