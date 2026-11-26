/* Test page for the Nemotron Parse WebGPU engine (3.5 by default, 2.0 with ?pack=v1) (engine/nemotron.js). Images or PDFs in; PDF pages are
   flattened by pdf-flatten.js the way anything-to-text renders them. Every step is logged through relay.js. */
import { NemotronParse } from "./engine/nemotron.js";
import { openPdf } from "./pdf-flatten.js";
import { marked } from "https://esm.sh/marked@12.0.2";
import DOMPurify from "https://esm.sh/dompurify@3.1.6";

const BUILD = "2026-11-26-e";   /* keep equal to build.json */
const relay = (window.Relay || { start() { return this; }, log() {}, warn() {}, error() {} }).start({ page: "nemotron-parse-pack", build: BUILD });
fetch("build.json", { cache: "no-store" }).then(r => (r.ok ? r.json() : null)).then(b => {
  if (b && b.build && b.build !== BUILD && !sessionStorage.getItem("np-reloaded")) {
    sessionStorage.setItem("np-reloaded", "1"); location.reload();
  }
}).catch(() => {});

const $ = id => document.getElementById(id);
/* A phone has far less memory for one tab than a laptop, and iOS reloads a tab that takes too much. */
const PHONE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));
const q = new URLSearchParams(location.search);
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/nemotron-parse-pack/";
const PACK = q.get("pack") || "v35i8";
const BASE = q.get("base") || R2 + PACK + "/";
$("build").textContent = BUILD + " · pack " + PACK;

const COLORS = {
  Title: "#d1495b", "Section-header": "#e07a1f", Text: "#2f5fd0", "List-item": "#2a9d8f", Table: "#8e44ad",
  Picture: "#6a994e", Chart: "#6a994e", Caption: "#b5838d", Footnote: "#7f8c8d", "Page-header": "#95a5a6",
  "Page-footer": "#95a5a6", Formula: "#c0392b", Code: "#34495e", Bibliography: "#16a085", TOC: "#16a085",
};
$("legend").innerHTML = Object.entries(COLORS).filter(([k], i, a) => a.findIndex(([, c]) => c === COLORS[k]) === i)
  .map(([k, c]) => `<span style="--c:${c}">${k}</span>`).join("");

let model = null, loading = null, image = null, pdf = null, abort = null;
const stats = {};
function setStat(k, v) { stats[k] = v; $("stats").innerHTML = Object.entries(stats).map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join(""); }
function status(msg, err) { $("status").textContent = msg; $("status").className = err ? "err" : "muted"; }
function progress(f) { const p = $("prog"); if (f == null) { p.hidden = true; return; } p.hidden = false; p.value = f; }

for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => {
  for (const o of document.querySelectorAll(".tabs button")) o.setAttribute("aria-selected", String(o === b));
  for (const t of document.querySelectorAll(".tab")) t.hidden = t.id !== b.dataset.tab;
};

function draw(blocks) {
  const cv = $("cv"), ctx = cv.getContext("2d");
  if (!image) return;
  const s = Math.min(1, 1400 / Math.max(image.width, image.height));
  cv.width = Math.round(image.width * s); cv.height = Math.round(image.height * s);
  ctx.drawImage(image, 0, 0, cv.width, cv.height);
  ctx.lineWidth = 2; ctx.font = "12px system-ui";
  (blocks || []).forEach((b, i) => {
    const c = COLORS[b.cls] || "#e0301e";
    const [x0, y0, x1, y1] = b.bbox.map(v => v * s);
    ctx.strokeStyle = c; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = (i + 1) + " " + b.cls;
    ctx.fillStyle = c; ctx.fillRect(x0, Math.max(0, y0 - 15), ctx.measureText(label).width + 6, 15);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x0 + 3, Math.max(11, y0 - 3));
  });
}

async function setImage(src, name) {
  image = src; draw([]);
  $("run").disabled = false;
  status(`${name}: ${src.width} × ${src.height} px. Press Parse.`);
  relay.log("image", { name, w: src.width, h: src.height });
}

async function loadFile(file) {
  try {
    pdf && pdf.destroy(); pdf = null; $("pagepick").hidden = true;
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      status("Opening the PDF…");
      pdf = await openPdf(file);
      $("pdfpage").innerHTML = Array.from({ length: pdf.numPages }, (_, i) => `<option value="${i + 1}">${i + 1} of ${pdf.numPages}</option>`).join("");
      $("pagepick").hidden = pdf.numPages < 2;
      relay.log("pdf", { name: file.name, pages: pdf.numPages });
      await showPdfPage(1, file.name);
    } else {
      await setImage(await createImageBitmap(file), file.name);
    }
  } catch (e) { status("Could not read the file: " + (e.message || e), true); relay.error("file", { error: String(e && e.stack || e) }); }
}
async function showPdfPage(n, name) {
  status(`Rendering page ${n}…`);
  const t0 = performance.now();
  const cv = await pdf.page(n, { longEdge: 2048 });
  setStat("PDF render", `${((performance.now() - t0) / 1000).toFixed(2)} s (pdf.js, flattened)`);
  await setImage(cv, `${name || "PDF"} page ${n}`);
}
$("pdfpage").onchange = () => showPdfPage(+$("pdfpage").value);
$("file").onchange = e => e.target.files[0] && loadFile(e.target.files[0]);
const drop = $("drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]); };
$("sample").onchange = async () => {
  const u = $("sample").value; if (!u) return;
  const r = await fetch(u); await setImage(await createImageBitmap(await r.blob()), u.split("/").pop());
};

async function ensureModel() {
  if (model) return model;
  if (loading) return loading;
  loading = (async () => {
    const t0 = performance.now();
    status("Downloading the model (kept by the browser after the first time)…"); progress(0);
    let downloaded = 0;
    const m = await NemotronParse.load({
      base: BASE,
      /* phones: smaller GPU submissions, so one dispatch never runs long enough for the OS to kill it */
      ...(PHONE ? { budget: 5e8 } : {}),
      onProgress: ({ phase, loaded, total }) => {
        if (phase === "download" && total) { downloaded = total; progress(loaded / total); $("size").textContent = (total / 1e6).toFixed(0) + " MB"; }
        status(phase === "download" ? `Downloading the model: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB (kept by the browser after the first time)…`
          : phase === "upload" ? "Copying the weights to the GPU…" : "Model ready.");
        if (phase === "ready") setStat("Download", downloaded ? (downloaded / 1e6).toFixed(1) + " MB (compressed)" : "none: weights from the browser cache");
      },
    });
    progress(null);
    const s = (performance.now() - t0) / 1000;
    setStat("Model load", `${s.toFixed(1)} s`);
    if (m.info) {
      if (m.info.downloadBytes != null) setStat("Download", (m.info.downloadBytes / 1e6).toFixed(1) + " MB");
      if (m.info.adapter) setStat("GPU", m.info.adapter);
    }
    relay.log("model loaded", { s, info: m.info || null });
    model = m; return m;
  })();
  try { return await loading; } catch (e) { loading = null; throw e; }
}

$("run").onclick = async () => {
  if (!image) return;
  $("run").disabled = true; $("stop").disabled = false;
  abort = new AbortController();
  try {
    if (!navigator.gpu) throw new Error("This browser has no WebGPU. Try a current Chrome, Edge or Safari.");
    const m = await ensureModel();
    const prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_" + ($("pictext").checked ? "text_in_pic>" : "no_text_in_pic>");
    const size = $("size-sel").value === "half" ? { height: 1024, width: 832 } : { height: 2048, width: 1664 };
    $("raw").textContent = ""; let n = 0; const t0 = performance.now(); let tFirst = 0;
    status("Encoding the page…");
    const r = await m.parse(image, {
      prompt, size, maxTokens: +$("maxtok").value || 4096, signal: abort.signal,
      onStage: (done, total) => { progress(done / total); status(`Encoding the page: step ${done} of ${total}…`); },
      onToken: (id, piece) => {
        if (!n) progress(null);
        if (!n) tFirst = performance.now();
        n++; $("raw").textContent += piece;
        if (n % 16 === 0) status(`Writing: ${n} tokens, ${(n / ((performance.now() - tFirst) / 1000 || 1)).toFixed(1)} tokens/s…`);
      },
    });
    const ms = r.ms || {};
    setStat("Encoder", ms.encode != null ? (ms.encode / 1000).toFixed(2) + " s" : "–");
    setStat("Decoder", ms.decode != null ? `${(ms.decode / 1000).toFixed(1)} s for ${r.tokens.length} tokens (${(r.tokens.length / (ms.decode / 1000)).toFixed(1)} tokens/s)` : "–");
    setStat("Total", ((performance.now() - t0) / 1000).toFixed(1) + " s");
    $("raw").textContent = r.raw;
    const md = toMarkdown(r.blocks);
    $("md").textContent = md;
    $("rendered").classList.remove("muted");
    $("rendered").innerHTML = DOMPurify.sanitize(marked.parse(md));
    draw(r.blocks);
    status(`Done: ${r.blocks.length} blocks, ${r.tokens.length} tokens.`);
    relay.log("parsed", { blocks: r.blocks.length, tokens: r.tokens.length, ms, size, hitLimit: !!r.hitLimit });
  } catch (e) {
    if (e && e.name === "AbortError") status("Stopped.");
    else { status("Failed: " + (e && e.message || e), true); relay.error("parse", { error: String(e && e.stack || e) }); }
  } finally {
    progress(null); $("run").disabled = false; $("stop").disabled = true;
  }
};
/* Page Markdown from the blocks in the decoder's order, as the benchmark adapter joins them: titles and section
   headers as headings, picture and chart text left out unless "text in pictures" was asked for. */
function toMarkdown(blocks) {
  return blocks.filter(b => b.text && b.text.trim() && ((b.cls !== "Picture" && b.cls !== "Chart") || $("pictext").checked))
    .map(b => b.cls === "Title" && !b.text.startsWith("#") ? "# " + b.text
      : b.cls === "Section-header" && !b.text.startsWith("#") ? "## " + b.text : b.text).join("\n\n");
}
$("stop").onclick = () => abort && abort.abort();

if (PHONE) {
  $("size-sel").value = "half";
  $("phone-note").hidden = false;
}
status(navigator.gpu ? "Pick an image or a PDF." : "This browser has no WebGPU, which the model needs.", !navigator.gpu);
relay.log("start", { webgpu: !!navigator.gpu, pack: PACK });
