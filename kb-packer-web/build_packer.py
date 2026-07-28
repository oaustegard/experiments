#!/usr/bin/env python3
"""Generate the single-file kb-packer.html for austegard.com/ai-tools/.

Inlines the build core (lexkb-web.mjs, exports stripped) plus the canonical
shipped runtime (search.js / search.py / bundle_SKILL.md, vendored from
oaustegard/claude-skills creating-kb) into one self-contained page that matches
the ai-tools single-file convention and runs with no fetch (works on file://).

Re-run after the vendored files change (this is the sync step):
    python3 build_packer.py            # reads vendor/, writes kb-packer.html
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "vendor"
OUT = HERE / "kb-packer.html"

core = (V / "lexkb-web.mjs").read_text(encoding="utf-8")
core = re.sub(r"^export ", "", core, flags=re.M)  # inline, not a module export
search_js = (V / "search.js").read_text(encoding="utf-8")
search_py = (V / "search.py").read_text(encoding="utf-8")
bundle_skill = (V / "bundle_SKILL.md").read_text(encoding="utf-8")

# Runtime hash: SHA-256 over the exact inlined runtime bytes, in fixed order.
# Embedded in the HTML and pinned in vendor/RUNTIME_HASH so check_sync.py can
# prove (a) the HTML was regenerated from the current vendor, and (b) — combined
# with the upstream diff in check_sync — that vendor matches canonical creating-kb.
_runtime_concat = "\n--KBPACKER-RUNTIME--\n".join([core, search_js, search_py, bundle_skill])
RUNTIME_HASH = hashlib.sha256(_runtime_concat.encode("utf-8")).hexdigest()
(V.parent / "vendor" / "RUNTIME_HASH").write_text(RUNTIME_HASH + "\n", encoding="utf-8")

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="ai-disclosure" content="ai-assisted">
<meta name="kb-packer-runtime-hash" content="@@RUNTIME_HASH@@">
<title>KB Packer — build a portable knowledgebase skill</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui,-apple-system,sans-serif; line-height:1.6; margin:0; padding:24px;
         background:#fafafa; color:#333; }
  .container { max-width:800px; margin:0 auto; background:#fff; padding:30px; border-radius:8px;
               box-shadow:0 2px 10px rgba(0,0,0,.1); }
  a.back { color:#2563eb; text-decoration:none; font-size:14px; }
  h1 { margin:.2em 0 .1em; font-size:24px; }
  .sub { color:#666; margin:0 0 22px; }
  #drop { border:2px dashed #cbd5e1; border-radius:10px; padding:36px 20px; text-align:center; color:#64748b;
          transition:.15s; cursor:pointer; }
  #drop.hot { border-color:#2563eb; background:#eff6ff; color:#1e3a8a; }
  #drop strong { color:#111; }
  .row { display:flex; gap:14px; flex-wrap:wrap; margin-top:16px; }
  .field { flex:1 1 150px; }
  label { display:block; font-size:12px; color:#666; margin-bottom:4px; }
  input { width:100%; padding:8px 10px; border:1px solid #d1d5db; border-radius:7px; font:inherit; }
  button { background:#2563eb; color:#fff; border:0; border-radius:8px; padding:11px 20px;
           font:600 15px/1 inherit; cursor:pointer; margin-top:18px; }
  button:disabled { opacity:.5; cursor:default; }
  ul.files { max-height:140px; overflow:auto; margin:12px 0 0; padding:0; list-style:none;
             font:12px ui-monospace,monospace; color:#64748b; }
  #status { margin-top:14px; font-size:14px; white-space:pre-wrap; }
  .ok { color:#15803d; } .warn { color:#b45309; }
  a.dl { display:inline-block; margin-top:16px; background:#15803d; color:#fff; text-decoration:none;
         padding:11px 20px; border-radius:8px; font-weight:600; }
  code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:13px; }
  details { margin-top:22px; color:#555; font-size:14px; }
  summary { cursor:pointer; color:#2563eb; }
</style>
</head>
<body>
<div class="container">
  <a class="back" href="/ai-tools/">&larr; ai-tools</a>
  <h1>KB Packer</h1>
  <p class="sub">Turn files into a portable, embedding-free <code>.skill</code> knowledgebase. Everything
  runs in your browser — no upload, no model, no network. Drop files, download the skill, install it in any
  agent that supports skills.</p>

  <div id="drop">
    <strong>Drop files or a folder here</strong><br>
    or click to choose · text formats + PDF
    <input id="picker" type="file" multiple hidden>
  </div>
  <p class="note" style="color:#64748b;font-size:13px;margin:8px 2px 0">
    Text formats (<code>.txt .md .html .rst .csv</code> …) and <code>.pdf</code> (parsed locally;
    pdf.js loads from a CDN on first PDF). Other binaries (DOCX, RTF) — convert to text first.
    Add extensions below to ingest more.</p>
  <ul class="files" id="filelist"></ul>

  <div class="row">
    <div class="field"><label>KB name (becomes the skill name)</label><input id="name" type="text" value="my-kb"></div>
    <div class="field"><label>Extensions</label><input id="ext" type="text" value="txt,md,html,htm,pdf,rst,org,csv"></div>
    <div class="field"><label>Chunk chars (0 = whole doc)</label><input id="target" type="number" value="0" min="0"></div>
  </div>
  <div class="row"><div class="field" style="flex:1 1 100%"><label>Source description (optional)</label>
    <input id="source" type="text" placeholder="what this corpus is"></div></div>

  <button id="build" disabled>Build .skill</button>
  <div id="status"></div>
  <div id="dl"></div>

  <details>
    <summary>How it works</summary>
    <p>The packer chunks your files, builds a BM25 inverted index, and zips it with a query protocol and two
    searchers (<code>search.js</code> / <code>search.py</code>) into a <code>&lt;name&gt;.skill</code>. There is no
    embedding model: the agent you install the skill into supplies the semantic layer by expanding each query into
    search terms. Install the downloaded skill, then ask the agent questions about your corpus.</p>
  </details>
</div>

<script type="text/plain" id="rt-search-js">@@SEARCH_JS@@</script>
<script type="text/plain" id="rt-search-py">@@SEARCH_PY@@</script>
<script type="text/plain" id="rt-bundle-skill">@@BUNDLE_SKILL@@</script>

<script type="module">
// --- build core (vendored from oaustegard/claude-skills creating-kb, lexkb-web.mjs) ---
@@CORE@@

// --- UI ---
const $ = (id) => document.getElementById(id);
const drop = $("drop"), status = $("status"), dlBox = $("dl");
let files = [];
const runtime = {
  searchJs: $("rt-search-js").textContent,
  searchPy: $("rt-search-py").textContent,
  bundleSkillMd: $("rt-bundle-skill").textContent,
};

function setStatus(m, c) { status.className = c || ""; status.textContent = m; }
function renderFiles() {
  $("filelist").innerHTML = files.map((f) => `<li>${f.name} · ${f.text.length}c</li>`).join("");
  $("build").disabled = files.length === 0;
}
function mergeFiles(next) {
  const byName = new Map(files.map((f) => [f.name, f]));
  for (const f of next) byName.set(f.name, f);
  files = [...byName.values()];
  renderFiles();
  setStatus(`${files.length} file(s) staged.` + (failed ? ` (${failed} skipped — unreadable / PDF parse failed)` : ""),
            failed ? "warn" : "");
}
// pdf.js loaded lazily (only when a PDF is dropped) so text-only use stays offline.
let pdfReady = null;
function loadPdfJs() {
  if (pdfReady) return pdfReady;
  const base = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174";
  pdfReady = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = base + "/pdf.min.js";
    s.onload = () => { window.pdfjsLib.GlobalWorkerOptions.workerSrc = base + "/pdf.worker.min.js"; res(window.pdfjsLib); };
    s.onerror = () => rej(new Error("could not load pdf.js (network needed for PDF parsing)"));
    document.head.appendChild(s);
  });
  return pdfReady;
}
async function pdfToText(file) {
  const lib = await loadPdfJs();
  const pdf = await lib.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const tc = await (await pdf.getPage(i)).getTextContent();
    let pageText = "", lastY = null;
    tc.items.forEach((item, idx) => {
      const t = item.str;
      if (!t.trim()) return;
      if (lastY !== null && Math.abs(item.transform[5] - lastY) > 5) pageText += "\\n";
      pageText += t;
      const nx = tc.items[idx + 1];
      if (nx && Math.abs(nx.transform[5] - item.transform[5]) < 5 &&
          !t.endsWith(" ") && !t.endsWith("-") && nx.str && !nx.str.startsWith(" ")) pageText += " ";
      lastY = item.transform[5];
    });
    pages.push(pageText.trim());
  }
  return pages.join("\\n\\n");
}

let failed = 0;
async function readFileObjs(objs) {
  const out = [];
  failed = 0;
  for (const f of objs) {
    const name = f.webkitRelativePath || f._rel || f.name;
    try {
      const text = /\\.pdf$/i.test(name) ? await pdfToText(f) : await f.text();
      out.push({ name, text });
    } catch (e) { failed++; console.warn("skip", name, e.message); }
  }
  return out;
}

drop.addEventListener("click", () => $("picker").click());
$("picker").addEventListener("change", async (e) => mergeFiles(await readFileObjs(e.target.files)));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hot"); }));

async function walkEntry(entry, prefix, out) {
  if (entry.isFile) await new Promise((res) => entry.file((f) => { f._rel = prefix + f.name; out.push(f); res(); }));
  else if (entry.isDirectory) {
    const r = entry.createReader();
    const ents = await new Promise((res) => r.readEntries(res));
    for (const e of ents) await walkEntry(e, prefix + entry.name + "/", out);
  }
}
drop.addEventListener("drop", async (e) => {
  const items = [...(e.dataTransfer.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (items.length) {
    const collected = [];
    for (const it of items) await walkEntry(it, "", collected);
    mergeFiles(await readFileObjs(collected));
  } else mergeFiles(await readFileObjs(e.dataTransfer.files));
});

$("build").addEventListener("click", () => {
  try {
    const exts = new Set($("ext").value.split(",").map((s) => s.trim().replace(/^\\./, "").toLowerCase()).filter(Boolean));
    const target = Number($("target").value) || 0;
    const name = ($("name").value.trim() || "my-kb").replace(/[^a-zA-Z0-9._-]/g, "-");
    const chunks = collectChunks(files, exts, target, 40);
    if (!chunks.length) { setStatus("No chunks produced — check your extensions filter.", "warn"); return; }
    const index = buildIndex(chunks, 1.5, 0.75);
    const bundle = bundleFiles(chunks, index, $("source").value.trim() || `${name} corpus`, runtime);
    const bytes = zipSkill(bundle, name);
    const nFiles = new Set(chunks.map((c) => c.meta.source_path)).size;
    setStatus(`Built ${chunks.length} chunks from ${nFiles} file(s) · vocab ${Object.keys(index.df).length} · ${(bytes.length / 1024).toFixed(1)} KB`, "ok");
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/zip" }));
    dlBox.innerHTML = `<a class="dl" download="${name}.skill" href="${url}">Download ${name}.skill</a>`;
  } catch (err) { setStatus("Error: " + err.message, "warn"); }
});
</script>
</body>
</html>
"""

html = (TEMPLATE
        .replace("@@CORE@@", core)
        .replace("@@SEARCH_JS@@", search_js)
        .replace("@@SEARCH_PY@@", search_py)
        .replace("@@BUNDLE_SKILL@@", bundle_skill)
        .replace("@@RUNTIME_HASH@@", RUNTIME_HASH))
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)/1024:.1f} KB)")
