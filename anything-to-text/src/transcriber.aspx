<%@ Page Language="C#" %>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MSD Local Transcriber</title>
<link rel="icon" type="image/png" href="data:image/png;base64,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" />
<!-- MSD brand fonts (Arial Narrow / Tahoma) are standard on Windows; no web fonts needed -->
<style>
  /* ---- theme tokens ---- */
  :root {
    --bg:        #ffffff;
    --bg-2:      #ffffff;
    --panel:     #f2f2f2;          /* MSD light gray */
    --line:      rgba(0,0,0,0.12);
    --ink:       #000000;          /* MSD black body text */
    --ink-dim:   rgba(0,0,0,0.62);
    --ink-faint: rgba(0,0,0,0.45);
    --accent:    #990000;          /* MSD Red — the only red in the palette */
    --accent-2:  #990000;          /* palette has no second accent; reuse MSD Red */
    --danger:    #990000;
    --radius:    6px;
    --mono:    "Arial Narrow", Arial, "Helvetica Neue", sans-serif;  /* MSD body / UI font */
    --display: "Arial Narrow", Arial, "Helvetica Neue", sans-serif;  /* MSD headings (bold) */
    --chrome:  "Tahoma", "Arial Narrow", Arial, sans-serif;          /* footer chrome */
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--mono);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  /* MSD brand: thin dark-red rule across the very top, echoing the deck footer bar */
  body::before {
    content: "";
    position: fixed; top: 0; left: 0; right: 0; height: 4px;
    background: var(--accent);
    z-index: 5;
  }

  .wrap {
    position: relative;
    z-index: 1;
    max-width: 920px;
    margin: 0 auto;
    padding: 40px 24px 80px;
  }

  header { margin-bottom: 28px; }
  .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
  .brand-mark { height: 30px; width: auto; display: block; }
  .brand-name { font-family: var(--display); font-weight: 700; font-size: 15px; letter-spacing: 0.02em; text-transform: uppercase; color: var(--ink); }
  .eyebrow {
    font-size: 11px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 10px;
  }
  h1 {
    font-family: var(--display);
    font-weight: 700;
    color: var(--accent);
    font-size: clamp(30px, 6vw, 50px);
    line-height: 1.0;
    letter-spacing: 0;
    text-transform: uppercase;
    margin: 0 0 12px;
  }
  .sub { color: var(--ink-dim); margin: 0; }
  .sub b { color: var(--ink); font-weight: 600; }

  /* ---- pipeline stage strip ---- */
  /* hidden attribute must win over explicit display rules (e.g. .pipeline's grid),
     otherwise toggling element.hidden silently fails on laid-out elements */
  [hidden] { display: none !important; }
  .pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 26px 0;
  }
  .stage {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 12px 14px;
    background: linear-gradient(180deg, var(--panel), transparent);
    transition: border-color .25s, color .25s, box-shadow .25s;
  }
  .stage .n { color: var(--ink-faint); font-size: 11px; }
  .stage .t { color: var(--ink-dim); font-size: 12px; margin-top: 4px; }
  .stage.active {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px rgba(153,0,0,0.45);
    background: var(--bg);
  }
  .stage.active .t, .stage.active .n { color: var(--accent); }
  .stage.done { border-color: var(--accent); background: var(--accent); }
  .stage.done .t, .stage.done .n { color: #ffffff; }

  /* ---- panels ---- */
  .panel {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: var(--panel);
    padding: 20px;
    margin-bottom: 18px;
  }

  /* drop zone (shared by transcribe + edit pickers) */
  .dropzone {
    border: 1.5px dashed var(--line);
    border-radius: var(--radius);
    padding: 34px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    background: rgba(255,255,255,0.012);
  }
  .dropzone:hover, .dropzone.hover { border-color: var(--accent); background: rgba(153,0,0,0.04); }
  .dropzone .big { font-family: var(--display); font-size: 18px; font-weight: 600; color: var(--ink); }
  .dropzone .hint { color: var(--ink-faint); font-size: 12px; margin-top: 6px; }
  .dropzone.has-file { border-style: solid; border-color: var(--accent); }
  .filemeta { color: var(--ink-dim); font-size: 12px; margin-top: 10px; word-break: break-all; }
  .filemeta b { color: var(--ink); }

  /* controls */
  .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
  label.field { display: block; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px; }
  select, .seg {
    width: 100%;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--ink);
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 9px 10px;
  }
  select:focus { outline: none; border-color: var(--accent); }
  .seg { display: flex; padding: 0; overflow: hidden; }
  .seg button {
    flex: 1; border: 0; background: transparent; color: var(--ink-dim);
    font-family: var(--mono); font-size: 13px; padding: 9px 0; cursor: pointer;
  }
  .seg button.on { background: var(--accent); color: #ffffff; font-weight: 700; }
  .radios { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; min-height: 38px; }
  .radio { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--ink); cursor: pointer; }
  .radio input { accent-color: var(--accent); cursor: pointer; margin: 0; }

  .runbar { display: flex; align-items: center; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
  button.primary {
    font-family: var(--display); font-weight: 700; font-size: 15px;
    letter-spacing: 0.04em; text-transform: uppercase;
    color: #ffffff; background: var(--accent);
    border: 0; border-radius: var(--radius); padding: 12px 28px; cursor: pointer;
    transition: transform .08s, filter .2s;
  }
  button.primary:hover:not(:disabled) { filter: brightness(1.18); }
  button.primary:active:not(:disabled) { transform: translateY(1px); }
  button.primary:disabled { opacity: 0.45; cursor: not-allowed; }
  .badge {
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--ink-dim);
  }
  .badge.gpu { color: #ffffff; background: var(--accent); border-color: var(--accent); }
  .badge.cpu { color: var(--accent); border-color: var(--accent); }

  /* progress + status */
  #status { color: var(--ink-dim); font-size: 12px; min-height: 18px; }
  #status .err { color: var(--danger); }
  .bar { height: 6px; background: rgba(0,0,0,0.08); border-radius: 999px; overflow: hidden; margin-top: 10px; display: none; }
  .bar.show { display: block; }
  .bar > i { display: block; height: 100%; width: 0%; background: var(--accent); transition: width .2s; }
  /* indeterminate: decode/resample report no percentage, so slide instead of fake a fill */
  .bar.indeterminate > i { width: 40%; transition: none; animation: bar-slide 1.1s ease-in-out infinite; }
  @keyframes bar-slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(280%); } }

  /* results */
  #results { display: none; }
  #results.show { display: block; }
  .result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
  .result-head h2 { font-family: var(--display); font-size: 20px; margin: 0; }
  .result-actions { display: flex; gap: 8px; }
  button.ghost {
    font-family: var(--mono); font-size: 12px; color: var(--ink-dim);
    background: transparent; border: 1px solid var(--line); border-radius: 8px;
    padding: 7px 12px; cursor: pointer;
  }
  button.ghost:hover { color: var(--ink); border-color: var(--accent); }

  #partial { color: var(--ink-faint); font-style: italic; white-space: pre-wrap; min-height: 0; }
  .plaintext { white-space: pre-wrap; line-height: 1.7; color: var(--ink); }

  .segments { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }
  .segments summary { cursor: pointer; color: var(--ink-dim); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; }
  .seg-row { display: grid; grid-template-columns: auto 1fr; gap: 12px; padding: 4px 0; border-bottom: 1px solid var(--line); align-items: start; }
  .seg-row .ts {
    display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
    font-family: var(--mono); font-size: 12px; color: var(--accent);
    background: transparent; border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 9px; cursor: pointer; text-align: left;
    transition: background .15s, border-color .15s, color .15s;
  }
  .seg-row .ts:hover:not(:disabled) { border-color: var(--accent); background: rgba(153,0,0,0.06); }
  .seg-row .ts:disabled { color: var(--ink-faint); border-color: transparent; cursor: default; }
  .seg-row .ts.playing { background: var(--accent); color: #ffffff; border-color: var(--accent); }
  .seg-row .ts .glyph { font-size: 9px; line-height: 1; }
  .seg-row .tx { color: var(--ink); padding-top: 4px; }
  .seg-row .tx.editing { background: #ffffff; border: 1px solid var(--line); border-radius: 4px; padding: 4px 8px; }
  .seg-row .tx.editing:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .tabs { display: flex; gap: 4px; margin-bottom: 18px; border-bottom: 2px solid var(--line); }
  .tab {
    appearance: none; background: none; border: none; cursor: pointer;
    font-family: var(--display); font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; font-size: 13px; color: var(--ink-dim);
    padding: 10px 16px; margin-bottom: -2px; border-bottom: 2px solid transparent;
  }
  .tab:hover { color: var(--ink); }
  .tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
  .edit-help { color: var(--ink-dim); font-size: 12px; margin: 0 0 14px; }
  .edit-inputs { display: flex; flex-direction: column; gap: 12px; }
  .edit-inputs .dropzone { padding: 24px 20px; }
  .edit-inputs .dropzone .big { font-size: 15px; }

  .meta-line { color: var(--ink-faint); font-size: 11px; margin-top: 14px; }
  footer { margin-top: 36px; }
  footer .tech-note { color: var(--ink-faint); font-size: 11px; line-height: 1.7; }
  footer code { color: var(--ink-dim); }
  a { color: var(--accent); }

  @media (max-width: 680px) {
    .pipeline { grid-template-columns: repeat(2, 1fr); }
    .controls { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <img class="brand-mark" alt="MSD" src="data:image/png;base64,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" />
      <span class="brand-name">Meso Scale Diagnostics</span>
    </div>
    <p class="eyebrow">On-device speech recognition</p>
    <h1>Local Transcriber</h1>
    <p class="sub">Drop in a recording &mdash; an <b>audio or video</b> file &mdash; and get a written transcript back. Everything happens right here on your own computer: <b>your file is never uploaded</b> and never leaves your device.</p>
  </header>

  <!-- mode tabs: transcribe a new file OR edit an existing transcript (mutually exclusive) -->
  <div class="tabs" role="tablist" aria-label="Mode">
    <button class="tab" id="tab-transcribe" type="button" role="tab" aria-selected="true" aria-controls="panel-transcribe">Transcribe</button>
    <button class="tab" id="tab-edit" type="button" role="tab" aria-selected="false" aria-controls="panel-edit">Edit existing</button>
  </div>

  <!-- transcribe panel -->
  <div class="panel" id="panel-transcribe" role="tabpanel" aria-labelledby="tab-transcribe">
    <div id="drop" class="dropzone">
      <div class="big">Drop a file or click to choose</div>
      <div class="hint">mp4 / mov / mkv / webm &middot; mp3 / wav / m4a / flac / ogg</div>
      <div class="filemeta" id="filemeta" hidden></div>
    </div>
    <input type="file" id="file" accept="audio/*,video/*" hidden />

    <div class="controls">
      <div>
        <label class="field" for="model">Model</label>
        <select id="model"></select>
      </div>
      <div>
        <label class="field" for="lang">Language</label>
        <select id="lang">
          <option value="auto">Auto-detect</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
          <option value="nl">Dutch</option>
          <option value="ja">Japanese</option>
          <option value="zh">Chinese</option>
          <option value="ko">Korean</option>
          <option value="ru">Russian</option>
          <option value="ar">Arabic</option>
          <option value="hi">Hindi</option>
        </select>
      </div>
      <div>
        <label class="field">Task</label>
        <div class="radios" id="task">
          <label class="radio"><input type="radio" name="task" value="transcribe" checked /> Transcribe</label>
          <label class="radio"><input type="radio" name="task" value="translate" /> Translate &rarr; EN</label>
        </div>
      </div>
    </div>

    <div class="runbar">
      <button class="primary" id="run" disabled>Transcribe</button>
      <span class="badge" id="device">detecting&hellip;</span>
      <span id="status"></span>
    </div>
    <div class="bar" id="bar"><i></i></div>
  </div>

  <!-- edit panel -->
  <div class="panel" id="panel-edit" role="tabpanel" aria-labelledby="tab-edit" hidden>
    <p class="edit-help">Load the original audio/video and its transcript. An .srt keeps timestamps and per-segment playback; a .txt loads as plain lines. Edit the text, play any segment to check it against the audio, then download the corrected file.</p>
    <div class="edit-inputs">
      <div class="dropzone" id="edit-audio-zone">
        <div class="big">Original audio / video</div>
        <div class="hint">Drop or click to choose &middot; the recording you transcribed</div>
        <div class="filemeta" id="edit-audio-meta" hidden></div>
      </div>
      <input type="file" id="edit-audio" accept="audio/*,video/*" hidden />

      <div class="dropzone" id="edit-transcript-zone">
        <div class="big">Transcript</div>
        <div class="hint">Drop or click to choose &middot; .srt or .txt</div>
        <div class="filemeta" id="edit-transcript-meta" hidden></div>
      </div>
      <input type="file" id="edit-transcript" accept=".srt,.txt,text/plain" hidden />
    </div>

    <div class="runbar">
      <button class="primary" id="edit-load" disabled>Load for editing</button>
      <span id="edit-status"></span>
    </div>
    <div class="bar" id="edit-bar"><i></i></div>
  </div>

  <!-- pipeline strip: progress while a job runs, sits above the result -->
  <div class="pipeline" id="pipeline">
    <div class="stage" data-stage="decode"><div class="n">01</div><div class="t">Extract audio</div></div>
    <div class="stage" data-stage="model"><div class="n">02</div><div class="t">Load model</div></div>
    <div class="stage" data-stage="transcribe"><div class="n">03</div><div class="t">Transcribe</div></div>
    <div class="stage" data-stage="present"><div class="n">04</div><div class="t">Read result</div></div>
  </div>

  <!-- results panel -->
  <div class="panel" id="results">
    <div class="result-head">
      <h2>Transcript</h2>
      <div class="result-actions">
        <button class="ghost" id="copy">Copy</button>
        <button class="ghost" id="dl-txt">Download .txt</button>
        <button class="ghost" id="dl-srt">Download .srt</button>
      </div>
    </div>
    <div id="partial"></div>
    <div class="plaintext" id="plaintext"></div>
    <details class="segments" id="segwrap">
      <summary>Timestamped segments</summary>
      <div id="segments"></div>
    </details>
    <div class="meta-line" id="metaline"></div>
  </div>

  <footer>
    <div class="tech-note" id="tech-note"></div>
  </footer>
</div>

<script type="module">
/* =========================================================================
   Local Transcriber
   - Decode/extract audio to 16 kHz mono Float32 on the main thread
       * native Web Audio for audio files (fast, free)
       * lazy ffmpeg.wasm only for video / when native decode fails
   - Run Whisper inference inside a Web Worker (keeps the UI responsive)
       * WebGPU when available, WASM fallback otherwise
   ========================================================================= */

/* ---- version pins (bump here if a CDN pairing breaks) ---- */
const TRANSFORMERS_VER = "4.2.0";
const FFMPEG_VER = "0.12.15";
const FFMPEG_UTIL_VER = "0.12.2";
const FFMPEG_CORE_VER = "0.12.10"; /* single-thread core: no SharedArrayBuffer needed */

/* ---- capability detection ---- */
const HAS_WEBGPU = (typeof navigator !== "undefined") && ("gpu" in navigator);
const DEVICE = HAS_WEBGPU ? "webgpu" : "wasm";
/* Per-model dtype, grounded in which ONNX variants actually build a session and in
   their download size:
   - large-v3-turbo: q4. Its fp32 encoder overflows the max single ArrayBuffer, so it
     must be quantized; its q4 weights are intact (~800 MB).
   - tiny/base/small: fp32 encoder + fp16 decoder. The encoder is the module Whisper is
     sensitive to, so it stays full precision; the decoder (the bulk of the download)
     runs at fp16 — which has no quantization scales, so it avoids the broken NBits
     embed_tokens that fails on q4/q8, is ~half the fp32 size, and is typically faster
     on WebGPU. Fixes the case where fp32 "small" (~950 MB) downloaded LARGER than q4
     turbo. The fp16 merged decoders are present and maintainer-validated. */
function resolveDtype(modelId) {
  if (/large/i.test(modelId)) return { encoder_model: "q4", decoder_model_merged: "q4" };
  return { encoder_model: "fp32", decoder_model_merged: "fp16" };
}

/* ---- model menu (all multilingual so language/translate work) ---- */
const MODELS = [
  { id: "onnx-community/whisper-tiny",           name: "tiny",           note: "fastest, lowest quality", size: "~80 MB" },
  { id: "onnx-community/whisper-base",           name: "base",           note: "fast",                    size: "~190 MB" },
  { id: "onnx-community/whisper-small",          name: "small",          note: "balanced",                size: "~650 MB" },
  { id: "onnx-community/whisper-large-v3-turbo", name: "large-v3-turbo", note: "best quality",            size: "~800 MB" }
];
/* turbo is great on GPU but heavy on CPU; pick a sane default per device */
const DEFAULT_MODEL = HAS_WEBGPU ? "onnx-community/whisper-large-v3-turbo" : "onnx-community/whisper-base";

/* ---- DOM ---- */
const el = (id) => document.getElementById(id);
const dom = {
  drop: el("drop"), file: el("file"), filemeta: el("filemeta"),
  model: el("model"), lang: el("lang"), task: el("task"),
  run: el("run"), device: el("device"), status: el("status"),
  bar: el("bar"), barFill: el("bar").firstElementChild,
  results: el("results"), partial: el("partial"),
  plaintext: el("plaintext"), segments: el("segments"),
  segwrap: el("segwrap"), metaline: el("metaline"),
  copy: el("copy"), dlTxt: el("dl-txt"), dlSrt: el("dl-srt"),
  pipeline: el("pipeline"), techNote: el("tech-note"),
  editAudio: el("edit-audio"), editTranscript: el("edit-transcript"), editLoad: el("edit-load"),
  editAudioZone: el("edit-audio-zone"), editTranscriptZone: el("edit-transcript-zone"),
  editAudioMeta: el("edit-audio-meta"), editTranscriptMeta: el("edit-transcript-meta"),
  editStatus: el("edit-status"), editBar: el("edit-bar"),
  tabTranscribe: el("tab-transcribe"), tabEdit: el("tab-edit"),
  panelTranscribe: el("panel-transcribe"), panelEdit: el("panel-edit")
};

/* ---- app state ---- */
const state = {
  file: null,
  task: "transcribe",
  loadedModel: null, /* which model the worker currently holds */
  busy: false,
  result: null,      /* { text, chunks, elapsed } */
  audio: null,       /* decoded 16kHz mono Float32Array, kept for per-segment playback */
  editAudio: null,       /* edit tab: chosen original media File (drop or click) */
  editTranscript: null   /* edit tab: chosen .srt/.txt File (drop or click) */
};

/* ---- device badge + model select init ---- */
/* =========================================================================
   Animated favicon: cycle the MSD mark M -> S -> D by swapping the
   <link rel="icon"> node. Paused (rests on the full MSD mark) while the tab is hidden so it
   never flickers in a background tab. Static head icon is the S frame. */
const FAVICON_M = "data:image/png;base64,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";
const FAVICON_D = "data:image/png;base64,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";
const FAVICON_MSD = "data:image/png;base64,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";
const FAVICON_S = "data:image/png;base64,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";
const FAVICON_FRAMES = [FAVICON_M, FAVICON_S, FAVICON_D, FAVICON_MSD, FAVICON_MSD];   /* M, S, D, then the full mark held an extra tick (2s) */
let favTimer = null, favIdx = FAVICON_FRAMES.length - 1;   /* MSD: the resting frame, shown statically */
function setFavicon(uri) {
  const cur = document.querySelector('link[rel="icon"]');
  const next = document.createElement("link");
  next.rel = "icon"; next.type = "image/png"; next.href = uri;
  if (cur && cur.parentNode) cur.parentNode.replaceChild(next, cur);
  else document.head.appendChild(next);
}
function startFaviconAnim() {
  if (favTimer) return;
  favTimer = setInterval(() => {
    favIdx = (favIdx + 1) % FAVICON_FRAMES.length;
    setFavicon(FAVICON_FRAMES[favIdx]);
  }, 1000);
}
function stopFaviconAnim() {
  if (favTimer) { clearInterval(favTimer); favTimer = null; }
  favIdx = FAVICON_FRAMES.length - 1; setFavicon(FAVICON_MSD);   /* rest on the full mark */
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopFaviconAnim(); else startFaviconAnim();
});

(function initUI() {
  dom.device.textContent = HAS_WEBGPU ? "WebGPU \u00b7 GPU" : "WASM \u00b7 CPU";
  dom.device.classList.add(HAS_WEBGPU ? "gpu" : "cpu");

  for (const m of MODELS) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name + " \u2014 " + m.note + " (" + m.size + ")";
    if (m.id === DEFAULT_MODEL) opt.selected = true;
    dom.model.appendChild(opt);
  }
  updateTechNote();
  console.log("[init] device=%s defaultModel=%s dtype=%o", DEVICE, DEFAULT_MODEL, resolveDtype(DEFAULT_MODEL));
  if (!document.hidden) startFaviconAnim();
})();

/* Footer note: name the selected model + its download size, and make the local-only
   nature explicit. Deliberately avoids naming libraries / the model host, which read
   as if the audio were sent somewhere. */
function updateTechNote() {
  if (!dom.techNote) return;
  const m = MODELS.find((x) => x.id === dom.model.value) || MODELS[0];
  const backend = HAS_WEBGPU ? "your GPU (WebGPU)" : "your CPU";
  dom.techNote.textContent =
    "Whisper " + m.name + " (" + m.size + " model) runs entirely in this page on " + backend +
    ". Your audio is transcribed on your device and is never uploaded.";
}

/* =========================================================================
   Worker: Transformers.js ASR pipeline (embedded as a module blob)
   ========================================================================= */
const WORKER_SOURCE = [
  'import { pipeline, WhisperTextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@' + TRANSFORMERS_VER + '";',
  'let transcriber = null;',
  'let loadedKey = null;',
  '',
  '/* ---- sequential path: live per-token streaming via WhisperTextStreamer ---- */',
  'async function transcribeStreaming(audio, language, task, chunkLengthS) {',
  '  const CHUNK_S = 30, STRIDE_S = 5, SR = 16000;',
  '  /* bundle mode (chunkLengthS<=0): the caller pre-segmented to <=30s, so this is one',
  '     full-context window with no internal chunking. chunked mode is the legacy path. */',
  '  const chunked = chunkLengthS && chunkLengthS > 0;',
  '  let totalWindows = 1;',
  '  if (chunked) {',
  '    const w = SR * CHUNK_S, s = SR * STRIDE_S, jump = w - 2 * s;',
  '    if (audio.length <= w) { totalWindows = 1; }',
  '    else { totalWindows = 0; let off = 0; while (true) { totalWindows++; if (off + w >= audio.length) break; off += jump; } }',
  '  }',
  /* no_repeat_ngram_size: TJS 4.2.0 ships no temperature-fallback recovery, so a chunk whose
     greedy decode enters a repetition loop is emitted verbatim. Banning any repeated 3-gram caps
     runaway repetition ("you you you", "thanks for watching x2") at the logits stage. n=3 is the
     lowest value that still leaves normal speech intact; the next lever (if loops persist) is a
     different axis -- repetition_penalty (~1.15) or a larger model -- not re-tuning this number. */
  '  const options = { return_timestamps: true, no_repeat_ngram_size: 3 };',
  '  if (chunked) { options.chunk_length_s = chunkLengthS; options.stride_length_s = STRIDE_S; }',
  '  if (language && language !== "auto") options.language = language;',
  '  options.task = task === "translate" ? "translate" : "transcribe";',
  '  let windowsDone = 0, live = "";',
  '  const sendProgress = (frac) => self.postMessage({ type: "progress", frac: frac, windowsDone: windowsDone, totalWindows: totalWindows });',
  '  try {',
  '    options.streamer = new WhisperTextStreamer(transcriber.tokenizer, {',
  '      skip_prompt: true,',
  '      callback_function: (t) => { live += t; self.postMessage({ type: "partial", text: live }); },',
  '      on_chunk_start: (t) => {',
  '        if (!chunked) return;',
  '        const onLastOfMany = totalWindows > 1 && windowsDone >= totalWindows - 1;',
  '        const nudge = onLastOfMany ? 0 : Math.min(t / CHUNK_S, 0.9);',
  '        sendProgress((windowsDone + nudge) / totalWindows);',
  '      },',
  '      on_finalize: () => { windowsDone++; live = ""; if (chunked) sendProgress(windowsDone / totalWindows); }',
  '    });',
  '  } catch (e) { self.postMessage({ type: "log", message: "streamer off: " + e.message }); }',
  '  const out = await transcriber(audio, options);',
  '  return { text: (out && out.text) || "", chunks: (out && out.chunks) || [] };',
  '}',
  '',
  'self.onmessage = async (event) => {',
  '  const msg = event.data;',
  '  try {',
  '    if (msg.type === "load") {',
  '      const key = msg.model + "|" + msg.device + "|" + JSON.stringify(msg.dtype);',
  '      if (transcriber && loadedKey === key) {',
  '        self.postMessage({ type: "ready", model: msg.model, cached: true });',
  '        return;',
  '      }',
  '      if (transcriber) {',
  '        try { await transcriber.dispose(); } catch (e) {}',
  '        transcriber = null; loadedKey = null;',
  '      }',
  '      transcriber = await pipeline("automatic-speech-recognition", msg.model, {',
  '        device: msg.device,',
  '        dtype: msg.dtype,',
  '        progress_callback: (p) => self.postMessage({ type: "download", payload: p })',
  '      });',
  '      loadedKey = key;',
  '      self.postMessage({ type: "ready", model: msg.model, cached: false });',
  '      return;',
  '    }',
  '',
  '    if (msg.type === "transcribe") {',
  '      if (!transcriber) { self.postMessage({ type: "error", message: "Model not loaded" }); return; }',
  '      const t0 = performance.now();',
  '      const result = await transcribeStreaming(msg.audio, msg.language, msg.task, msg.chunkLengthS || 0);',
  '      const elapsed = (performance.now() - t0) / 1000;',
  '      self.postMessage({ type: "complete", text: (result.text || "").trim(), chunks: result.chunks || [], elapsed: elapsed });',
  '      return;',
  '    }',
  '  } catch (err) {',
  '    self.postMessage({ type: "error", message: (err && err.message) ? err.message : String(err) });',
  '  }',
  '};'
].join("\n");

let worker = null;
const pending = { load: null, transcribe: null };
const download = { files: {}, };

function ensureWorker() {
  if (worker) return worker;
  const blob = new Blob([WORKER_SOURCE], { type: "text/javascript" });
  worker = new Worker(URL.createObjectURL(blob), { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    console.error("[worker:error]", e);
    const msg = e.message || "Worker failed (module workers require http(s), not file://)";
    if (pending.load) { pending.load.reject(new Error(msg)); pending.load = null; }
    if (pending.transcribe) { pending.transcribe.reject(new Error(msg)); pending.transcribe = null; }
  };
  console.log("[worker] created");
  return worker;
}

function onWorkerMessage(event) {
  const m = event.data;
  switch (m.type) {
    case "download": {
      const p = m.payload || {};
      if (p.status === "progress" && p.file) {
        download.files[p.file] = { loaded: p.loaded || 0, total: p.total || 0 };
        renderDownloadProgress();
      } else if (p.status === "initiate" && p.file) {
        download.files[p.file] = download.files[p.file] || { loaded: 0, total: 0 };
      }
      break;
    }
    case "ready":
      console.log("[worker] model ready (cached=%s)", m.cached);
      hideBar();
      if (pending.load) { pending.load.resolve(m); pending.load = null; }
      break;
    case "partial":
      /* running text for the current window; the clean merged text arrives in "complete" */
      dom.partial.textContent = (m.text || "");
      break;
    case "progress":
      /* chunks-done / total-chunks fraction; kept monotonic as a flicker guard */
      transcribeProgress.frac = Math.max(transcribeProgress.frac, m.frac || 0);
      transcribeProgress.windowsDone = m.windowsDone != null ? m.windowsDone : transcribeProgress.windowsDone;
      transcribeProgress.totalWindows = m.totalWindows || transcribeProgress.totalWindows;
      break;
    case "complete":
      console.log("[worker] complete in %ss", m.elapsed.toFixed(1));
      transcribeProgress.done = true;
      if (pending.transcribe) { pending.transcribe.resolve(m); pending.transcribe = null; }
      break;
    case "log":
      console.log("[worker:log]", m.message);
      break;
    case "error":
      console.error("[worker:error]", m.message);
      if (pending.load) { pending.load.reject(new Error(m.message)); pending.load = null; }
      if (pending.transcribe) { pending.transcribe.reject(new Error(m.message)); pending.transcribe = null; }
      break;
    default:
      break;
  }
}

function loadModel(model) {
  return new Promise((resolve, reject) => {
    pending.load = { resolve, reject };
    download.files = {};
    const dtype = resolveDtype(model);
    console.log("[load] model=%s device=%s dtype=%o", model, DEVICE, dtype);
    ensureWorker().postMessage({ type: "load", model, device: DEVICE, dtype });
  });
}

function transcribe(audio, language, task, chunkLengthS) {
  return new Promise((resolve, reject) => {
    pending.transcribe = { resolve, reject };
    /* NOTE: intentionally NOT transferring audio.buffer \u2014 we keep the decoded
       samples on the main thread for per-segment playback. The worker receives a
       structured-clone copy. The clone (~4 bytes/sample) is negligible vs inference. */
    ensureWorker().postMessage({ type: "transcribe", audio, language, task, chunkLengthS: chunkLengthS || 0 });
  });
}

/* =========================================================================
   Audio extraction -> 16 kHz mono Float32Array
   ========================================================================= */
const TARGET_RATE = 16000;

async function fileToMono16k(file, onStage) {
  const isVideo = (file.type || "").startsWith("video/");
  if (!isVideo) {
    try {
      return await decodeNative(file);
    } catch (err) {
      console.warn("[decode] native decode failed, falling back to ffmpeg:", err.message);
    }
  }
  onStage && onStage("Extracting audio with ffmpeg\u2026");
  return await decodeWithFFmpeg(file);
}

async function decodeNative(file) {
  const buf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let decoded;
  try {
    /* slice(0): decodeAudioData detaches the input buffer */
    decoded = await ctx.decodeAudioData(buf.slice(0));
  } finally {
    try { ctx.close(); } catch (e) {}
  }
  return resampleToMono16k(decoded);
}

async function resampleToMono16k(audioBuffer) {
  const frames = Math.max(1, Math.ceil(audioBuffer.duration * TARGET_RATE));
  /* 1-channel destination -> Web Audio downmixes to mono automatically */
  const offline = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = offline.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

let ffmpeg = null;
let ffmpegUtil = null;
let ffmpegLoading = null;

async function ensureFFmpeg() {
  if (ffmpeg) return;
  /* Coalesce concurrent callers onto one load. Crucially, only PUBLISH the
     instance to `ffmpeg` after load() succeeds \u2014 otherwise a failed load left a
     non-null-but-unloaded instance, and the guard above made the next attempt
     skip loading and call exec() on it, yielding the misleading
     "ffmpeg is not loaded" error instead of the real (network/CSP) failure. */
  if (!ffmpegLoading) {
    ffmpegLoading = (async () => {
      console.log("[ffmpeg] loading core %s", FFMPEG_CORE_VER);
      const ffMod = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@" + FFMPEG_VER + "/dist/esm/index.js");
      const util = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/util@" + FFMPEG_UTIL_VER + "/dist/esm/index.js");
      const inst = new ffMod.FFmpeg();
      inst.on("log", (e) => console.log("[ffmpeg]", e.message));
      const base = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@" + FFMPEG_CORE_VER + "/dist/esm";
      const ffEsm = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@" + FFMPEG_VER + "/dist/esm";
      try {
        /* @ffmpeg/ffmpeg spawns its own worker via new Worker(new URL("./worker.js",
           import.meta.url)). Loaded from a CDN that URL is cross-origin to the page,
           and new Worker() forbids cross-origin scripts ("Script at ... cannot be
           accessed from origin ..."). The fix is a SAME-ORIGIN blob worker that just
           imports the absolute worker.js: cross-origin module imports ARE allowed
           (unlike new Worker), and worker.js resolves its own ./const.js / ./errors.js
           against its absolute URL, not the blob. */
        const classWorkerURL = URL.createObjectURL(new Blob(
          ['import "' + ffEsm + '/worker.js";'],
          { type: "text/javascript" }
        ));
        const coreURL = await util.toBlobURL(base + "/ffmpeg-core.js", "text/javascript");
        const wasmURL = await util.toBlobURL(base + "/ffmpeg-core.wasm", "application/wasm");
        await inst.load({ classWorkerURL: classWorkerURL, coreURL: coreURL, wasmURL: wasmURL });
      } catch (e) {
        console.error("[ffmpeg] load failed:", e);
        throw new Error("could not load the ffmpeg audio extractor (" +
          ((e && e.message) ? e.message : String(e)) +
          ") \u2014 likely the page CSP or network is blocking the CDN core/worker/wasm");
      }
      ffmpegUtil = util;   /* publish both only on success */
      ffmpeg = inst;
      console.log("[ffmpeg] ready");
    })();
  }
  try {
    await ffmpegLoading;
  } finally {
    ffmpegLoading = null;   /* clear so a later attempt can retry whether this one passed or failed */
  }
}

async function decodeWithFFmpeg(file) {
  await ensureFFmpeg();
  const inName = "input_" + Date.now();
  await ffmpeg.writeFile(inName, await ffmpegUtil.fetchFile(file));
  /* extract: no video, mono, 16 kHz, raw 32-bit float little-endian */
  await ffmpeg.exec(["-i", inName, "-vn", "-ac", "1", "-ar", String(TARGET_RATE), "-f", "f32le", "out.raw"]);
  const data = await ffmpeg.readFile("out.raw"); /* Uint8Array */
  try { await ffmpeg.deleteFile(inName); await ffmpeg.deleteFile("out.raw"); } catch (e) {}
  /* copy into a tightly-aligned buffer before viewing as Float32 */
  const aligned = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const f32 = new Float32Array(aligned);
  if (!f32.length) throw new Error("ffmpeg produced no audio (no decodable audio track?)");
  return f32;
}

/* =========================================================================
   Orchestration
   ========================================================================= */
function setStage(name, mode) {
  const node = dom.pipeline.querySelector('[data-stage="' + name + '"]');
  if (!node) return;
  node.classList.remove("active", "done");
  if (mode) node.classList.add(mode);
}
function resetStages() { ["decode", "model", "transcribe", "present"].forEach((s) => setStage(s, null)); }

function setStatusOn(node, text, isErr) {
  node.innerHTML = isErr ? '<span class="err">' + text + "</span>" : text;
}
function setStatus(text, isErr) { setStatusOn(dom.status, text, isErr); }
function setEditStatus(text, isErr) { setStatusOn(dom.editStatus, text, isErr); }
function showBar() { dom.bar.classList.add("show"); }
function hideBar() { dom.bar.classList.remove("show"); dom.barFill.style.width = "0%"; }

function renderDownloadProgress() {
  let loaded = 0, total = 0;
  for (const k in download.files) { loaded += download.files[k].loaded; total += download.files[k].total; }
  if (total > 0) {
    const pct = Math.min(100, (loaded / total) * 100);
    showBar();
    dom.barFill.style.width = pct.toFixed(1) + "%";
    setStatus("Downloading model \u00b7 " + fmtBytes(loaded) + " / " + fmtBytes(total) + " (" + pct.toFixed(0) + "%)");
  }
}

/* Transcription progress: honest chunks-done / total-chunks, plus a ticking
   elapsed clock so a slow chunk still reads as active work. The interval is the
   single writer of #status while running. A slow final chunk now shows e.g.
   "97% (36/37 chunks)" truthfully instead of a frozen "12:13 / 12:13 (99%)". */
const transcribeProgress = { frac: 0, windowsDone: 0, totalWindows: 0, done: false };
let transcribeTimer = null;
function startTranscribeUI() {
  transcribeProgress.frac = 0;
  transcribeProgress.windowsDone = 0;
  transcribeProgress.totalWindows = 0;
  transcribeProgress.done = false;
  const t0 = performance.now();
  showBar();
  const tick = () => {
    const wd = transcribeProgress.windowsDone || 0;
    const tw = transcribeProgress.totalWindows || 0;
    /* hold below 100% until the "complete" message: the final chunk's on_finalize
       can read 100% a beat before merge finishes, and 100%-while-running is the lie. */
    const pct = transcribeProgress.done ? 100 : Math.min(99, (transcribeProgress.frac || 0) * 100);
    const elapsed = (performance.now() - t0) / 1000;
    dom.barFill.style.width = pct.toFixed(1) + "%";
    let label;
    if (tw && wd >= tw) {
      /* every chunk streamed; the pipeline is merging/decoding the final result */
      label = "Finalizing transcript\u2026 \u00b7 " + elapsed.toFixed(0) + "s elapsed";
    } else {
      label = "Transcribing \u00b7 " + pct.toFixed(0) + "%"
            + (tw ? " (" + wd + "/" + tw + " chunks)" : "")
            + " \u00b7 " + elapsed.toFixed(0) + "s elapsed";
    }
    setStatus(label);
  };
  tick();
  transcribeTimer = setInterval(tick, 250);
}
function stopTranscribeUI() {
  if (transcribeTimer) { clearInterval(transcribeTimer); transcribeTimer = null; }
}

/* =========================================================================
   VAD chunking: detect speech regions, then pack them into <=30s bundles so
   each transcribe call is one full-context window (silence dropped, no stride
   overlap). Replaces fixed-window chunking on the fresh path. Validated against
   real meeting audio: ~35% less compute and cleaner boundaries (no overlap
   re-stitching) vs the old fixed-window path. Pure functions (packBundles,
   materializeBundle, mapBundleTime) are node-validated.
   ========================================================================= */
const VAD_PARAMS = { threshold: 0.015, minSilenceMs: 1200, minSpeechMs: 800, padMs: 250, maxRegionS: 28, bundleTargetS: 30, seamS: 0.16 };

/* energy VAD -> [ [startSec, endSec], ... ], each region <= maxRegionS */
function detectSpeech(audio, sr, p) {
  const frame = Math.max(1, Math.round(sr * 0.030)); /* 30 ms frames */
  const n = audio.length;
  const flags = [];
  for (let i = 0; i < n; i += frame) {
    const end = Math.min(n, i + frame);
    let sum = 0;
    for (let j = i; j < end; j++) sum += audio[j] * audio[j];
    flags.push(Math.sqrt(sum / (end - i)) >= p.threshold);
  }
  const frameMs = 30;
  const minSil = Math.max(1, Math.round(p.minSilenceMs / frameMs));
  const minSp = Math.max(1, Math.round(p.minSpeechMs / frameMs));
  const pad = Math.round(p.padMs / frameMs);
  const runs = [];
  let i = 0;
  while (i < flags.length) {
    if (!flags[i]) { i++; continue; }
    let j = i + 1;
    while (j < flags.length) {
      if (flags[j]) { j++; continue; }
      let k = j; while (k < flags.length && !flags[k]) k++;
      if ((k - j) < minSil) { j = k; } else break; /* bridge short pauses */
    }
    runs.push([i, j]); i = j;
  }
  const out = [];
  for (const run of runs) {
    const a = run[0], b = run[1];
    if ((b - a) < minSp) continue; /* drop blips */
    let startS = Math.max(0, (a - pad)) * frame / sr;
    let endS = Math.min(n, (b + pad) * frame) / sr;
    while (endS - startS > p.maxRegionS + 0.01) {
      out.push([startS, startS + p.maxRegionS]);
      startS += p.maxRegionS;
    }
    out.push([startS, endS]);
  }
  return out;
}

/* greedy first-fit pack of regions into bundles <= targetS (with seams between) */
function packBundles(regions, sr, targetS, seamS) {
  const targetSamples = Math.round(targetS * sr);
  const seamSamples = Math.max(0, Math.round(seamS * sr));
  const bundles = [];
  let cur = null;
  for (const reg of regions) {
    const a = Math.round(reg[0] * sr), b = Math.round(reg[1] * sr);
    const rs = b - a;
    const add = (cur && cur.regs.length ? seamSamples : 0) + rs;
    if (cur && cur.regs.length && (cur.samples + add) > targetSamples) { bundles.push(cur); cur = null; }
    if (!cur) cur = { regs: [], samples: 0 };
    cur.samples += (cur.regs.length ? seamSamples : 0) + rs;
    cur.regs.push([a, b, reg[0]]);
  }
  if (cur && cur.regs.length) bundles.push(cur);
  return bundles;
}

/* concat a bundle's regions into one buffer + a sample-exact time map (segs).
   each seg = { b0, b1 (bundle-relative secs), o0 (original start sec) }. */
function materializeBundle(bundle, audio, sr, seamSamples) {
  let samples = 0;
  for (let i = 0; i < bundle.regs.length; i++) {
    samples += (bundle.regs[i][1] - bundle.regs[i][0]);
    if (i) samples += seamSamples;
  }
  const buffer = new Float32Array(samples);
  const segs = [];
  let cursor = 0;
  for (let i = 0; i < bundle.regs.length; i++) {
    const a = bundle.regs[i][0], b = bundle.regs[i][1], o0 = bundle.regs[i][2];
    if (i) cursor += seamSamples; /* leave the seam silent */
    buffer.set(audio.subarray(a, b), cursor);
    const rs = b - a;
    segs.push({ b0: cursor / sr, b1: (cursor + rs) / sr, o0 });
    cursor += rs;
  }
  return { buffer, segs };
}

/* map a bundle-relative timestamp back to original-audio seconds. single-region
   bundle reduces to o0 + bt; a timestamp in a seam snaps to the next region. */
function mapBundleTime(bt, segs) {
  if (bt == null) return null;
  if (bt <= segs[0].b0) return segs[0].o0;
  for (let k = 0; k < segs.length; k++) {
    const s = segs[k];
    if (bt <= s.b1) return s.o0 + (bt - s.b0);
    const next = segs[k + 1];
    if (next && bt < next.b0) return next.o0;
  }
  const last = segs[segs.length - 1];
  return last.o0 + (last.b1 - last.b0);
}

async function run() {
  if (state.busy || !state.file) return;
  state.busy = true;
  dom.run.disabled = true;
  maybeRequestNotify(); acquireWakeLock();
  dom.results.classList.remove("show");
  dom.partial.textContent = "";
  dom.plaintext.textContent = "";
  dom.segments.innerHTML = "";
  dom.metaline.textContent = "";
  stopSpan();
  teardownElementPlayback();      /* a fresh run uses the buffer backend; release any edit-mode element/URL */
  audioPlayback.mode = "buffer";
  audioPlayback.buffer = null;    /* new run -> rebuild from the new audio on first play */
  state.audio = null;
  resetStages();

  const model = dom.model.value;
  const language = dom.lang.value;
  const task = state.task;

  try {
    /* stages 1 + 2 overlap: decode audio while the model downloads */
    setStage("decode", "active");
    setStage("model", "active");
    setStatus("Preparing model and extracting audio\u2026");

    const decodePromise = fileToMono16k(state.file, (s) => setStatus(s))
      .then((audio) => { setStage("decode", "done"); return audio; });

    const loadPromise = (state.loadedModel === model)
      ? Promise.resolve()
      : loadModel(model).then(() => { state.loadedModel = model; });

    const [audio] = await Promise.all([decodePromise, loadPromise]);
    setStage("model", "done");
    hideBar();

    const seconds = audio.length / TARGET_RATE;
    state.audio = audio;   /* retained for per-segment playback */
    console.log("[run] audio ready: %d samples (%ss)", audio.length, seconds.toFixed(1));

    /* stage 3: detect speech -> pack into <=30s bundles -> transcribe each bundle
       as one full-context window, rendering segments as each bundle lands. The
       growing, editable transcript doubles as the progress indicator. */
    setStage("transcribe", "active");
    dom.results.classList.add("show");
    setStatus("Detecting speech\u2026");
    const regions = detectSpeech(audio, TARGET_RATE, VAD_PARAMS);
    const bundles = packBundles(regions, TARGET_RATE, VAD_PARAMS.bundleTargetS, VAD_PARAMS.seamS);
    const speechS = regions.reduce((acc, r) => acc + (r[1] - r[0]), 0);
    const seamSamples = Math.round(VAD_PARAMS.seamS * TARGET_RATE);
    console.log("[vad] %d regions -> %d bundles, %ss speech / %ss total",
      regions.length, bundles.length, speechS.toFixed(0), seconds.toFixed(0));

    /* editable, growing surface (mirrors renderResult's fresh setup, but additive) */
    dom.plaintext.style.display = "none";
    dom.segments.innerHTML = "";
    dom.segwrap.style.display = "";
    dom.segwrap.open = true;
    const head = document.querySelector(".result-head h2");
    if (head) head.textContent = "Transcript";

    startTranscribeUI();
    transcribeProgress.totalWindows = bundles.length || 1;

    state.result = { text: "", chunks: [], elapsed: 0 };
    let elapsed = 0;
    for (let r = 0; r < bundles.length; r++) {
      dom.partial.textContent = "";
      const built = materializeBundle(bundles[r], audio, TARGET_RATE, seamSamples);
      const spanS = built.buffer.length / TARGET_RATE;
      const out = await transcribe(built.buffer, language, task, spanS > 30 ? 30 : 0);
      elapsed += out.elapsed;

      const from = state.result.chunks.length;
      let lastStart = built.segs[0].o0;
      for (const c of (out.chunks || [])) {
        const ts = c.timestamp || [null, null];
        const st = ts[0] != null ? mapBundleTime(ts[0], built.segs) : lastStart;
        const en = ts[1] != null ? mapBundleTime(ts[1], built.segs) : null;
        lastStart = st;
        state.result.chunks.push({ timestamp: [st, en], text: c.text });
      }
      appendSegments(state.result.chunks, from, seconds, true);
      syncEditedText();

      transcribeProgress.windowsDone = r + 1;
      transcribeProgress.frac = (r + 1) / (bundles.length || 1);
    }
    transcribeProgress.done = true;
    state.result.elapsed = elapsed;
    dom.partial.textContent = "";
    stopTranscribeUI();
    hideBar();
    setStage("transcribe", "done");

    /* stage 4 */
    setStage("present", "done");
    if (!state.result.chunks.length) {
      dom.plaintext.style.display = "";
      dom.plaintext.textContent = "(no speech detected)";
      dom.segwrap.style.display = "none";
    }
    setResultMeta(state.result, seconds, true);
    setStatus("Done in " + elapsed.toFixed(1) + "s \u00b7 " + bundles.length + " bundles, " + state.result.chunks.length + " segments.");
    notify("Transcription complete", fmtClock(seconds) + " transcribed in " + elapsed.toFixed(1) + "s", false);
  } catch (err) {
    console.error("[run] failed:", err);
    stopTranscribeUI();
    hideBar();
    setStage("transcribe", null);
    setStatus(friendlyError(err), true);
    notify("Transcription failed", friendlyError(err), true);
  } finally {
    stopTranscribeUI();
    releaseWakeLock();
    state.busy = false;
    dom.run.disabled = !state.file;
  }
}

function friendlyError(err) {
  const m = (err && err.message) || String(err);
  if (/file:\/\/|module worker|importScripts/i.test(m)) return "This page must be served over http(s) \u2014 open it from a local/static server, not the filesystem.";
  if (/webgpu|gpu adapter/i.test(m)) return "WebGPU initialization failed: " + m + ". Try a smaller model or a Chromium browser.";
  if (/out of memory|allocation|oom|array buffer/i.test(m)) return "Not enough memory to load this model \u2014 switch to a smaller model (small / base / tiny) and try again.";
  if (/ffmpeg/i.test(m)) return "Audio extraction failed: " + m;
  return "Error: " + m;
}

/* ---- result rendering ---- */
/* =========================================================================
   Per-segment audio playback. Plays the decoded 16 kHz mono input (exactly what
   the model heard) for a given [start,end] span, so a questionable segment can be
   checked against the audio. Robust for both audio and video inputs and guaranteed
   to share the transcript's timeline. Low-fi by design (it's the model's input).
   ========================================================================= */
/* Two playback backends behind playSpan/stopSpan:
   - "buffer": fresh transcription slices the already-decoded 16kHz mono state.audio
     (sample-accurate, and free since the buffer is a transcription byproduct).
   - "element": the edit tab seeks the ORIGINAL file in an <audio> element, so loading
     skips the full decode/resample entirely and playback is full-fidelity.
   Mode is set by run() (buffer) and loadForEditing (element); the displayed result always
   matches the active mode because each path rewrites the results panel as it sets the mode. */
const audioPlayback = { ctx: null, buffer: null, source: null, button: null, mode: "buffer", el: null, url: null, onTime: null };

function playbackReady() {
  return audioPlayback.mode === "element" ? !!audioPlayback.el : !!(state.audio && state.audio.length);
}

function setupElementPlayback(file) {
  teardownElementPlayback();
  audioPlayback.buffer = null;   /* release any prior buffer backend */
  state.audio = null;
  const url = URL.createObjectURL(file);
  const el = new Audio();
  el.preload = "metadata";
  el.src = url;
  audioPlayback.mode = "element";
  audioPlayback.el = el;
  audioPlayback.url = url;
  return el;
}

function teardownElementPlayback() {
  if (audioPlayback.el) { try { audioPlayback.el.pause(); } catch (e) {} }
  if (audioPlayback.url) { try { URL.revokeObjectURL(audioPlayback.url); } catch (e) {} }
  audioPlayback.el = null;
  audioPlayback.url = null;
  audioPlayback.onTime = null;
}

function ensurePlayBuffer() {
  if (audioPlayback.buffer) return audioPlayback.buffer;
  if (!state.audio || !state.audio.length) return null;
  if (!audioPlayback.ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioPlayback.ctx = new AC();
  }
  const buf = audioPlayback.ctx.createBuffer(1, state.audio.length, TARGET_RATE);
  buf.copyToChannel(state.audio, 0);
  audioPlayback.buffer = buf;
  return buf;
}

function setPlayButtonState(btn, playing) {
  if (!btn) return;
  btn.classList.toggle("playing", playing);
  const glyph = btn.querySelector(".glyph");
  if (glyph) glyph.textContent = playing ? "\u25A0" : "\u25B6"; /* stop square / play triangle */
}

function stopSpan() {
  if (audioPlayback.source) {   /* buffer backend */
    try { audioPlayback.source.onended = null; audioPlayback.source.stop(); } catch (e) {}
    audioPlayback.source = null;
  }
  if (audioPlayback.el) {       /* element backend */
    try { audioPlayback.el.pause(); } catch (e) {}
    if (audioPlayback.onTime) { audioPlayback.el.removeEventListener("timeupdate", audioPlayback.onTime); audioPlayback.onTime = null; }
    audioPlayback.el.onended = null;
  }
  if (audioPlayback.button) { setPlayButtonState(audioPlayback.button, false); audioPlayback.button = null; }
}

function playSpan(start, end, btn) {
  /* clicking the row that is already playing stops it (button is the active indicator for both backends) */
  if (audioPlayback.button === btn) { stopSpan(); return; }
  stopSpan();
  if (audioPlayback.mode === "element") playSpanElement(start, end, btn);
  else playSpanBuffer(start, end, btn);
}

function playSpanBuffer(start, end, btn) {
  const buf = ensurePlayBuffer();
  if (!buf) { console.warn("[play] no decoded audio available"); return; }
  const ctx = audioPlayback.ctx;
  if (ctx.state === "suspended") { ctx.resume().catch((e) => console.warn("[play] resume failed", e)); }
  const s = Math.max(0, start || 0);
  const e = (end != null && end > s) ? end : buf.duration;
  const dur = Math.min(buf.duration - s, e - s);
  if (dur <= 0) { console.warn("[play] empty span"); return; }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => { if (audioPlayback.source === src) stopSpan(); };
  try { src.start(0, s, dur); } catch (e) { console.error("[play] start failed", e); return; }
  audioPlayback.source = src;
  audioPlayback.button = btn;
  setPlayButtonState(btn, true);
  console.log("[play:buffer] %ss .. %ss (%ss)", s.toFixed(2), (s + dur).toFixed(2), dur.toFixed(2));
}

function playSpanElement(start, end, btn) {
  const el = audioPlayback.el;
  if (!el) { console.warn("[play] no audio element"); return; }
  const s = Math.max(0, start || 0);
  const e = (end != null && end > s) ? end : (isFinite(el.duration) ? el.duration : Infinity);
  /* timeupdate fires ~4x/s, so the span end is honored within ~250ms -- fine for spot-checking */
  const onTime = () => { if (el.currentTime >= e) stopSpan(); };
  audioPlayback.onTime = onTime;
  el.addEventListener("timeupdate", onTime);
  el.onended = () => { if (audioPlayback.button === btn) stopSpan(); };
  audioPlayback.button = btn;
  setPlayButtonState(btn, true);
  try { el.currentTime = s; } catch (err) { console.warn("[play] seek failed", err); }
  el.play().catch((err) => { console.error("[play:element] play failed", err); stopSpan(); });
  console.log("[play:element] %ss .. %ss", s.toFixed(2), (e === Infinity ? "end" : e.toFixed(2)));
}

/* Whisper segment END times are unreliable (often truncated, with the spoken audio
   spilling into the gap before the next segment). The effective end of a segment is the
   NEXT segment's start; the last runs to the end of the decoded audio. Start times are the
   trustworthy anchors. Returns null when neither a later start nor an audio length is known
   (caller decides the fallback). Single source for display, playback, and SRT export. */
function effectiveEnd(chunks, i, audioSeconds) {
  const ts = chunks[i] && chunks[i].timestamp;
  if (!ts || ts[0] == null) return null;
  const nxt = chunks[i + 1];
  const nextStart = (nxt && nxt.timestamp && nxt.timestamp.length) ? nxt.timestamp[0] : null;
  if (nextStart != null && nextStart > ts[0]) return nextStart;
  if (audioSeconds && audioSeconds > ts[0]) return audioSeconds;
  return null;
}

/* build one segment row. play-end is recomputed live from the chunks array so it
   stays correct as later bundles refine the boundary. */
function makeSegRow(chunks, ci, audioSeconds, editable) {
  const c = chunks[ci];
  const ts = (c.timestamp && c.timestamp.length) ? c.timestamp : [null, null];
  const row = document.createElement("div");
  row.className = "seg-row";

  const btn = document.createElement("button");
  btn.className = "ts";
  btn.type = "button";
  const glyph = document.createElement("span");
  glyph.className = "glyph";
  glyph.textContent = "\u25B6";
  const label = document.createElement("span");
  const refresh = function () {
    const pe = effectiveEnd(chunks, ci, audioSeconds);
    label.textContent = fmtClock(ts[0]) + (pe != null ? " \u2192 " + fmtClock(pe) : "");
  };
  refresh();
  btn.appendChild(glyph);
  btn.appendChild(label);
  if (ts[0] == null || !playbackReady()) {
    btn.disabled = true;
    btn.title = "No audio available for playback";
  } else {
    btn.title = "Play this span";
    btn.addEventListener("click", () => playSpan(ts[0], effectiveEnd(chunks, ci, audioSeconds), btn));
  }

  const x = document.createElement("div");
  x.className = "tx";
  x.textContent = (c.text || "").trim();
  if (editable) {
    x.contentEditable = "true";
    x.spellcheck = true;
    x.classList.add("editing");
    x.addEventListener("input", function () { c.text = x.textContent; syncEditedText(); });
  }

  row.appendChild(btn);
  row.appendChild(x);
  row._refreshLabel = refresh;   /* used to refine the boundary end when a successor arrives */
  return row;
}

/* append rows for chunks[fromIndex..end). purely additive: existing rows (and any
   in-progress edits) are never rebuilt; only the prior boundary row's end label is
   refreshed now that it has a successor. */
function appendSegments(chunks, fromIndex, audioSeconds, editable) {
  if (fromIndex > 0) {
    const prev = dom.segments.children[fromIndex - 1];
    if (prev && prev._refreshLabel) prev._refreshLabel();
  }
  for (let ci = fromIndex; ci < chunks.length; ci++) {
    dom.segments.appendChild(makeSegRow(chunks, ci, audioSeconds, editable));
  }
  if (chunks.length) dom.segwrap.style.display = "";
}

/* heading + meta line, shared by the fresh (stats) and loaded-for-editing paths */
function setResultMeta(out, audioSeconds, editable) {
  const h2 = document.querySelector(".result-head h2");
  const hasStats = (out.elapsed != null);   /* fresh runs carry compute stats; loaded-for-editing do not */
  if (h2) h2.textContent = hasStats ? "Transcript" : "Edit transcript";

  if (!hasStats) {
    dom.metaline.textContent = (out.chunks ? out.chunks.length : 0) + " segments - " + fmtClock(audioSeconds) + " audio - editing (downloads reflect your edits)";
    return;
  }

  const rt = audioSeconds > 0 ? (audioSeconds / out.elapsed).toFixed(1) + "\u00d7 real-time" : "";
  dom.metaline.textContent = [
    dom.model.value.split("/").pop(),
    DEVICE.toUpperCase(),
    fmtClock(audioSeconds) + " audio",
    out.elapsed.toFixed(1) + "s compute",
    rt,
    editable ? "editable (downloads reflect your edits)" : ""
  ].filter(Boolean).join("  \u00b7  ");
}

function renderResult(out, audioSeconds, editable) {
  dom.partial.textContent = "";
  if (editable) {
    dom.plaintext.style.display = "none";
  } else {
    dom.plaintext.style.display = "";
    dom.plaintext.textContent = out.text || "(no speech detected)";
  }

  dom.segments.innerHTML = "";
  if (out.chunks && out.chunks.length) {
    appendSegments(out.chunks, 0, audioSeconds, editable);
  } else {
    dom.segwrap.style.display = "none";
  }

  if (editable) dom.segwrap.open = true;
  setResultMeta(out, audioSeconds, editable);
}

/* ---- helpers ---- */
function fmtBytes(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"]; let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(i ? 1 : 0) + " " + u[i];
}
function fmtClock(sec) {
  if (sec == null || isNaN(sec)) return "--:--";
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const pad = (x) => String(x).padStart(2, "0");
  return (h ? pad(h) + ":" : "") + pad(m) + ":" + pad(s);
}
function fmtSrtTime(sec) {
  sec = Math.max(0, sec || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec - Math.floor(sec)) * 1000);
  const pad = (x, n) => String(x).padStart(n || 2, "0");
  return pad(h) + ":" + pad(m) + ":" + pad(s) + "," + pad(ms, 3);
}
function toSrt(chunks, audioSeconds, normalize) {
  if (!chunks || !chunks.length) return "";
  return chunks.map((c, i) => {
    const ts = c.timestamp || [0, 0];
    /* fresh transcripts: rewrite the end to the displayed effective span (next-start / audio end).
       loaded .srt (normalize=false): keep the authored end untouched. */
    const eff = normalize ? effectiveEnd(chunks, i, audioSeconds) : null;
    const end = eff != null ? eff : (ts[1] != null ? ts[1] : ts[0]);
    return (i + 1) + "\n" + fmtSrtTime(ts[0]) + " --> " + fmtSrtTime(end) + "\n" + (c.text || "").trim() + "\n";
  }).join("\n");
}
function downloadBlob(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function baseName() {
  const n = (state.file && state.file.name) || "transcript";
  return n.replace(/\.[^.]+$/, "") || "transcript";
}

/* =========================================================================
   Wake Lock + system notifications (best-effort)
   Long jobs let the user look away; keep the screen awake while the tab is
   visible, and ping them on finish if the tab is in the background.
   ========================================================================= */
let wakeLock = null;
async function acquireWakeLock() {
  try {
    if (navigator.wakeLock && navigator.wakeLock.request && !document.hidden) {
      wakeLock = await navigator.wakeLock.request("screen");
      console.log("[wakelock] acquired");
    }
  } catch (e) { console.log("[wakelock] unavailable:", e && e.message); }
}
async function releaseWakeLock() {
  try { if (wakeLock) { await wakeLock.release(); console.log("[wakelock] released"); } }
  catch (e) {} finally { wakeLock = null; }
}
function maybeRequestNotify() {
  try {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().catch(function () {});
    }
  } catch (e) {}
}
function notify(title, body, isError) {
  try {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    if (!document.hidden) return;   /* tab is in view -> the in-page status already says it */
    const n = new Notification(title, { body: body, icon: FAVICON_S, tag: "msd-transcriber" });
    n.onclick = function () { try { window.focus(); } catch (e) {} n.close(); };
  } catch (e) { console.log("[notify] failed:", e && e.message); }
}

/* =========================================================================
   Edit an existing transcript: load a prior audio/video file plus its
   transcript (.srt keeps timestamps + per-segment playback; .txt loads as
   plain lines), edit the segment text in place, then re-download. Reuses the
   results view, per-segment playback, and the .txt/.srt/.wav exporters.
   ========================================================================= */
function parseSrt(text) {
  const blocks = text.replace(/\r/g, "").split(/\n\n+/);
  const chunks = [];
  const toSec = function (h, m, sec, ms) { return (+h) * 3600 + (+m) * 60 + (+sec) + (+ms) / 1000; };
  for (const b of blocks) {
    const lines = b.split("\n").filter(function (l) { return l.trim() !== ""; });
    if (!lines.length) continue;
    let i = 0;
    if (/^\d+$/.test(lines[0].trim())) i = 1;   /* skip the optional index line */
    const tline = lines[i] || "";
    const m = tline.match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/);
    if (!m) { chunks.push({ timestamp: [null, null], text: lines.slice(i).join(" ") }); continue; }
    chunks.push({ timestamp: [toSec(m[1], m[2], m[3], m[4]), toSec(m[5], m[6], m[7], m[8])], text: lines.slice(i + 1).join(" ").trim() });
  }
  return chunks;
}
function parseTxtToChunks(text) {
  const lines = text.replace(/\r/g, "").split("\n").map(function (l) { return l.trim(); }).filter(Boolean);
  if (!lines.length) return [{ timestamp: [null, null], text: text.trim() }];
  return lines.map(function (l) { return { timestamp: [null, null], text: l }; });
}
/* =========================================================================
   Edit-mode text sync: recompose the flowing transcript from the per-segment
   chunks after any segment edit (or after loading an external .srt/.txt).
   ========================================================================= */
function syncEditedText() {
  if (!state.result || !state.result.chunks) return;
  state.result.text = state.result.chunks.map(function (c) { return (c.text || "").trim(); }).filter(Boolean).join(" ");
}
async function loadForEditing() {
  const af = state.editAudio;
  const tf = state.editTranscript;
  if (!af || !tf) return;
  dom.editLoad.disabled = true;
  dom.editBar.classList.add("show", "indeterminate");
  try {
    stopSpan();
    setEditStatus("Reading media\u2026");
    /* Edit mode plays the original file directly (no Whisper here), so seek an <audio>
       element instead of decoding/resampling. Only metadata is read -> near-instant. */
    const el = setupElementPlayback(af);
    const playable = await new Promise((res) => {
      if (el.readyState >= 1 && isFinite(el.duration)) return res(true);
      el.addEventListener("loadedmetadata", () => res(true), { once: true });
      el.addEventListener("error", () => res(false), { once: true });
    });
    let audioSeconds;
    if (playable) {
      audioSeconds = isFinite(el.duration) ? el.duration : 0;
    } else {
      /* the element can't play this container/codec (e.g. mkv): fall back to the decode
         path so playback still works, at the cost of the resample wait. */
      console.warn("[edit] element playback unsupported for this file, decoding for playback");
      teardownElementPlayback();
      audioPlayback.mode = "buffer";
      audioPlayback.buffer = null;
      setEditStatus("Decoding audio\u2026 long recordings can take a moment.");
      const audio = await fileToMono16k(af, function (st) { setEditStatus(st); });
      state.audio = audio;
      audioSeconds = audio.length / TARGET_RATE;
    }
    acceptFile(af);   /* register as the working file (re-download naming, optional re-transcribe) */
    setEditStatus("Reading transcript\u2026");
    const text = await tf.text();
    const isSrt = /\.srt$/i.test(tf.name) || /-->/.test(text);
    const chunks = isSrt ? parseSrt(text) : parseTxtToChunks(text);
    state.result = { text: "", chunks: chunks, elapsed: null };
    syncEditedText();
    dom.results.classList.add("show");
    renderResult(state.result, audioSeconds, true);
    setEditStatus("Loaded " + chunks.length + " segment(s). Edit text, play any segment to check it, then download.");
    dom.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error("[edit] load failed:", err);
    setEditStatus(friendlyError(err), true);
  } finally {
    dom.editBar.classList.remove("show", "indeterminate");
    dom.editLoad.disabled = !(state.editAudio && state.editTranscript);
  }
}

/* =========================================================================
   Wiring
   ========================================================================= */
function acceptFile(file) {
  if (!file) return;
  state.file = file;
  dom.drop.classList.add("has-file");
  dom.filemeta.hidden = false;
  dom.filemeta.innerHTML = "<b>" + file.name + "</b> \u00b7 " + fmtBytes(file.size) + (file.type ? " \u00b7 " + file.type : "");
  dom.run.disabled = state.busy;
  setStatus("Ready.");
  console.log("[file] %s (%s, %d bytes)", file.name, file.type, file.size);
}

/* Reusable drag-drop / click-to-pick zone: wires a zone element to its hidden file input
   and invokes onFile(file) on a drop or a pick. One definition shared by the transcribe
   picker and both edit pickers, so all three behave identically. */
function makeDropZone(zone, input, onFile) {
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => onFile(e.target.files && e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("hover"); }));
  zone.addEventListener("dragleave", (e) => { e.preventDefault(); zone.classList.remove("hover"); });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("hover");
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onFile(f);
  });
}

/* edit tab: the File is the source of truth (drop and click feed the same slot), so the
   zone shows the chosen name and loadForEditing reads state, not the input element. */
function acceptEditFile(slot, file) {
  if (!file) return;
  const zone = slot === "editAudio" ? dom.editAudioZone : dom.editTranscriptZone;
  const meta = slot === "editAudio" ? dom.editAudioMeta : dom.editTranscriptMeta;
  state[slot] = file;
  zone.classList.add("has-file");
  meta.hidden = false;
  meta.innerHTML = "<b>" + file.name + "</b> \u00b7 " + fmtBytes(file.size);
  console.log("[edit-file] %s = %s (%d bytes)", slot, file.name, file.size);
  updateEditLoadState();
}

makeDropZone(dom.drop, dom.file, acceptFile);
makeDropZone(dom.editAudioZone, dom.editAudio, (f) => acceptEditFile("editAudio", f));
makeDropZone(dom.editTranscriptZone, dom.editTranscript, (f) => acceptEditFile("editTranscript", f));
dom.model.addEventListener("change", updateTechNote);

dom.task.addEventListener("change", () => {
  const r = dom.task.querySelector('input[name="task"]:checked');
  if (r) state.task = r.value;
});

dom.run.addEventListener("click", run);

dom.copy.addEventListener("click", async () => {
  const text = (state.result && state.result.text) || "";
  try { await navigator.clipboard.writeText(text); dom.copy.textContent = "Copied"; }
  catch (e) {
    /* clipboard API can throw if document is not focused; fall back to execCommand */
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); dom.copy.textContent = "Copied"; }
    catch (e2) { dom.copy.textContent = "Copy failed"; }
    ta.remove();
  }
  setTimeout(() => (dom.copy.textContent = "Copy"), 1400);
});
dom.dlTxt.addEventListener("click", () => {
  if (state.result) downloadBlob(baseName() + ".txt", state.result.text || "", "text/plain");
});
dom.dlSrt.addEventListener("click", () => {
  if (!state.result) return;
  const audioSeconds = state.audio ? state.audio.length / TARGET_RATE : 0;
  /* normalize only for fresh transcripts (elapsed != null); a loaded .srt keeps its authored ends */
  const normalize = state.result.elapsed != null;
  downloadBlob(baseName() + ".srt", toSrt(state.result.chunks, audioSeconds, normalize), "text/plain");
});

function updateEditLoadState() {
  dom.editLoad.disabled = !(state.editAudio && state.editTranscript);
}
dom.editLoad.addEventListener("click", loadForEditing);

/* mode tabs: swap the visible input panel. The 01-04 pipeline is transcribe-only,
   so hide it on the edit tab; the edit panel carries its own progress feedback. */
function selectTab(mode) {
  const isEdit = mode === "edit";
  dom.tabTranscribe.setAttribute("aria-selected", String(!isEdit));
  dom.tabEdit.setAttribute("aria-selected", String(isEdit));
  dom.panelTranscribe.hidden = isEdit;
  dom.panelEdit.hidden = !isEdit;
  dom.pipeline.hidden = isEdit;
  console.log("[tab] mode =", mode);
}
dom.tabTranscribe.addEventListener("click", () => selectTab("transcribe"));
dom.tabEdit.addEventListener("click", () => selectTab("edit"));

document.addEventListener("visibilitychange", function () {
  if (!document.hidden && state.busy && !wakeLock) acquireWakeLock();
});

console.log("[ready] Local Transcriber initialized");
</script>
</body>
</html>
