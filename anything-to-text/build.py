#!/usr/bin/env python3
"""
Build web-utilities/anything-to-text.html — a composite "Anything to Text" SPA
for austegard.com, merged from two source apps:

  - parseimages.aspx  : in-browser OCR (images + PDF) via PaddleOCR / pdf.js
  - transcriber.aspx  : in-browser speech-to-text (audio + video) via Whisper

The two engines are kept VERBATIM (their inline comments document hard-won
edge-case fixes), each in its own <script type="module"> so their top-level
symbols never collide. Surgical transforms only:

  - strip the animated MSD favicon (constants + functions + calls) and brand marks
  - de-collide the 3 transcriber DOM ids that clash with OCR (run/status/copy -> tr-*)
  - replace each engine's own dropzone wiring with a shared router that classifies
    dropped files by type and dispatches intake:ocr / intake:transcribe events

Branding is the site's own "Grouch's Workshop" system (styles/style.css tokens),
not the removed Meso Scale Diagnostics theme.
"""
import re
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
# Canonical, editable, version-controlled copies of the two source apps. Originally
# lifted from the session uploads; vendored here so engine changes are reproducible and
# don't depend on ephemeral upload paths.
SRC = HERE / "src"
OCR_SRC = SRC / "parseimages.aspx"
TR_SRC = SRC / "transcriber.aspx"
OUT = pathlib.Path(
    "/home/user/claude-workspace/.spokes/oaustegard.github.io/web-utilities/anything-to-text.html"
)


def extract_module(text):
    m = re.search(r'<script type="module">(.*?)</script>', text, re.S)
    if not m:
        raise SystemExit("no module script found")
    return m.group(1)


def strip_favicon(js):
    """Remove the animated-favicon block (comment + consts + 3 fns + visibilitychange)."""
    # comment opener through the close of the visibilitychange handler
    pat = re.compile(
        r'/\*[\s\S]{0,160}?Animated favicon[\s\S]*?'
        r'addEventListener\("visibilitychange"[\s\S]*?\}\);\n',
        re.S,
    )
    js, n = pat.subn("", js)
    if n != 1:
        raise SystemExit(f"favicon block: expected 1 removal, got {n}")
    return js


def transform_ocr(js):
    js = strip_favicon(js)
    # leftover favicon calls inside runBatch / reprocessPage
    js = js.replace("\n  ocrRunning = true; startFaviconAnim();", "")
    js = js.replace("\n  ocrRunning = false; stopFaviconAnim();", "")
    # brand-mark wiring + resting favicon at the end of the script
    brand = (
        'const brandMark = $("#brandMark");\n'
        "if (brandMark) { brandMark.src = FAVICON_MSD; brandMark.onerror = () => brandMark.remove(); }\n"
        "setFavicon(FAVICON_MSD);   /* rest on the full mark until a batch runs */\n"
    )
    if brand not in js:
        raise SystemExit("OCR brand-mark block not found")
    js = js.replace(brand, "")
    # replace the dropzone/file/paste wiring with the shared-router intake listener
    wiring = re.compile(
        r'/\* -+ wiring -+ \*/\n'
        r'const drop = \$\("#drop"\), fileInput = \$\("#file"\);\n'
        r'[\s\S]*?'
        r'window\.addEventListener\("paste", e => \{\n'
        r'[\s\S]*?\n\}\);\n'
    )
    new_wiring = (
        "/* ---------- intake: files arrive from the shared router ---------- */\n"
        'const ocrSection = document.getElementById("ocr-section");\n'
        'window.addEventListener("intake:ocr", e => { ocrSection.hidden = false; accept(e.detail); });\n'
    )
    js, n = wiring.subn(new_wiring, js)
    if n != 1:
        raise SystemExit(f"OCR wiring: expected 1 replacement, got {n}")
    # sanity: no MSD references survive
    for bad in ("FAVICON", "Meso Scale", "MSD", "#brandMark"):
        if bad in js:
            raise SystemExit(f"OCR still references {bad!r}")
    return js


def transform_tr(js):
    js = strip_favicon(js)
    # initUI's startup favicon kick
    js = js.replace("\n  if (!document.hidden) startFaviconAnim();", "")
    # notification icon + tag (were MSD favicon / brand)
    js = js.replace('icon: FAVICON_S', 'icon: "/favicon.ico"')
    js = js.replace('tag: "msd-transcriber"', 'tag: "a2t-transcriber"')
    # de-collide the three ids that clash with the OCR engine
    js = js.replace('run: el("run")', 'run: el("tr-run")')
    js = js.replace('status: el("status")', 'status: el("tr-status")')
    js = js.replace('copy: el("copy")', 'copy: el("tr-copy")')
    # main dropzone now fed by the shared router; keep edit-mode zones as-is
    main_zone = "makeDropZone(dom.drop, dom.file, acceptFile);\n"
    if main_zone not in js:
        raise SystemExit("transcriber main makeDropZone not found")
    js = js.replace(
        main_zone,
        '/* main media file arrives from the shared router; edit-mode zones stay local */\n'
        'const trSection = document.getElementById("tr-section");\n'
        'window.addEventListener("intake:transcribe", e => { trSection.hidden = false; acceptFile(e.detail[0]); });\n',
    )
    for bad in ("FAVICON", "Meso Scale", "MSD"):
        if bad in js:
            raise SystemExit(f"transcriber still references {bad!r}")
    return js


OCR_JS = transform_ocr(extract_module(OCR_SRC.read_text()))
TR_JS = transform_tr(extract_module(TR_SRC.read_text()))

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="ai-disclosure" content="ai-assisted">
<meta name="theme-color" content="#5d6d3b">
<meta name="description" content="Turn anything into text in your browser - OCR for images and PDFs, speech-to-text for audio and video. Nothing is uploaded; everything runs on your device.">
<title>Anything to Text - in-browser OCR &amp; transcription</title>
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="stylesheet" href="/styles/style.css">
<style>
/* ---- Anything to Text: component styles on the site's Grouch's Workshop tokens ---- */
:root {
  --a2t-accent:    var(--grouch);
  --a2t-strong:    var(--grouch-deep);
  --a2t-danger:    var(--tongue);
  --a2t-panel:     var(--code-bg);
  --a2t-line:      var(--rule);
  --a2t-surface:   var(--surface);
  --a2t-dim:       var(--muted);
  --a2t-faint:     color-mix(in srgb, var(--muted) 70%, var(--bg));
  --ui:    var(--font-body);
  --disp:  var(--font-display);
  --mono:  var(--font-mono);
}
/* widen the page: the site default is a 70ch reading column, too narrow for the tool UI */
html { max-width: 1000px; padding: 1.5em 1em; font-size: 16px; }
* { box-sizing: border-box; }
[hidden] { display: none !important; }

.wrap { position: relative; }
header { margin-bottom: 24px; }
.eyebrow { font-size: 11px; letter-spacing: 0.28em; text-transform: uppercase; color: var(--a2t-accent); margin: 0 0 8px; font-family: var(--mono); }
h1 { margin: 0 0 10px; }
.sub { color: var(--a2t-dim); margin: 0; max-width: 70ch; }
.sub b { color: var(--fg); font-weight: 600; }

.panel { border: 1px solid var(--a2t-line); border-radius: var(--radius); background: var(--a2t-panel); padding: 18px; margin-bottom: 16px; }
section.engine { margin-bottom: 8px; }

/* shared drop zone */
.dropzone { border: 1.5px dashed var(--a2t-line); border-radius: var(--radius); padding: 34px 20px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; background: var(--a2t-surface); }
.dropzone:hover, .dropzone.over, .dropzone.hover { border-color: var(--a2t-accent); background: color-mix(in srgb, var(--a2t-accent) 7%, var(--a2t-surface)); }
.dropzone .big { font-family: var(--disp); font-size: 18px; font-weight: 700; color: var(--fg); }
.dropzone .hint { color: var(--a2t-faint); font-size: 12px; margin-top: 6px; }
.dropzone.has-file { border-style: solid; border-color: var(--a2t-accent); }
#route-note { color: var(--a2t-dim); font-size: 12px; margin: 12px 2px 0; min-height: 16px; font-family: var(--mono); }

.filemeta { color: var(--a2t-dim); font-size: 12px; margin: 0 0 14px; word-break: break-all; }
.filemeta b { color: var(--fg); }

/* controls */
.controls { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
.controls > div { flex: 0 1 auto; }
label.field { display: block; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--a2t-faint); margin-bottom: 6px; font-family: var(--mono); }
input[type=number], select { font-family: var(--ui); font-size: 13px; color: var(--fg); background: var(--a2t-surface); border: 1px solid var(--a2t-line); border-radius: var(--radius); padding: 9px 10px; }
input[type=number] { width: 96px; }
input[type=number]:focus, select:focus { outline: none; border-color: var(--a2t-accent); }
.radios { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; min-height: 38px; }
.radio { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--fg); cursor: pointer; }
.radio input { accent-color: var(--a2t-accent); cursor: pointer; margin: 0; }

.runbar { display: flex; align-items: center; gap: 12px; margin-top: 18px; flex-wrap: wrap; }
button.primary { font-family: var(--disp); font-weight: 700; font-size: 14px; letter-spacing: 0.02em; color: #fff; background: var(--a2t-accent); border: 0; border-radius: var(--radius); padding: 11px 22px; cursor: pointer; transition: transform .08s, filter .2s; }
button.primary:hover:not(:disabled) { filter: brightness(1.1); }
button.primary:active:not(:disabled) { transform: translateY(1px); }
button.primary:disabled { opacity: 0.4; cursor: not-allowed; }
button.ghost { font-family: var(--ui); font-size: 12px; color: var(--fg); background: transparent; border: 1px solid var(--a2t-line); border-radius: var(--radius); padding: 8px 14px; cursor: pointer; transition: color .15s, border-color .15s; }
button.ghost:hover:not(:disabled) { color: var(--a2t-accent); border-color: var(--a2t-accent); }
button.ghost:disabled { color: var(--a2t-faint); border-color: var(--a2t-line); opacity: 0.6; cursor: not-allowed; }
@media (prefers-color-scheme: dark) { button.primary { color: #15170f; } }

#status, #tr-status { color: var(--a2t-dim); font-size: 12px; min-height: 18px; white-space: pre-wrap; flex: 1 1 200px; }
#status .err, #tr-status .err { color: var(--a2t-danger); }
#eta { color: var(--a2t-faint); font-size: 12px; margin: 10px 0 0; min-height: 16px; }

/* exports */
.result-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.result-actions .label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--a2t-faint); margin-right: 4px; font-family: var(--mono); }

/* OCR per-page cards */
.page { border: 1px solid var(--a2t-line); border-radius: var(--radius); background: var(--a2t-surface); padding: 12px 14px; margin: 12px 0; }
.page > h2 { font-size: 13px; font-weight: 600; margin: 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-family: var(--ui); color: var(--fg); }
.page .pg-name { color: var(--fg); word-break: break-all; }
.page .meta { font-size: 11px; color: var(--a2t-faint); font-variant-numeric: tabular-nums; }
.page .copy-page, .page .reocr { padding: 5px 11px; font-size: 11px; }
.page .hdr-actions { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
.src { font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--a2t-line); color: var(--a2t-dim); }
.src.ocr { color: var(--a2t-accent); border-color: color-mix(in srgb, var(--a2t-accent) 45%, transparent); }
.page .cols { display: flex; gap: 14px; margin-top: 8px; align-items: flex-start; }
.page .preview { width: 300px; flex: 0 0 300px; height: auto; border: 1px solid var(--a2t-line); border-radius: var(--radius); background: #fff; cursor: zoom-in; }
.page .preview:target, .page .preview.zoomed { width: 100%; flex-basis: 100%; max-width: 100%; cursor: zoom-out; }
.page .col-text { flex: 1 1 auto; min-width: 0; }
.page .col-text .out { margin-top: 0; }
.out.muted { color: var(--a2t-faint); }
@media (max-width: 680px) { .page .cols { flex-direction: column; } .page .preview { width: 100%; flex-basis: auto; max-width: 360px; } }
.badge { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--a2t-line); color: var(--a2t-dim); font-family: var(--mono); }
.badge.pending { color: var(--a2t-faint); }
.badge.done { color: #fff; background: var(--a2t-accent); border-color: var(--a2t-accent); }
.badge.error { color: var(--a2t-danger); border-color: var(--a2t-danger); }
.out { white-space: pre-wrap; font-family: var(--mono); font-size: 13px; line-height: 1.5; background: var(--a2t-panel); border-radius: var(--radius); padding: 10px; margin-top: 8px; max-height: 280px; overflow: auto; color: var(--fg); }
.out.err { color: var(--a2t-danger); background: color-mix(in srgb, var(--a2t-danger) 8%, var(--a2t-surface)); }

/* OCR overlay */
#overlay { position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,0.55); display: flex; flex-direction: column; }
#overlay[hidden] { display: none; }
.ov-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 12px; background: var(--a2t-panel); border-bottom: 1px solid var(--a2t-line); font: 13px var(--ui); color: var(--fg); }
#ov-title { font-weight: bold; }
#ov-body { flex: 1; min-height: 0; overflow: auto; background: var(--bg); }
#ov-body pre { margin: 0; padding: 12px; font: 12px/1.5 var(--mono); white-space: pre-wrap; word-break: break-word; color: var(--fg); }
#ov-body iframe { display: block; width: 100%; height: 100%; border: 0; background: #fff; }

/* transcription pipeline strip */
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 18px 0; }
.stage { border: 1px solid var(--a2t-line); border-radius: var(--radius); padding: 12px 14px; background: var(--a2t-panel); transition: border-color .25s, color .25s, box-shadow .25s; }
.stage .n { color: var(--a2t-faint); font-size: 11px; font-family: var(--mono); }
.stage .t { color: var(--a2t-dim); font-size: 12px; margin-top: 4px; }
.stage.active { border-color: var(--a2t-accent); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--a2t-accent) 45%, transparent); background: var(--a2t-surface); }
.stage.active .t, .stage.active .n { color: var(--a2t-accent); }
.stage.done { border-color: var(--a2t-accent); background: var(--a2t-accent); }
.stage.done .t, .stage.done .n { color: #fff; }
@media (prefers-color-scheme: dark) { .stage.done .t, .stage.done .n { color: #15170f; } }

/* transcription progress bar */
.bar { height: 6px; background: color-mix(in srgb, var(--fg) 8%, transparent); border-radius: 999px; overflow: hidden; margin-top: 10px; display: none; }
.bar.show { display: block; }
.bar > i { display: block; height: 100%; width: 0%; background: var(--a2t-accent); transition: width .2s; }
.bar.indeterminate > i { width: 40%; transition: none; animation: bar-slide 1.1s ease-in-out infinite; }
@keyframes bar-slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(280%); } }

/* device badge */
.badge.gpu { color: #fff; background: var(--a2t-accent); border-color: var(--a2t-accent); }
.badge.cpu { color: var(--a2t-accent); border-color: var(--a2t-accent); }
@media (prefers-color-scheme: dark) { .badge.gpu { color: #15170f; } }

/* transcription results */
#results { display: none; }
#results.show { display: block; }
.result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.result-head h2 { font-family: var(--disp); font-size: 20px; margin: 0; color: var(--fg); }
#partial { color: var(--a2t-faint); font-style: italic; white-space: pre-wrap; min-height: 0; }
.plaintext { white-space: pre-wrap; line-height: 1.7; color: var(--fg); }
.segments { margin-top: 18px; border-top: 1px solid var(--a2t-line); padding-top: 14px; }
.segments summary { cursor: pointer; color: var(--a2t-dim); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; font-family: var(--mono); }
.seg-row { display: grid; grid-template-columns: auto 1fr; gap: 12px; padding: 4px 0; border-bottom: 1px solid var(--a2t-line); align-items: start; }
.seg-row .ts { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; font-family: var(--mono); font-size: 12px; color: var(--a2t-accent); background: transparent; border: 1px solid var(--a2t-line); border-radius: var(--radius); padding: 3px 9px; cursor: pointer; text-align: left; transition: background .15s, border-color .15s, color .15s; }
.seg-row .ts:hover:not(:disabled) { border-color: var(--a2t-accent); background: color-mix(in srgb, var(--a2t-accent) 8%, transparent); }
.seg-row .ts:disabled { color: var(--a2t-faint); border-color: transparent; cursor: default; }
.seg-row .ts.playing { background: var(--a2t-accent); color: #fff; border-color: var(--a2t-accent); }
.seg-row .ts .glyph { font-size: 9px; line-height: 1; }
.seg-row .tx { color: var(--fg); padding-top: 4px; }
.seg-row .tx.editing { background: var(--a2t-surface); border: 1px solid var(--a2t-line); border-radius: var(--radius); padding: 4px 8px; }
.seg-row .tx.editing:focus { outline: none; border-color: var(--a2t-accent); box-shadow: 0 0 0 1px var(--a2t-accent); }

/* transcription tabs (transcribe / edit existing) */
.tabs { display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 2px solid var(--a2t-line); }
.tab { appearance: none; background: none; border: none; cursor: pointer; font-family: var(--disp); font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; font-size: 13px; color: var(--a2t-dim); padding: 10px 16px; margin-bottom: -2px; border-bottom: 2px solid transparent; }
.tab:hover { color: var(--fg); }
.tab[aria-selected="true"] { color: var(--a2t-accent); border-bottom-color: var(--a2t-accent); }
.edit-help { color: var(--a2t-dim); font-size: 12px; margin: 0 0 14px; }
.edit-inputs { display: flex; flex-direction: column; gap: 12px; }
.edit-inputs .dropzone { padding: 24px 20px; }
.edit-inputs .dropzone .big { font-size: 15px; }
.meta-line { color: var(--a2t-faint); font-size: 11px; margin-top: 14px; font-family: var(--mono); }
.tech-note { color: var(--a2t-faint); font-size: 11px; line-height: 1.7; margin-top: 18px; }

@media (max-width: 680px) { .pipeline { grid-template-columns: repeat(2, 1fr); } .controls { flex-direction: column; gap: 12px; } }
</style>
</head>
<body>
<div class="wrap">
  <a href="/web-utilities/" class="back-link">Web &amp; File Utilities</a>
  <header>
    <p class="eyebrow">In your browser &middot; nothing uploaded</p>
    <h1>Anything to Text</h1>
    <p class="sub">Drop an <b>image</b>, a <b>PDF</b>, or an <b>audio/video</b> recording and get text back. Images and PDFs are read with on-device OCR; audio and video are transcribed with a local speech model. <b>Your files never leave your device</b> &mdash; only the models download on first run.</p>
  </header>

  <!-- shared intake: one drop zone for everything, routed by file type -->
  <div class="panel">
    <div id="drop" class="dropzone">
      <div class="big">Drop anything &mdash; images, a PDF, audio, or video</div>
      <div class="hint">paste an image, or click to choose files</div>
    </div>
    <input id="file" type="file" accept="image/*,application/pdf,audio/*,video/*" multiple hidden>
    <p id="route-note"></p>
  </div>

  <!-- ===================== OCR engine (images + PDF) ===================== -->
  <section id="ocr-section" class="engine" hidden>
    <div class="panel">
      <div class="controls">
        <div>
          <label class="field" for="tier">OCR model tier</label>
          <select id="tier">
            <option value="tiny">Tiny (~3M, fastest)</option>
            <option value="small">Small (~7M, balanced)</option>
            <option value="medium" selected>Medium (~35M, best)</option>
          </select>
        </div>
        <div><label class="field" for="dpi">PDF DPI</label><input id="dpi" type="number" value="200" step="50" min="72" max="400"></div>
        <div><label class="field" for="batch">Batch size</label><input id="batch" type="number" value="10" step="1" min="1" max="50"></div>
        <div>
          <label class="field" for="columns">Columns</label>
          <select id="columns">
            <option value="auto" selected>Auto-detect</option>
            <option value="1">1 (single)</option>
            <option value="2">2</option>
            <option value="3">3</option>
          </select>
        </div>
        <div>
          <label class="field" for="textorder">PDF text order</label>
          <select id="textorder">
            <option value="geometric" selected>Geometric (column-aware)</option>
            <option value="pdf">Preserve PDF order</option>
          </select>
        </div>
      </div>
      <div class="runbar">
        <button id="run" class="primary" disabled>Run batch</button>
        <button id="continue" class="ghost" disabled>Continue</button>
        <button id="estimate" class="ghost" disabled>Estimate</button>
        <button id="clear" class="ghost" disabled>Clear</button>
        <span id="status">Loading OCR libraries&hellip;</span>
      </div>
      <div id="eta"></div>
    </div>

    <div class="panel">
      <div class="result-actions">
        <span class="label">Export</span>
        <button id="copy" class="ghost" disabled>Copy all text</button>
        <button id="txt" class="ghost" disabled>Download .txt</button>
        <button id="json" class="ghost" disabled>Download .json</button>
        <button id="html" class="ghost" disabled>Download .html</button>
        <button id="show-json" class="ghost" disabled>Show .json</button>
        <button id="show-html" class="ghost" disabled>Show .html</button>
      </div>
    </div>

    <div id="pages"></div>

    <div id="overlay" hidden>
      <div class="ov-bar">
        <span id="ov-title"></span>
        <button id="ov-close" class="ghost">Close</button>
      </div>
      <div id="ov-body"></div>
    </div>
  </section>

  <!-- ===================== Transcription engine (audio + video) ===================== -->
  <section id="tr-section" class="engine" hidden>
    <div class="tabs" role="tablist" aria-label="Mode">
      <button class="tab" id="tab-transcribe" type="button" role="tab" aria-selected="true" aria-controls="panel-transcribe">Transcribe</button>
      <button class="tab" id="tab-edit" type="button" role="tab" aria-selected="false" aria-controls="panel-edit">Edit existing</button>
    </div>

    <div class="panel" id="panel-transcribe" role="tabpanel" aria-labelledby="tab-transcribe">
      <div class="filemeta" id="filemeta" hidden></div>
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
            <label class="radio"><input type="radio" name="task" value="transcribe" checked> Transcribe</label>
            <label class="radio"><input type="radio" name="task" value="translate"> Translate &rarr; EN</label>
          </div>
        </div>
      </div>
      <div class="runbar">
        <button class="primary" id="tr-run" disabled>Transcribe</button>
        <span class="badge" id="device">detecting&hellip;</span>
        <span id="tr-status"></span>
      </div>
      <div class="bar" id="bar"><i></i></div>
    </div>

    <div class="panel" id="panel-edit" role="tabpanel" aria-labelledby="tab-edit" hidden>
      <p class="edit-help">Load the original audio/video and its transcript. An .srt keeps timestamps and per-segment playback; a .txt loads as plain lines. Edit the text, play any segment to check it against the audio, then download the corrected file.</p>
      <div class="edit-inputs">
        <div class="dropzone" id="edit-audio-zone">
          <div class="big">Original audio / video</div>
          <div class="hint">Drop or click to choose &middot; the recording you transcribed</div>
          <div class="filemeta" id="edit-audio-meta" hidden></div>
        </div>
        <input type="file" id="edit-audio" accept="audio/*,video/*" hidden>
        <div class="dropzone" id="edit-transcript-zone">
          <div class="big">Transcript</div>
          <div class="hint">Drop or click to choose &middot; .srt or .txt</div>
          <div class="filemeta" id="edit-transcript-meta" hidden></div>
        </div>
        <input type="file" id="edit-transcript" accept=".srt,.txt,text/plain" hidden>
      </div>
      <div class="runbar">
        <button class="primary" id="edit-load" disabled>Load for editing</button>
        <span id="edit-status"></span>
      </div>
      <div class="bar" id="edit-bar"><i></i></div>
    </div>

    <div class="pipeline" id="pipeline">
      <div class="stage" data-stage="decode"><div class="n">01</div><div class="t">Extract audio</div></div>
      <div class="stage" data-stage="model"><div class="n">02</div><div class="t">Load model</div></div>
      <div class="stage" data-stage="transcribe"><div class="n">03</div><div class="t">Transcribe</div></div>
      <div class="stage" data-stage="present"><div class="n">04</div><div class="t">Read result</div></div>
    </div>

    <div class="panel" id="results">
      <div class="result-head">
        <h2>Transcript</h2>
        <div class="result-actions">
          <button class="ghost" id="tr-copy">Copy</button>
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

    <div class="tech-note" id="tech-note"></div>
  </section>

  <footer>
    <p class="provenance">In-browser OCR (PaddleOCR PP-OCRv6 + pdf.js) and speech-to-text (Whisper via Transformers.js). All processing is client-side; nothing is uploaded. AI-assisted build.</p>
  </footer>
</div>

<!-- ===================== shared router ===================== -->
<script type="module">
  /* One drop zone, two engines. Classify each dropped file by type and fan out
     to the OCR engine (images / PDF) or the transcription engine (audio / video)
     via intake events; the engines reveal their own sections on receipt. */
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("file");
  const note = document.getElementById("route-note");
  const isOcr = f => f.type.startsWith("image/") || f.type === "application/pdf" || /\\.pdf$/i.test(f.name || "");
  const isAv  = f => f.type.startsWith("audio/") || f.type.startsWith("video/") || /\\.(mp4|mov|mkv|webm|mp3|wav|m4a|flac|ogg|opus|aac)$/i.test(f.name || "");
  function route(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const ocr = files.filter(isOcr);
    const av  = files.filter(isAv);
    if (ocr.length) window.dispatchEvent(new CustomEvent("intake:ocr", { detail: ocr }));
    if (av.length)  window.dispatchEvent(new CustomEvent("intake:transcribe", { detail: av }));
    const parts = [];
    if (ocr.length) parts.push(ocr.length + " image/PDF \\u2192 OCR");
    if (av.length)  parts.push((av.length > 1 ? "1 of " + av.length + " " : "") + "audio/video \\u2192 transcript");
    const skipped = files.length - ocr.length - av.length;
    if (skipped > 0) parts.push(skipped + " unsupported skipped");
    note.textContent = parts.join("  \\u00b7  ");
  }
  drop.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { if (fileInput.files.length) route(fileInput.files); fileInput.value = ""; });
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  drop.addEventListener("dragleave", e => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files.length) route(e.dataTransfer.files); });
  window.addEventListener("paste", e => {
    const it = [...((e.clipboardData && e.clipboardData.items) || [])].find(i => i.type.startsWith("image/"));
    if (it) { const f = it.getAsFile(); if (f) route([f]); }
  });
  console.log("[router] ready");
</script>

<!-- ===================== OCR engine (verbatim from the source app; original branding removed) ===================== -->
<script type="module">
__OCR_JS__
</script>

<!-- ===================== Transcription engine (verbatim from the source app; original branding removed) ===================== -->
<script type="module">
__TR_JS__
</script>
</body>
</html>
"""

doc = HEAD.replace("__OCR_JS__", OCR_JS).replace("__TR_JS__", TR_JS)
OUT.write_text(doc)
print(f"wrote {OUT} ({len(doc)} bytes, OCR {len(OCR_JS)}B, TR {len(TR_JS)}B)")
