// Tesseract's own page layout, in the browser.
//
// tessLayout(gray | rgba, width, height, { psm }) runs Tesseract's layout analysis (TessBaseAPI::AnalyseLayout:
// page segmentation, column and block finding, line finding; no character classifier and no recognition) from the
// tesseract.js-core Emscripten build, and returns the text-line boxes in Tesseract's own block and reading order.
//
// Loaded at run time from jsDelivr, pinned:
//   tesseract.js-core@5.1.1, tesseract-core-simd-lstm.js (+ its .wasm; plain tesseract-core-lstm.js when no SIMD)
//   tessdata_fast eng.traineddata at a fixed commit of tesseract-ocr/tessdata_fast
// Both the module and the model are fetched once per page and cached. The model is written to the module's
// virtual file system at /tessdata/eng.traineddata, and Init(datapath, "eng", OEM_LSTM_ONLY) reads it from there.
//
// Bindings used (all present in the 5.1.1 build; checked against its embind glue):
//   TessBaseAPI: Init_3, SetPageSegMode_1, SetImage_5, AnalyseLayout_0, Clear_0, Version_0
//   PageIterator: Begin_0, Empty_1, IsAtBeginningOf_1, BoundingBox_5, Next_1, delete
//   pointerHelper: int out-parameters for BoundingBox
// AnalyseLayout is the closest exact path: it is layout-only by definition, so no recognition step is skipped.
//
// Returns { lines: [{ box: [x0, y0, x1, y1], block, order }], ms }.
//   box: pixel coordinates in the input image, x1/y1 exclusive as Tesseract reports them.
//   block: 1-based block number, incremented at each block start in Tesseract's order.
//   order: 0-based index of the line in Tesseract's reading order across the page.
// Errors throw. An empty page returns lines: [] (a legitimate answer, not a failure).

export const CORE_VERSION = "5.1.1";
const CORE_BASE = `https://cdn.jsdelivr.net/npm/tesseract.js-core@${CORE_VERSION}/`;
const TESSDATA_COMMIT = "87416418657359cb625c412a48b6e1d6d41c29bd"; // tesseract-ocr/tessdata_fast main, 2026-11-20
const MODEL_URL = `https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_fast@${TESSDATA_COMMIT}/eng.traineddata`;
const DATAPATH = "/tessdata/";
const OEM_LSTM_ONLY = 1;
const RIL_BLOCK = 0;
const RIL_TEXTLINE = 2;

// A 20-byte module that only validates when the engine supports WebAssembly SIMD (wasm-feature-detect's probe).
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0,
  65, 0, 253, 15, 253, 98, 11]);

let runtimePromise = null; // { M, api, info } once the module and model are ready
const stats = { coreFile: null, coreBytes: null, modelBytes: null, loadMs: null, tesseractVersion: null };

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`tesslayout: script failed to load: ${src}`));
    document.head.appendChild(s);
  });
}

async function fetchBytes(url) {
  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error(`tesslayout: ${res.status} fetching ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function instantiate() {
  const simd = WebAssembly.validate(SIMD_PROBE);
  const file = simd ? "tesseract-core-simd-lstm.js" : "tesseract-core-lstm.js";
  stats.coreFile = file;
  await loadScript(CORE_BASE + file);
  const factory = globalThis.TesseractCore;
  if (typeof factory !== "function") throw new Error("tesslayout: TesseractCore factory missing after load");
  // The factory resolves to the initialised Emscripten module; locateFile points the .wasm at jsDelivr.
  return factory({
    locateFile: (p) => CORE_BASE + p,
    print: () => {},
    printErr: () => {},
  });
}

async function boot() {
  const t0 = performance.now();
  const M = await instantiate();
  const model = await fetchBytes(MODEL_URL);
  stats.modelBytes = model.byteLength;
  try { M.FS.mkdir("/tessdata"); } catch (_) { /* already exists after an earlier boot */ }
  M.FS.writeFile(DATAPATH + "eng.traineddata", model);
  const api = new M.TessBaseAPI();
  const rc = api.Init(DATAPATH, "eng", OEM_LSTM_ONLY);
  if (rc !== 0) throw new Error(`tesslayout: TessBaseAPI::Init returned ${rc}`);
  stats.tesseractVersion = api.Version();
  stats.loadMs = performance.now() - t0;
  return { M, api };
}

function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = boot().catch((e) => { runtimePromise = null; throw e; });
  }
  return runtimePromise;
}

// Grey buffer for Tesseract. RGBA is reduced with BT.601 luma, the same weights as the Rec. 601 grey used elsewhere in
// this repository, so the layout sees one channel regardless of the caller's pixel format.
function toGrey(src, width, height) {
  const n = width * height;
  if (src.length === n) return src;
  if (src.length !== 4 * n) throw new Error(`tesslayout: buffer of ${src.length} bytes is not ${width}x${height} grey or RGBA`);
  const g = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    g[i] = (src[j] * 299 + src[j + 1] * 587 + src[j + 2] * 114 + 500) / 1000 | 0;
  }
  return g;
}

export async function tessLayout(pixels, width, height, { psm = 3 } = {}) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error("tesslayout: width and height must be positive integers");
  }
  const { M, api } = await getRuntime();
  const grey = toGrey(pixels, width, height);

  const t0 = performance.now();
  const ptr = M._malloc(grey.length);
  let it = null;
  try {
    M.HEAPU8.set(grey, ptr);
    api.SetPageSegMode(psm);
    api.SetImage(ptr, width, height, 1, width); // Tesseract copies the pixels, so ptr can be freed below
    it = api.AnalyseLayout();
    const lines = [];
    if (it) {
      it.Begin();
      if (!it.Empty(RIL_TEXTLINE)) {
        let block = 0;
        do {
          if (it.IsAtBeginningOf(RIL_BLOCK)) block++;
          const h = new M.pointerHelper();
          const ok = it.BoundingBox(RIL_TEXTLINE, h.i32("x0"), h.i32("y0"), h.i32("x1"), h.i32("y1"));
          const r = h.get();
          if (ok) lines.push({ box: [r.x0, r.y0, r.x1, r.y1], block, order: lines.length });
        } while (it.Next(RIL_TEXTLINE));
      }
    }
    api.Clear(); // drops this page's image and layout; the model stays loaded for the next call
    return { lines, ms: performance.now() - t0 };
  } finally {
    if (it && typeof it.delete === "function") it.delete();
    M._free(ptr);
  }
}

// What the last load fetched and how long it took. Byte counts are the decoded bodies fetched by this page.
export function tessLayoutInfo() {
  return { ...stats, tesseractCoreVersion: CORE_VERSION, tessdataCommit: TESSDATA_COMMIT };
}
