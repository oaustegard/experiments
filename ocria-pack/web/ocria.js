// OCRia-nano-1 (LlamaIndex) in a browser tab, from packed weights, on LiteParse's own WebAssembly build.
//
// LiteParse (Apache-2.0) runs the detector, script router, recognizers and LayoutNet in WebAssembly on the
// CPU: no WebGPU needed. Its runtime comes unchanged from jsdelivr (5.7 MB brotli). The model files come
// from this folder's packs: every large float tensor stored as int8 per row, turned back here into the
// exact files LiteParse loads (see run/pack.py), about four times fewer bytes than the upstream files.
//
//   import { loadOcria } from "https://austegard.com/experiments/ocria-pack/web/ocria.js";
//   const ocr = await loadOcria({ layout: true, scripts: ["latin"] });
//   const page = await ocr.read(canvas);   // { markdown, text, items: [{text, confidence, box}], blocks, ms }
//
// The weights are OCRia-nano-1 by LlamaIndex under the OCRia-nano-1 Model License (modified OpenRAIL-M):
// see models/LICENSE and NOTICE.md beside this file. Their use restrictions apply to anyone using them.

export const LITEPARSE_VERSION = "2.21.2";
const CDN = `https://cdn.jsdelivr.net/npm/@llamaindex/liteparse-wasm@${LITEPARSE_VERSION}/pkg/`;
const RENDER_DPI = 250;   // the PDF page is sized so LiteParse renders it at this DPI pixel for pixel; 250 is the DPI the benchmark ran at, and LiteParse's layout rules measure text in points

let lpModule = null;
async function liteparse(cdn) {
  if (!lpModule) {
    lpModule = (async () => {
      const m = await import(cdn + "liteparse_wasm.js");
      await m.default({ module_or_path: cdn + "liteparse_wasm_bg.wasm" });
      return m;
    })();
    lpModule.catch(() => { lpModule = null; });
  }
  return lpModule;
}

/** Turn an OCPK bundle back into its files: { name: Uint8Array }. Mirrors run/pack.py unpack(). */
export function unpack(buf) {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== "OCPK") throw new Error("not an OCPK bundle");
  const n = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const blob0 = 8 + n;
  const out = {};
  for (const f of head.files) {
    const file = new Uint8Array(f.size);
    for (const [off, len, bo] of f.kept) file.set(u8.subarray(blob0 + bo, blob0 + bo + len), off);
    const fv = new DataView(file.buffer);
    for (const [off, rows, cols, qo, so] of f.q) {
      const q = new Int8Array(u8.buffer, u8.byteOffset + blob0 + qo, rows * cols);
      for (let r = 0; r < rows; r++) {
        const s = dv.getFloat32(blob0 + so + 4 * r, true);
        const base = off + 4 * r * cols, qb = r * cols;
        for (let c = 0; c < cols; c++) fv.setFloat32(base + 4 * c, Math.fround(q[qb + c] * s), true);
      }
    }
    out[f.name] = file;
  }
  return out;
}

async function fetchBytes(url, onBytes, fetcher = fetch) {
  const res = await fetcher(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const raw = await res.arrayBuffer();
  if (onBytes) onBytes(raw.byteLength);
  if (!url.endsWith(".gz")) return new Uint8Array(raw);
  // gzipped by us, not by the server (GitHub Pages sends .pack files as they are)
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Bytes of a one-page PDF holding `gray` (w x h, 8-bit) losslessly, sized for a render at RENDER_DPI. */
export async function grayPdf(gray, w, h, deflate) {
  const data = deflate ? await deflate(gray) : await zlib(gray);
  const pw = (w * 72 / RENDER_DPI).toFixed(4), ph = (h * 72 / RENDER_DPI).toFixed(4);
  const content = new TextEncoder().encode(`q ${pw} 0 0 ${ph} 0 0 cm /Im0 Do Q`);
  const enc = (s) => new TextEncoder().encode(s);
  const objs = [
    [enc("<< /Type /Catalog /Pages 2 0 R >>")],
    [enc("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")],
    [enc(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pw} ${ph}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>`)],
    [enc(`<< /Type /XObject /Subtype /Image /Width ${w} /Height ${h} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length ${data.length} >>\nstream\n`), data, enc("\nendstream")],
    [enc(`<< /Length ${content.length} >>\nstream\n`), content, enc("\nendstream")],
  ];
  const parts = [enc("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")];
  let len = parts[0].length;
  const offs = [];
  objs.forEach((o, i) => {
    offs.push(len);
    for (const p of [enc(`${i + 1} 0 obj\n`), ...o, enc("\nendobj\n")]) { parts.push(p); len += p.length; }
  });
  const xref = len;
  let tail = `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  for (const o of offs) tail += `${String(o).padStart(10, "0")} 00000 n \n`;
  tail += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  parts.push(enc(tail));
  const out = new Uint8Array(len + parts[parts.length - 1].length);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function zlib(bytes) {
  const cs = new CompressionStream("deflate");
  const stream = new Blob([bytes]).stream().pipeThrough(cs);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Luminance of a canvas / ImageData / {data,width,height} RGBA image, as LiteParse's gray conversion would see it. */
function toGray(src) {
  let img = src;
  if (typeof HTMLCanvasElement !== "undefined" && src instanceof HTMLCanvasElement ||
      typeof OffscreenCanvas !== "undefined" && src instanceof OffscreenCanvas) {
    img = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, src.width, src.height);
  }
  const { data, width, height } = img;
  const g = new Uint8Array(width * height);
  for (let i = 0, j = 0; j < g.length; i += 4, j++) {
    const a = data[i + 3] / 255;   // transparent pixels read as white paper
    g[j] = Math.round((0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) * a + 255 * (1 - a));
  }
  return { gray: g, width, height };
}

/**
 * Load OCRia-nano-1. Options:
 *   base     where the packs are (default: ./models/ beside this file)
 *   layout   load LayoutNet (tables, headings, lists; about 4.0 MB more). Default true.
 *   scripts  extra recognizers beyond Latin: "world", "rtl", "cjk". Lines the router sends to a
 *            recognizer that is not loaded are read by the Latin one. Default [].
 *   cdn      where LiteParse's wasm build is (default: jsdelivr, pinned)
 *   log      function(message) for progress
 *   lp       an initialized LiteParse wasm module to use instead of loading one (tests)
 *   fetch    a fetch to load the packs with (tests)
 */
export async function loadOcria(opts = {}) {
  const base = opts.base || new URL("./models/", import.meta.url).href;
  const layout = opts.layout !== false;
  const scripts = (opts.scripts || []).filter((s) => ["world", "rtl", "cjk"].includes(s));
  const log = opts.log || (() => {});
  const t0 = performance.now();
  let bytes = 0;
  const count = (n) => { bytes += n; };
  const packs = ["ocria-core", ...(layout ? ["ocria-layout"] : []), ...scripts.map((s) => `ocria-${s}`)].map((p) => p + ".pack.gz");
  log(`fetching ${packs.join(", ")} and LiteParse ${LITEPARSE_VERSION}`);
  const [lp, ...bufs] = await Promise.all([opts.lp || liteparse(opts.cdn || CDN),
    ...packs.map((p) => fetchBytes(base + p, count, opts.fetch))]);
  const files = Object.assign({}, ...bufs.map(unpack));
  const rec = (s) => files[`rec_${s}.onnx`] && {
    onnx: files[`rec_${s}.onnx`], charset: files[`rec_${s}.charset.json`],
    ...(files[`rec_${s}.meta.json`] ? { meta: files[`rec_${s}.meta.json`] } : {}),
  };
  const models = { det: files["det.onnx"], detMeta: files["det.meta.json"], router: files["router.onnx"], recLatin: rec("latin") };
  for (const s of scripts) models[{ world: "recWorld", rtl: "recRtl", cjk: "recCjk" }[s]] = rec(s);
  if (layout) models.tablenet = files["tablenet.bin"];
  const ocria = lp.OcriaNano.load(models);
  const parser = lp.LiteParse.withOcriaNano({
    ocrEnabled: true, outputFormat: "markdown", dpi: RENDER_DPI, keepHeadersFooters: true,
    imageMode: "off", extractBlocks: true, quiet: true,
  }, ocria);
  const loadMs = performance.now() - t0;
  log(`ready in ${Math.round(loadMs)} ms (${(bytes / 1e6).toFixed(2)} MB of packs)`);
  const k = RENDER_DPI / 72;   // LiteParse reports boxes in PDF points; the canvas is points * k

  return {
    name: "OCRia-nano-1",
    layout, scripts, loadMs, packBytes: bytes,
    /** Read one page image (canvas, OffscreenCanvas or ImageData). */
    async read(src) {
      const t = performance.now();
      const { gray, width, height } = toGray(src);
      const pdf = await grayPdf(gray, width, height);
      const res = await parser.parse(pdf);
      const pg = res.pages[0] || { markdown: "", text: "", textItems: [] };
      const items = (pg.textItems || []).filter((x) => (x.text || "").trim()).map((x) => ({
        text: x.text, confidence: x.confidence ?? 1,
        box: { x: x.x * k, y: x.y * k, width: x.width * k, height: x.height * k },
      }));
      const blocks = (pg.blocks || []).map((b) => ({
        kind: b.kind, text: b.text, level: b.level,
        box: b.bbox && { x: b.bbox.x * k, y: b.bbox.y * k, width: b.bbox.width * k, height: b.bbox.height * k },
      }));
      return { markdown: pg.markdown || "", text: pg.text || "", items, blocks, width, height, ms: performance.now() - t };
    },
  };
}
