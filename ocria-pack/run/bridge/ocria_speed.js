// GPU bridge job: OCRia-nano-1 on three Omni Parse Bench pages, two ways, on the device running the tab.
//   gpu:  web/gpu/ocria-gpu.js (WGSL nets + the JS port of LiteParse's page logic), lines only
//   wasm: web/ocria.js (LiteParse 2.21.2 WebAssembly on one CPU thread), layout off, so both do the same work
// Each path reads page 1 once to warm up, then every page once; times are per page, in the worker.
const BASE = "https://austegard.com/experiments/ocria-pack/web/";
const HF = "https://huggingface.co/datasets/datalab-to/omni_parse_bench/resolve/main/pages/";
const ids = ctx.inputs.cfg.ids;
const pages = [];
for (const [i, id] of ids.entries()) {
  ctx.progress(i / (ids.length * 3), "fetch " + id.slice(0, 40));
  const bmp = await createImageBitmap(new Blob([await ctx.fetchBytes(HF + id + ".png")]));
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  cv.getContext("2d").drawImage(bmp, 0, 0);
  pages.push({ id, cv });
}
const out = { device: null, gpu: {}, wasm: {} };
const device = await ctx.gpu();
out.device = ctx.describe ? ctx.describe().adapter || null : null;
let t = performance.now();
const { loadOcriaGpu } = await ctx.importModule(BASE + "gpu/ocria-gpu.js");
const g = await loadOcriaGpu({ device, scripts: ["world", "rtl", "cjk"] });
out.gpu.load_ms = Math.round(performance.now() - t);
await g.read(pages[0].cv);
out.gpu.pages = [];
for (const [i, p] of pages.entries()) {
  ctx.progress((ids.length + i) / (ids.length * 3), "gpu " + i);
  t = performance.now();
  const lines = await g.read(p.cv);
  out.gpu.pages.push({ id: p.id, w: p.cv.width, h: p.cv.height, ms: Math.round(performance.now() - t), lines: lines.length });
}
out.gpu.stats = g.engine.stats || null;
t = performance.now();
const { loadOcria } = await ctx.importModule(BASE + "ocria.js");
const w = await loadOcria({ layout: false, scripts: ["world", "rtl", "cjk"] });
out.wasm.load_ms = Math.round(performance.now() - t);
await w.read(pages[0].cv);
out.wasm.pages = [];
for (const [i, p] of pages.entries()) {
  ctx.progress((2 * ids.length + i) / (ids.length * 3), "wasm " + i);
  t = performance.now();
  const r = await w.read(p.cv);
  out.wasm.pages.push({ id: p.id, ms: Math.round(performance.now() - t), items: r.items.length });
}
ctx.log(JSON.stringify(out));
return out;
