// GPU Bridge job: Indra in the browser on a real GPU. Inputs: pages (u8, concatenated grayscale
// pages), dims (json: [[w, h], ...]), cfg (json: {base, pack, reps, lm}). Returns per-page
// timings, the markdown (for parity with the container's SwiftShader run) and the adapter.
const cfg = ctx.inputs.cfg;
const base = cfg.base; // e.g. https://cdn.jsdelivr.net/gh/oaustegard/experiments@SHA/indra-browser/web/
const { Indra } = await ctx.importModule(base + "indra.js");
const { Gpu } = await ctx.importModule(base + "gpu.js");
const device = await ctx.gpu();
const key = `indra:${base}:${cfg.pack}`;
let ocr = ctx.cache.get(key);
const t0 = performance.now();
let loadMs = 0;
if (!ocr) {
  ocr = await Indra.load(base + cfg.pack, { gpu: new Gpu(device, "bridge"), fetchBytes: ctx.fetchBytes, log: (m) => ctx.log(m) });
  ctx.cache.set(key, ocr);
  loadMs = performance.now() - t0;
}
ocr.noLm = cfg.lm === false;
ocr.furniture = true;
const pages = ctx.inputs.pages, dims = ctx.inputs.dims;
const out = [];
let off = 0;
for (let i = 0; i < dims.length; i++) {
  const [w, h] = dims[i];
  const gray = { w, h, data: pages.subarray(off, off + w * h) };
  off += w * h;
  const runs = [];
  let page;
  for (let r = 0; r < (cfg.reps ?? 2); r++) {
    const s = performance.now();
    page = await ocr.read(gray);
    runs.push({ ms: performance.now() - s, ...page.timings });
    ctx.progress((i + (r + 1) / (cfg.reps ?? 2)) / dims.length, `page ${i + 1}/${dims.length}`);
  }
  out.push({ w, h, lines: page.lines.length, runs, markdown: page.markdown });
  ctx.log(`page ${i + 1}: ${page.lines.length} lines, ${runs.map((r) => r.ms.toFixed(0)).join(" / ")} ms`);
}
return { loadMs, adapter: ctx.describe ? ctx.describe().adapter ?? null : null, pages: out };
