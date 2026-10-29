// Headless checks on SwiftShader WebGPU (correctness, not speed):
//   node test/verify.mjs filter          GPU pyramid, band-pass, ROI means and composite against evm.js's float64 Reference
//   node test/verify.mjs rate [cases]    the whole page on synthetic video of known pulse rate (?synth=), every case in
//                                        CASES or the named ones; writes results/rate.json
//   node test/verify.mjs live            the page with Chromium's fake camera: frames flow, no WebGPU errors
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), WEB = path.join(HERE, "..", "web"), OUT = path.join(HERE, "..", "results");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript" };
const srv = http.createServer((q, r) => {
  const p = path.join(WEB, decodeURIComponent(new URL(q.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  if (!p.startsWith(WEB) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" }); r.end(fs.readFileSync(p));
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}/`;
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
const what = process.argv[2] || "filter";
let failed = false;
async function open(url) {
  const page = await b.newPage(), errors = [];
  page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(base + url); return { page, errors };
}

// Synthetic video cases: [name, query, expected bpm or null for no pulse]
const CASES = [
  ["52", "synth=52", 52], ["72", "synth=72", 72], ["96", "synth=96", 96], ["128", "synth=128", 128],
  ["72-faint", "synth=72&amp=0.002", 72], ["72-noisy", "synth=72&noise=6", 72],
  ["72-steady", "synth=72&drift=0", 72],
  ["none", "synth=0", null],
  ["none-flicker", "synth=0&flicker=1.3,0.003", null],
  ["72-flicker", "synth=72&flicker=1.3,0.003", 72],
];

if (what === "filter") {
  const { page, errors } = await open("index.html?nostart=1");
  const res = await page.evaluate(async () => {
    const E = await import("./evm.js");
    const dev = await (await navigator.gpu.requestAdapter()).requestDevice();
    const errs = []; dev.addEventListener("uncapturederror", (ev) => errs.push(ev.error.message));
    const out = [];
    // sizes chosen so the pyramid has odd edges (ceil) and the render target's row pitch is a multiple of 256 bytes
    for (const [w, h, levels, view] of [[128, 96, 3, 0], [192, 100, 4, 1], [64, 120, 2, 0]]) {
      const eng = await E.EvmEngine.create(dev, { w, h, levels, format: "rgba8unorm" });
      eng.params = { amp: 40, fl: 0.7, fh: 3.0, view }; eng.face = [0.31, 0.22, 0.69, 0.71]; eng.bg = [0.02, 0.1, 0.2, 0.93];
      const ref = new E.Reference(w, h, levels), cv = new OffscreenCanvas(w, h), cx = cv.getContext("2d");
      const target = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const rb = dev.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      let s = 12345; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
      const st = { filt: 0, lev: 0, roi: 0, comp: 0, compOff: 0, filtPeak: 0, frames: 0 };
      let t = 0;
      for (let f = 0; f < 40; f++) {
        t += [1 / 30, 1 / 30, 0.05, 1 / 15, 0.02][f % 5];   // uneven frame spacing, as dropped camera frames give
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++)
          img.data[(y * w + x) * 4 + c] = c === 3 ? 255 : 128 + 60 * Math.sin(x / 7 + c) * Math.cos(y / 9) + 6 * Math.sin(2 * Math.PI * 1.2 * t + c) * (x / w) + 20 * (rnd() - 0.5);
        cx.putImageData(img, 0, 0);
        const m = await eng.frame(cv, t, target.createView());
        const enc = dev.createCommandEncoder(); enc.copyTextureToBuffer({ texture: target }, { buffer: rb, bytesPerRow: w * 4 }, [w, h]);
        dev.queue.submit([enc.finish()]); await rb.mapAsync(GPUMapMode.READ); const comp = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap();
        const r = ref.step(img.data, t, eng.params, eng.face, eng.bg);
        const gf = await eng.read(eng.filt), gl = await eng.read(eng.bufs[levels]);
        for (let i = 0; i < eng.n; i++) for (let c = 0; c < 3; c++) {
          st.filt = Math.max(st.filt, Math.abs(gf[i * 4 + c] - r.filt[i * 3 + c])); st.filtPeak = Math.max(st.filtPeak, Math.abs(r.filt[i * 3 + c]));
          st.lev = Math.max(st.lev, Math.abs(gl[i * 4 + c] - r.levels[levels][i * 3 + c]));
        }
        for (const k of ["face", "bg", "filt"]) for (let c = 0; c < 3; c++) st.roi = Math.max(st.roi, Math.abs(m[k][c] - r.roi[k][c]));
        for (let i = 0; i < w * h * 4; i++) { const d = Math.abs(comp[i] - r.comp[i]); st.comp = Math.max(st.comp, d); if (d) st.compOff++; }
        st.frames++;
      }
      out.push({ w, h, levels, view, coarse: [eng.lw, eng.lh], ...st, compOffFrac: st.compOff / (40 * w * h * 4) });
      eng.destroy();
    }
    return { cases: out, errors: errs };
  });
  errors.push(...res.errors);
  for (const c of res.cases) {
    // f32 against float64: pyramid and filtered values to ~1e-6; the composite to within one 8-bit level
    const ok = c.lev < 1e-5 && c.filt < 1e-5 && c.roi < 1e-5 && c.comp <= 1 && c.compOffFrac < 0.01;
    console.log(`${ok ? "PASS" : "FAIL"} ${c.w}x${c.h} L${c.levels} view${c.view} -> ${c.coarse.join("x")}: level ${c.lev.toExponential(2)}, filtered ${c.filt.toExponential(2)} (peak ${c.filtPeak.toExponential(2)}), roi ${c.roi.toExponential(2)}, composite max ${c.comp} level, ${(100 * c.compOffFrac).toFixed(3)}% of bytes off`);
    failed ||= !ok;
  }
  fs.mkdirSync(OUT, { recursive: true }); fs.writeFileSync(path.join(OUT, "filter.json"), JSON.stringify({ cases: res.cases, errors }, null, 1));
  if (errors.length) { console.log("errors:", errors); failed = true; }
} else if (what === "rate") {
  const pick = process.argv.slice(3), rows = [], file = path.join(OUT, "rate.json");
  const prev = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file)).rows : [];
  for (const [name, q, expect] of CASES.filter(([n]) => !pick.length || pick.includes(n))) {
    const t0 = Date.now(), { page, errors } = await open(`index.html?${q}&seconds=${process.env.DUR || 20}`);
    await page.waitForFunction(() => window.__done || (window.__stats && window.__stats().errors.length), null, { timeout: 20 * 60e3, polling: 1000 });
    const s = await page.evaluate(() => window.__stats());
    await page.close();
    const last = s.log.filter((r) => r.t >= 10), err = (k) => expect == null ? null : last.map((r) => r[k] ? Math.abs(r[k].bpm - expect) : null);
    const mae = (k) => { const e = err(k); return e && e.every((v) => v != null) ? +(e.reduce((a, b) => a + b, 0) / e.length).toFixed(2) : null; };
    const row = { name, q, expect, windows: last.length, posMAE: mae("pos"), evmMAE: mae("evm"), greenMAE: mae("green"), shownErr: expect == null ? null : +Math.abs(last[last.length - 1].shown - expect).toFixed(2),
      final: last[last.length - 1], meanPosSNR: +(last.reduce((a, r) => a + (r.pos?.snr ?? 0), 0) / last.length).toFixed(2),
      suspectFrac: +(last.filter((r) => r.suspect).length / last.length).toFixed(2), meanExpCorr: +(last.reduce((a, r) => a + (r.expCorr ?? 0), 0) / last.length).toFixed(3),
      frameMs: s.frameMs, wallS: (Date.now() - t0) / 1000, errors: [...errors, ...s.errors] };
    rows.push(row);
    console.log(JSON.stringify({ name, expect, posMAE: row.posMAE, evmMAE: row.evmMAE, greenMAE: row.greenMAE, shownErr: row.shownErr, meanPosSNR: row.meanPosSNR, suspectFrac: row.suspectFrac, meanExpCorr: row.meanExpCorr, final: { pos: row.final.pos, bg: row.final.bg, bgDrift: row.final.bgDrift }, wallS: row.wallS }));
    if (row.errors.length) { console.log("errors:", row.errors); failed = true; }
  }
  const merged = [...prev.filter((r) => !rows.some((x) => x.name === r.name)), ...rows];
  const byName = Object.fromEntries(merged.map((r) => [r.name, r]));
  fs.mkdirSync(OUT, { recursive: true }); fs.writeFileSync(file, JSON.stringify({ seconds: +(process.env.DUR || 20), rows: merged, byName }, null, 1));
} else {
  const { page, errors } = await open("index.html?autostart=1&roi=fixed");
  await page.waitForTimeout(+(process.argv[3] || 15000));
  const s = await page.evaluate(() => window.__stats());
  console.log(JSON.stringify({ frames: s.frames, skipped: s.skipped, size: s.size, coarse: s.coarse, frameMs: s.frameMs, est: s.est && { pos: s.est.pos, fps: s.est.fps, bgDrift: s.est.bgDrift }, errors: [...errors, ...s.errors] }, null, 1));
  if (errors.length || s.errors.length || s.frames < 20) failed = true;
}
await b.close(); srv.close();
process.exit(failed ? 1 : 0);
