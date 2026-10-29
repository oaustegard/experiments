// Headless checks on SwiftShader WebGPU (correctness, not speed):
//   node test/verify.mjs selftest   GPU block stream against the float64 reference(), both kernels, walls changing
//   node test/verify.mjs modes      rectangle and disc drums on the GPU: spectral peaks against the discrete
//                                   dispersion relation, the continuous membrane and the Bessel zeros
//   node test/verify.mjs live       the page with Chromium's fake camera: walls found, blocks stream, no errors
// Results are written to ../results/<what>.json.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");   // the container has it global; require() honours NODE_PATH
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), WEB = path.join(HERE, "..", "web");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const srv = http.createServer((q, r) => {
  const p = path.join(WEB, decodeURIComponent(new URL(q.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  if (!p.startsWith(WEB) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" }); r.end(fs.readFileSync(p));
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}/`;
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
const page = await b.newPage(); const errors = [];
page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const what = process.argv[2] || "selftest";
let res;
if (what === "selftest") {
  await page.goto(base + "index.html?nostart=1");
  res = await page.evaluate(async () => {
    const E = await import("./engine.js");
    const ad = await navigator.gpu.requestAdapter(), dev = await ad.requestDevice({ requiredLimits: {
      maxComputeWorkgroupStorageSize: ad.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: ad.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: ad.limits.maxComputeWorkgroupSizeX } });
    const out = [];
    for (const kernel of ["tile", "multi"]) for (const [nx, ny, block] of [[40, 30, 256], [27, 19, 128]]) out.push(await E.selfTest(dev, { nx, ny, block, kernel, blocks: 6 }));
    return { limits: E.tileLimits(dev), runs: out };
  });
  res.pass = res.runs.every((r) => r.rmsDb < -80);
} else if (what === "modes") {
  await page.goto(base + "index.html?nostart=1");
  res = await page.evaluate(async () => {
    const E = await import("./engine.js");
    const ad = await navigator.gpu.requestAdapter(), dev = await ad.requestDevice({ requiredLimits: {
      maxComputeWorkgroupStorageSize: ad.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: ad.limits.maxComputeInvocationsPerWorkgroup } });
    const sr = 48000, B = 1024, nblk = 16, L = B * nblk;   // 16384 samples, 2.9 Hz bins
    function fft(re, im) {   // in place, radix 2
      const n = re.length;
      for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
      for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len;
        for (let i = 0; i < n; i += len) for (let k = 0; k < len / 2; k++) { const c = Math.cos(a * k), s = Math.sin(a * k), xr = re[i + k + len / 2], xi = im[i + k + len / 2];
          const tr = xr * c - xi * s, ti = xr * s + xi * c; re[i + k + len / 2] = re[i + k] - tr; im[i + k + len / 2] = im[i + k] - ti; re[i + k] += tr; im[i + k] += ti; } } }
    function peaks(x, fmax) {   // Hann window, log magnitude, local maxima 30 dB within the top, parabolic interpolation
      const n = x.length, re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) re[i] = x[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
      fft(re, im);
      const kmax = Math.floor(fmax / sr * n), db = new Float64Array(kmax + 2);
      for (let k = 0; k <= kmax + 1; k++) db[k] = 10 * Math.log10(re[k] ** 2 + im[k] ** 2 + 1e-30);
      const top = Math.max(...db.slice(2)), list = [];
      for (let k = 2; k <= kmax; k++) if (db[k] > db[k - 1] && db[k] >= db[k + 1] && db[k] > top - 30) {
        const d = 0.5 * (db[k - 1] - db[k + 1]) / (db[k - 1] - 2 * db[k] + db[k + 1]); list.push({ f: (k + d) * sr / n, db: db[k] - top });
      }
      return list;
    }
    async function run(nx, ny, mask, sx, sy, pl) {
      const e = await E.DrumEngine.create(dev, { nx, ny, block: B, sr, kernel: "tile", lam2: 0.45, a0: 1e-6, b: 0, gain: 1 });
      e.setMask(mask); e.setPickups(pl, pl);
      const x = new Float64Array(L);
      for (let k = 0; k < nblk; k++) { const o = await e.render(k ? [] : [{ x: sx, y: sy, r: 1.2, amp: 1, js: 0, T: 3 }]); for (let n = 0; n < B; n++) x[k * B + n] = o[2 * n]; }
      e.destroy(); return x;
    }
    const cents = (a, c) => 1200 * Math.log2(a / c);
    // rectangle: a 33 x 23 open interior inside a 35 x 25 grid with a wall ring, so the drum is 34 x 24 spacings
    const nx = 35, ny = 25, m = new Float32Array(nx * ny);
    for (let y = 1; y < ny - 1; y++) for (let x = 1; x < nx - 1; x++) m[y * nx + x] = 1;
    const xr = await run(nx, ny, m, 7.3, 5.2, 4 + 17 * nx), pr = peaks(xr, 6000);
    const pred = E.rectModes(nx - 2, ny - 2, 0.45, sr, 10);
    const rect = pred.map((md) => { const near = pr.reduce((a, c) => Math.abs(c.f - md.fd) < Math.abs(a.f - md.fd) ? c : a, { f: 1e9 });
      return { p: md.p, q: md.q, fDiscrete: +md.fd.toFixed(2), fContinuous: +md.fc.toFixed(2), fMeasured: +near.f.toFixed(2), centsVsDiscrete: +cents(near.f, md.fd).toFixed(2), centsVsContinuous: +cents(near.f, md.fc).toFixed(2) }; });
    // disc of radius R cells: partial ratios against the Bessel zeros j_{m,k} / j_{0,1}
    const R = 15.5, D = 34, md = new Float32Array(D * D), c0 = (D - 1) / 2;
    for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) md[y * D + x] = Math.hypot(x - c0, y - c0) < R ? 1 : 0;
    const xd = await run(D, D, md, c0 + 4.3, c0 - 2.7, Math.round(c0 - 6) + Math.round(c0 + 3) * D), pd = peaks(xd, 4000);
    const J = [[0, 1, 2.404826], [1, 1, 3.831706], [2, 1, 5.135622], [0, 2, 5.520078], [3, 1, 6.380162], [1, 2, 7.015587], [4, 1, 7.588342], [2, 2, 8.417244]];
    const fc = (j) => Math.sqrt(0.45) * sr * j / (2 * Math.PI * R);
    const disc = J.map(([mm, k, j]) => { const f = fc(j), near = pd.reduce((a, c) => Math.abs(c.f - f) < Math.abs(a.f - f) ? c : a, { f: 1e9 });
      const ok = Math.abs(cents(near.f, f)) < 80;   // further off is another mode: this one is on a node of the pickup or strike
      return { m: mm, k, bessel: j, fContinuous: +f.toFixed(2), fMeasured: ok ? +near.f.toFixed(2) : null, cents: ok ? +cents(near.f, f).toFixed(1) : null }; });
    const ratio = disc.map((d) => d.fMeasured && +(d.fMeasured / disc[0].fMeasured).toFixed(4));
    return { sr, samples: L, binHz: sr / L, rect, disc, discRatios: ratio, besselRatios: J.map((j) => +(j[2] / J[0][2]).toFixed(4)),
      maxCentsRectDiscrete: Math.max(...rect.map((r) => Math.abs(r.centsVsDiscrete))), maxCentsDisc: Math.max(...disc.filter((d) => d.cents !== null).map((d) => Math.abs(d.cents))),
      maxRatioErrPct: Math.max(...disc.filter((d) => d.fMeasured).map((d) => 100 * Math.abs(d.fMeasured / disc[0].fMeasured / (d.bessel / J[0][2]) - 1))) };
  });
  res.pass = res.maxCentsRectDiscrete < 5;
} else {
  await page.goto(base + `index.html?autostart=1&${process.argv[3] || ""}`);
  await page.waitForTimeout(+(process.argv[4] || 15000));
  const before = await page.evaluate(() => window.__stats().strikes);
  for (let k = 1; k <= 3; k++) { await page.keyboard.press(`Digit${k}`); await page.waitForTimeout(400); }   // the pads, from the keyboard
  res = await page.evaluate(() => window.__stats && window.__stats());
  res.padStrikes = res.strikes - before;
  res.pass = !!res && res.sent > 4 && res.walls > 0 && res.pads.length > 0 && res.padStrikes > 0 && res.errors.length === 0;
}
const out = { what, res, errors };
console.log(JSON.stringify(out, null, 1));
fs.writeFileSync(path.join(HERE, "..", "results", `${what}-swiftshader.json`), JSON.stringify(out, null, 1) + "\n");
await b.close(); srv.close();
if (errors.length || !res?.pass) process.exit(1);
