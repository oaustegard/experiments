// GPU Bridge job: deep zoom on one high-resolution image (web/deepzoom.js), measured.
// Inputs: png (u8, the source image file), cfg (JSON), and the sources of web/trainer.js, gputrain.js, instanced.js
// and deepzoom.js (mods: u8 JSON {name: source}), imported from Blob URLs with their relative imports rewritten.
// cfg: {id, base: long side of the base fit (768), nBase, nTile, levels: [1, 2] (which levels to measure),
//   arms: ["residual", "prefix", "indep", "indep2"], steps (tile schedule), baseSteps, tiles: null | [[L, tx, ty], ...]
//   (a subset, for timing runs), views: [[L, fx, fy, size]] (renders to return), draw: [[w, h, L]] (frame timing)}.
// Per tile and arm it records the core's sum of squared 8-bit errors against the level, so per-level PSNR can be
// pooled over tiles: base only, base plus the levels below, base plus detail up to the level, independent tiles
// (fitted alone, no background from coarser levels), and bicubic upscaling of the base-size raster.
const cfg = ctx.inputs.cfg, dec = (b) => new TextDecoder().decode(b);
const mods = JSON.parse(dec(ctx.inputs.mods)), urls = {};
const load = async (name) => {
  if (urls[name]) return import(urls[name]);
  let src = mods[name];
  for (const dep of ["trainer.js", "gputrain.js", "instanced.js"]) if (src.includes(`"./${dep}"`)) { await load(dep); src = src.replaceAll(`"./${dep}"`, `"${urls[dep]}"`); }
  urls[name] = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
  return import(urls[name]);
};
const { encodeGPU } = await load("gputrain.js");
const { emitInstanced, createRenderer } = await load("instanced.js");
const DZ = await load("deepzoom.js");
const dev = await ctx.gpu();
dev.addEventListener("uncapturederror", (e) => ctx.log("gpu error: " + e.error.message));

// source and level images, resampled by the browser as the page does
const bmp = await createImageBitmap(new Blob([ctx.inputs.png]));
const Ws = bmp.width, Hs = bmp.height, sc = Math.min(1, cfg.base / Math.max(Ws, Hs));
const W0 = Math.round(Ws * sc), H0 = Math.round(Hs * sc);
function resample(w, h) {
  if (w === Ws && h === Hs) { const c = new OffscreenCanvas(w, h), x = c.getContext("2d"); x.drawImage(bmp, 0, 0); return x.getImageData(0, 0, w, h).data; }
  const c = new OffscreenCanvas(w, h), x = c.getContext("2d"); x.imageSmoothingQuality = "high"; x.drawImage(bmp, 0, 0, w, h);
  return x.getImageData(0, 0, w, h).data;
}
const L0 = resample(W0, H0);
const levels = DZ.pyramid(W0, H0, Ws, Hs);
const levelImg = levels.map((lv) => resample(lv.W, lv.H));
const crop = async (lv, x0, y0, x1, y1) => {
  const src = levelImg[lv.L - 1], w = x1 - x0, out = new Uint8ClampedArray(w * (y1 - y0) * 4);
  for (let y = y0; y < y1; y++) out.set(src.subarray((y * lv.W + x0) * 4, (y * lv.W + x1) * 4), (y - y0) * w * 4);
  return out;
};
const out = { id: cfg.id, Ws, Hs, W0, H0, levels: levels.map(({ L, s, W, H, TX, TY }) => ({ L, s, W, H, TX, TY })), tiles: [], arms: {}, views: [], draws: [] };
ctx.log(`source ${Ws}x${Hs}, base ${W0}x${H0}, levels ${levels.map((l) => `${l.W}x${l.H}`).join(" ")}`);

// bicubic (Keys, a = -0.5) upscale of the base raster, over a core rectangle of a level
function bicubicSSE(lv, core, tgt, crop) {
  const k = (t) => { t = Math.abs(t); return t <= 1 ? 1.5 * t ** 3 - 2.5 * t ** 2 + 1 : t < 2 ? -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2 : 0; };
  const at = (x, y, c) => L0[(Math.min(H0 - 1, Math.max(0, y)) * W0 + Math.min(W0 - 1, Math.max(0, x))) * 4 + c];
  const w = crop[2] - crop[0]; let s = 0;
  for (let y = core[1]; y < core[3]; y++) {
    const v = (y + 0.5) * H0 / lv.H - 0.5, y0 = Math.floor(v), wy = [k(v - y0 + 1), k(v - y0), k(v - y0 - 1), k(v - y0 - 2)];
    for (let x = core[0]; x < core[2]; x++) {
      const u = (x + 0.5) * W0 / lv.W - 0.5, x0 = Math.floor(u), wx = [k(u - x0 + 1), k(u - x0), k(u - x0 - 1), k(u - x0 - 2)];
      const i = ((y - crop[1]) * w + (x - crop[0])) * 4;
      for (let c = 0; c < 3; c++) {
        let a = 0; for (let j = 0; j < 4; j++) for (let m = 0; m < 4; m++) a += wy[j] * wx[m] * at(x0 - 1 + m, y0 - 1 + j, c);
        const d = Math.round(Math.min(255, Math.max(0, a))) - tgt[i + c]; s += d * d;
      }
    }
  }
  return s;
}

// base list
const t0 = performance.now();
const base = await encodeGPU(dev, { rgba: L0, width: W0, height: H0 }, { n: cfg.nBase, ...(cfg.baseSteps || {}),
  onProgress: (p) => ctx.progress(0.05 * p.step / p.totalSteps, `base ${p.phase}`) });
out.base = { n: cfg.nBase, seconds: base.seconds, psnr: base.points[base.points.length - 1].psnr, P: Array.from(base.P), bg: Array.from(base.bg), A: base.A };
ctx.log(`base n=${cfg.nBase}: ${base.seconds.toFixed(1)} s, ${out.base.psnr.toFixed(2)} dB at ${W0}x${H0}`);
const rend = createRenderer(dev, "rgba8unorm", emitInstanced(base));

// which tiles: all tiles of the measured levels, coarse first, or a given subset
const want = cfg.tiles || levels.filter((lv) => (cfg.levels || [1, 2, 3, 4]).includes(lv.L)).flatMap((lv) => {
  const a = []; for (let ty = 0; ty < lv.TY; ty++) for (let tx = 0; tx < lv.TX; tx++) a.push([lv.L, tx, ty]); return a; });
const steps = { ...DZ.TILE_STEPS, ...(cfg.steps || {}) };
const totalWork = want.length * (cfg.arms || ["residual"]).length;
let done = 0;
const tick = (label) => { done++; ctx.progress(0.05 + 0.9 * done / totalWork, label); };

for (const arm of cfg.arms || ["residual"]) {
  const rec = out.arms[arm] = { tiles: [], seconds: 0 };
  const ta = performance.now();
  if (arm === "residual" || arm === "prefix") {
    const dz = new DZ.DeepZoom(dev, { base: { ...base, n: cfg.nBase }, W0, H0, Ws, Hs, crop, nTile: cfg.nTile, prefix: arm === "prefix", steps });
    dz.attach(rend);
    // a subset still needs its dependencies first
    const order = [], seen = new Set();
    const add = (t) => { const k = t.join(","); if (seen.has(k)) return; seen.add(k); for (const d of dz.deps(...t)) add(d); order.push(t); };
    for (const t of want) add(t);
    for (const [L, tx, ty] of order) {
      const r = await dz.trainTile(L, tx, ty);
      const lv = levels[L - 1], tgt = await crop(lv, ...r.crop), w = r.crop[2] - r.crop[0], h = r.crop[3] - r.crop[1];
      const t = { L, tx, ty, n: r.n, coreN: r.coreN, seconds: +r.seconds.toFixed(3), cropMs: +r.cropMs.toFixed(1), bgMs: +r.bgMs.toFixed(1), totalMs: +r.totalMs.toFixed(1),
        sseBelow: r.sseBelow, sseDetail: r.coreN * 3 * 255 * 255 / 10 ** (r.psnr / 10) };
      if (arm === "residual") {
        const bOnly = L === 1 ? null : await rend.readback(w, h, cfg.nBase, DZ.cropCam(lv, r.crop, base.A), 0);
        t.sseBase = L === 1 ? r.sseBelow : DZ.coreSSE(bOnly, tgt, r.crop, r.core);
        t.sseBicubic = bicubicSSE(lv, r.core, tgt, r.crop);
        // check: the renderer, drawing the tile over its background, gives what the trainer measured
        if (cfg.checkDraw) {
          const img = await rend.readback(w, h, cfg.nBase, DZ.cropCam(lv, r.crop, base.A), dz.countFor(L));
          t.sseDrawn = DZ.coreSSE(img, tgt, r.crop, r.core);
        }
        if (r.points.length > 1) t.prefix = r.points.map((p) => [p.n, +p.psnrFocus.toFixed(3)]);
      }
      rec.tiles.push(t);
      tick(`${arm} L${L} ${tx},${ty}`);
    }
    if (arm === "residual") {
      out.dz = dz.stats(); out.detailBytes = dz.bytes;
      // renders of chosen views: base only, base + each level, and the source
      for (const [L, fx, fy, size] of cfg.views || []) {
        const lv = levels[L - 1], x0 = Math.max(0, Math.min(lv.W - size, Math.round(fx * lv.W - size / 2))), y0 = Math.max(0, Math.min(lv.H - size, Math.round(fy * lv.H - size / 2)));
        const c = [x0, y0, x0 + size, y0 + size], cam = DZ.cropCam(lv, c, base.A), v = { L, crop: c };
        const u8 = (f) => Uint8Array.from(f, (x) => Math.round(Math.min(1, Math.max(0, x)) * 255));
        for (let k = 0; k <= L; k++) out[`view_${out.views.length}_d${k}`] = u8(await rend.readback(size, size, cfg.nBase, cam, dz.countFor(k)));
        out[`view_${out.views.length}_src`] = Uint8Array.from(await crop(lv, ...c));
        out.views.push(v);
      }
      // seams: render whole levels in 512-pixel chunks and pool squared error by distance to the nearest inner tile
      // boundary (0 = a pixel touching it), with detail and with the base alone (which has no seams)
      out.seams = [];
      const BINS = [0, 1, 2, 4, 8, 16, 1e9];
      for (const L of cfg.seams || []) {
        const lv = levels[L - 1], acc = { L, bins: BINS.slice(0, -1), n: BINS.slice(0, -1).map(() => 0), detail: BINS.slice(0, -1).map(() => 0), base: BINS.slice(0, -1).map(() => 0) };
        const dist = (x, W, T) => { let d = 1e9; for (let k = 1; k * DZ.TILE < W; k++) d = Math.min(d, x < k * DZ.TILE ? k * DZ.TILE - 1 - x : x - k * DZ.TILE); return d; };
        for (let y0 = 0; y0 < lv.H; y0 += 512) for (let x0 = 0; x0 < lv.W; x0 += 512) {
          const c = [x0, y0, Math.min(lv.W, x0 + 512), Math.min(lv.H, y0 + 512)], w = c[2] - c[0], h = c[3] - c[1], cam = DZ.cropCam(lv, c, base.A);
          const tgt = await crop(lv, ...c), imD = await rend.readback(w, h, cfg.nBase, cam, dz.countFor(L)), imB = await rend.readback(w, h, cfg.nBase, cam, 0);
          for (let y = 0; y < h; y++) { const dy = dist(y0 + y, lv.H); for (let x = 0; x < w; x++) {
            const d = Math.min(dy, dist(x0 + x, lv.W)); let bi = 0; while (d >= BINS[bi + 1]) bi++;
            const i = (y * w + x) * 4; let sd = 0, sb = 0;
            for (let ch = 0; ch < 3; ch++) { const t = tgt[i + ch], u = Math.round(Math.min(1, Math.max(0, imD[i + ch])) * 255) - t, v = Math.round(Math.min(1, Math.max(0, imB[i + ch])) * 255) - t; sd += u * u; sb += v * v; }
            acc.n[bi]++; acc.detail[bi] += sd; acc.base[bi] += sb; } }
        }
        acc.detailDb = acc.n.map((n, i) => n ? +(10 * Math.log10(255 * 255 * 3 * n / acc.detail[i])).toFixed(3) : null);
        acc.baseDb = acc.n.map((n, i) => n ? +(10 * Math.log10(255 * 255 * 3 * n / acc.base[i])).toFixed(3) : null);
        out.seams.push(acc); ctx.log(`seams L${L}: detail ${JSON.stringify(acc.detailDb)} base ${JSON.stringify(acc.baseDb)} px ${JSON.stringify(acc.n)}`);
      }
      // frame times: base alone and with every detail splat up to level L, at w x h (median of 7)
      for (const [w, h, L] of cfg.draw || []) {
        const tex = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const d = { w, h, L, nDetail: dz.countFor(L) };
        for (const [tag, nd] of [["base", 0], ["detail", dz.countFor(L)]]) {
          const cam = { z: Math.max(1, levels[L - 1].s * H0 / Math.min(w / base.A, h)), x: 0.5, y: 0.5 }, ts = [];
          for (let i = 0; i < 8; i++) { const s = performance.now(); rend.draw(tex.createView(), w, h, cfg.nBase, cam, nd); await dev.queue.onSubmittedWorkDone(); if (i) ts.push(performance.now() - s); }
          ts.sort((a, b) => a - b); d[tag] = +ts[ts.length >> 1].toFixed(2);
        }
        tex.destroy(); out.draws.push(d); ctx.log(JSON.stringify(d));
      }
    }
  } else {   // independent tiles: fitted alone over the plain four-corner background, scored on the core
    const nMul = arm === "indep2" ? 2 : 1;
    for (const [L, tx, ty] of want) {
      const lv = levels[L - 1], { core, crop: c } = DZ.tileGeom(lv, tx, ty), w = c[2] - c[0], h = c[3] - c[1];
      const rgba = await crop(lv, ...c), mask = new Float32Array(w * h);
      for (let y = core[1]; y < core[3]; y++) for (let x = core[0]; x < core[2]; x++) mask[(y - c[1]) * w + (x - c[0])] = 1;
      const n = Math.max(16, Math.round(cfg.nTile * nMul * Math.min(1, w * h / (DZ.TILE + 2 * DZ.MARGIN) ** 2)));
      const r = await encodeGPU(dev, { rgba, width: w, height: h }, { n, ...steps, prefix: false, focusEval: mask, seed: L * 7919 + ty * 131 + tx });
      const coreN = (core[2] - core[0]) * (core[3] - core[1]);
      rec.tiles.push({ L, tx, ty, n, coreN, seconds: +r.seconds.toFixed(3), sseDetail: coreN * 3 * 255 * 255 / 10 ** (r.points[0].psnrFocus / 10) });
      tick(`${arm} L${L} ${tx},${ty}`);
    }
  }
  rec.seconds = (performance.now() - ta) / 1000;
  // pooled PSNR per level
  const db = (sse, n) => +(10 * Math.log10(255 * 255 * 3 * n / sse)).toFixed(3);
  rec.levels = levels.map((lv) => {
    const ts = rec.tiles.filter((t) => t.L === lv.L); if (!ts.length) return null;
    const N = ts.reduce((a, t) => a + t.coreN, 0), sum = (k) => ts.reduce((a, t) => a + t[k], 0), o = { L: lv.L, tiles: ts.length, splats: ts.reduce((a, t) => a + t.n, 0), seconds: +ts.reduce((a, t) => a + t.seconds, 0).toFixed(2) };
    o.detail = db(sum("sseDetail"), N);
    if (ts[0].sseBelow != null) o.below = db(sum("sseBelow"), N);
    if (ts[0].sseBase != null) { o.base = db(sum("sseBase"), N); o.bicubic = db(sum("sseBicubic"), N); }
    if (ts[0].sseDrawn != null) o.drawn = db(sum("sseDrawn"), N);
    return o;
  }).filter(Boolean);
  ctx.log(`${arm}: ${rec.seconds.toFixed(1)} s ${JSON.stringify(rec.levels)}`);
}
out.seconds = (performance.now() - t0) / 1000;
rend.destroy();
return out;
