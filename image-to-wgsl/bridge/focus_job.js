// GPU Bridge job: web/focus.js on a real device (face detector and depth model, both downloaded from CDNs),
// then a focused WGSL training run, to check that foveation works and how long each part takes there.
// Inputs: rgba (u8), cfg {W, H, n, depthAt: [x, y]}, trainer, gputrain, focus (sources of the web/ files).
const cfg = ctx.inputs.cfg, dec = (b) => new TextDecoder().decode(b), blob = (s) => URL.createObjectURL(new Blob([s], { type: "text/javascript" }));
const tUrl = blob(dec(ctx.inputs.trainer)), trainer = await import(tUrl);
const { encodeGPU } = await import(blob(dec(ctx.inputs.gputrain).replace('from "./trainer.js"', `from "${tUrl}"`)));
const focus = await import(blob(dec(ctx.inputs.focus)));
const W = cfg.W, H = cfg.H, img = { rgba: ctx.inputs.rgba, width: W, height: H }, out = {};
const time = async (name, f) => { const t = performance.now(); try { const v = await f(); out[name + "Ms"] = Math.round(performance.now() - t); return v; }
  catch (e) { out[name + "Error"] = String(e?.message || e); ctx.log(name + " failed: " + out[name + "Error"]); return null; } };
ctx.progress(0.05, "face detector");
const faces = await time("face", () => focus.detectFaces(img));
out.faces = faces; ctx.log(`faces: ${JSON.stringify(faces)} in ${out.faceMs} ms`);
ctx.progress(0.3, "depth model");
const depth = await time("depth", () => focus.depthMap(img));
if (depth) {
  const [x, y] = cfg.depthAt, d = depth[Math.floor(y * H) * W + Math.floor(x * W)];
  const m = await focus.focusMap("depth", img, { depth: d, width: 0.15 });
  let s = 0; for (const v of m) s += v; out.depthAt = d; out.depthMapMean = s / m.length;
  ctx.log(`depth in ${out.depthMs} ms; depth at (${x}, ${y}) = ${d.toFixed(3)}; band covers ${(100 * out.depthMapMean).toFixed(1)}% (mean weight)`);
}
if (faces?.length) {
  ctx.progress(0.6, "focused training");
  const fmap = trainer.normaliseFocus(await focus.focusMap("face", img, { faces }), W * H);
  const dev = await ctx.gpu();
  const r = await time("train", () => encodeGPU(dev, img, { n: cfg.n, focus: fmap,
    onProgress: (p) => ctx.progress(0.6 + 0.4 * p.step / p.totalSteps, `n=${cfg.n} ${p.phase}`) }));
  if (r) { const p = r.points[r.points.length - 1]; out.train = { n: cfg.n, seconds: r.seconds, psnr: p.psnr, psnrFocus: p.psnrFocus };
    ctx.log(`face-focused n=${cfg.n}: ${r.seconds.toFixed(1)} s, ${p.psnr.toFixed(2)} dB, ${p.psnrFocus.toFixed(2)} dB in focus`); }
}
return out;
