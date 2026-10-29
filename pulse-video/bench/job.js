// GPU Bridge job: the pulse-video engine on real hardware. Imports evm.js and synth.js from jsDelivr at the commit in
// inputs.meta.sha, checks the GPU against the float64 Reference on a small frame, then times whole frames
// (canvas upload, pyramid, band-pass, ROI means, composite draw, 48-byte readback) at camera sizes.
const { sha, frames = 150, sizes = [[320, 240, 3], [640, 480, 4], [480, 640, 4], [1280, 720, 5]] } = ctx.inputs.meta;
const root = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${sha}/pulse-video/web/`;
const E = await ctx.importModule(root + "evm.js"), S = await ctx.importModule(root + "synth.js");
const dev = await ctx.gpu();

ctx.progress(0, "accuracy check");
const w0 = 128, h0 = 96, chk = await E.EvmEngine.create(dev, { w: w0, h: h0, levels: 3 }), ref = new E.Reference(w0, h0, 3);
chk.params = { amp: 40, fl: 0.7, fh: 3, view: 0 };
const syn0 = new S.Synth({ w: w0, h: h0, bpm: 72 }), cv0 = new OffscreenCanvas(w0, h0), cx0 = cv0.getContext("2d", { willReadFrequently: true });
let maxErr = 0, peakFilt = 0;
for (let f = 0; f < 30; f++) {
  const t = f / 30; syn0.draw(cx0, t);
  await chk.frame(cv0, t);
  const r = ref.step(cx0.getImageData(0, 0, w0, h0).data, t, chk.params, chk.face, chk.bg), g = await chk.read(chk.filt);
  for (let i = 0; i < chk.n; i++) for (let c = 0; c < 3; c++) { maxErr = Math.max(maxErr, Math.abs(g[i * 4 + c] - r.filt[i * 3 + c])); peakFilt = Math.max(peakFilt, Math.abs(r.filt[i * 3 + c])); }
}
chk.destroy();
ctx.log(`filtered level, GPU f32 vs float64: max abs ${maxErr.toExponential(2)} (peak ${peakFilt.toExponential(2)})`);

const rows = [];
for (const [k, [w, h, levels]] of sizes.entries()) {
  ctx.progress((k + 1) / (sizes.length + 1), `${w}x${h}`);
  const eng = await E.EvmEngine.create(dev, { w, h, levels }), syn = new S.Synth({ w, h, bpm: 72 });
  const cv = new OffscreenCanvas(w, h), cx = cv.getContext("2d");
  const target = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const ms = [], drawMs = [];
  for (let f = 0; f < frames; f++) {
    const t = f / 30, d0 = performance.now(); syn.draw(cx, t); drawMs.push(performance.now() - d0);
    const t0 = performance.now(); await eng.frame(cv, t, target.createView()); ms.push(performance.now() - t0);
  }
  eng.destroy(); target.destroy();
  const q = (a, p) => { const s = a.slice(10).sort((x, y) => x - y); return +s[Math.floor(p * (s.length - 1))].toFixed(2); };
  const r = { w, h, levels, coarse: [eng.lw, eng.lh], medianMs: q(ms, 0.5), p90Ms: q(ms, 0.9), maxMs: q(ms, 1), synthDrawMs: q(drawMs, 0.5) };
  rows.push(r); ctx.log(JSON.stringify(r));
}
return { check: { maxErr, peakFilt }, frames, rows, adapter: await ctx.describe() };
