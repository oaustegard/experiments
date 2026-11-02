// GPU Bridge job: check the page's time estimate. Probe like page.js sizeMenu(), predict, then train
// with the page's settings and compare. Inputs: rgba, cfg {W, H, ns: [..], profile: {pixBudget, stepsScale}}, trainer.
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");
const cfg = ctx.inputs.cfg, P = cfg.profile;
const trainer = await import(URL.createObjectURL(new Blob([new TextDecoder().decode(ctx.inputs.trainer)], { type: "text/javascript" })));
tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);
await tf.setBackend("webgl"); await tf.ready();
const t0 = performance.now();
const small = await trainer.probe(tf, { n: 64, nPix: 1024 }), big = await trainer.probe(tf, { n: 256, nPix: 8192 });
const probeSec = (performance.now() - t0) / 1000;
const b = Math.max(0, (big - small) / (256 * 8192 - 64 * 1024)), a0 = Math.max(0, small - b * 64 * 1024);
const work = (k) => k * Math.min(8192, Math.max(1024, Math.floor(P.pixBudget / k)));
const est = (n) => { const s = trainer.checkpoints(n), sc = P.stepsScale, ms = (k) => a0 + b * work(k);
  return (s.slice(0, -1).reduce((acc, k) => acc + 100 * sc * ms(k), 0) + 450 * sc * ms(n)) / 1000 + 2; };
ctx.log(`probe ${probeSec.toFixed(1)} s: small ${small.toFixed(1)} ms, big ${big.toFixed(1)} ms`);
const runs = [];
for (const n of cfg.ns) {
  const r = await trainer.encode(tf, { rgba: ctx.inputs.rgba, width: cfg.W, height: cfg.H },
    { n, init: "structure", importance: 0.5, pixBudget: P.pixBudget, stepsScale: P.stepsScale });
  runs.push({ n, est: est(n), actual: r.seconds, psnr: r.points.at(-1).psnr });
  ctx.log(`n=${n}: estimated ${est(n).toFixed(1)} s, took ${r.seconds.toFixed(1)} s, ${r.points.at(-1).psnr.toFixed(2)} dB`);
}
return { small, big, probeSec, runs };
