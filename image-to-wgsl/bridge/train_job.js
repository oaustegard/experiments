// GPU Bridge job: run web/trainer.js (TF.js on WebGPU) on a real device.
// Inputs: rgba (u8, W*H*4), cfg (json {W, H, runs: [{n, stepsStage, stepsFinal, stepsQat}]}),
// trainer (u8, the source of web/trainer.js, imported from a blob URL).
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
              "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js");
const cfg = ctx.inputs.cfg;
const src = new TextDecoder().decode(ctx.inputs.trainer);
const trainer = await import(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
const want = cfg.backend || "webgl";
try { if (!(await tf.setBackend(want))) throw new Error("refused"); await tf.ready(); }
catch (e) { ctx.log(want + " backend failed: " + e.message); await tf.setBackend("cpu"); await tf.ready(); }
const out = { backend: tf.getBackend(), runs: [], cumsum: [] };
if (cfg.cumsumCheck) // Direct check of tf.cumsum on this backend against a JS reference (rows of length G, last axis).
for (const G of [16, 32, 33, 64, 128, 256]) {
  const P = 1000, a = new Float32Array(P * G);
  for (let i = 0; i < a.length; i++) a[i] = -Math.random();
  const got = await tf.cumsum(tf.tensor2d(a, [P, G]), 1).data();
  let maxErr = 0;
  for (let r = 0; r < P; r++) { let acc = 0; for (let j = 0; j < G; j++) { acc += a[r * G + j]; maxErr = Math.max(maxErr, Math.abs(acc - got[r * G + j])); } }
  out.cumsum.push({ G, maxErr });
}
ctx.log("cumsum max error by row length: " + JSON.stringify(out.cumsum));
for (const [ri, run] of cfg.runs.entries()) {
  const trace = [];
  const r = await trainer.encode(tf, { rgba: ctx.inputs.rgba, width: cfg.W, height: cfg.H }, {
    ...run, onProgress: (p) => { trace.push([p.phase, p.step, +p.psnrSample.toFixed(2), +p.seconds.toFixed(1)]);
      ctx.progress((ri + p.step / p.totalSteps) / cfg.runs.length, `n=${run.n} ${p.phase}`); } });
  out.runs.push({ cfg: run, seconds: r.seconds, points: r.points, trace, P: Array.from(r.P), bg: Array.from(r.bg),
                  A: r.A, wgsl: trainer.emitWGSL(r) });
  ctx.log(`n=${run.n}: ${r.seconds.toFixed(1)} s, ${JSON.stringify(r.points.map((p) => [p.n, +p.psnr.toFixed(2)]))}`);
}
return out;
