// GPU Bridge job: train with web/gputrain.js (WGSL, footprint-limited) on a real device, and optionally draw
// splat lists with web/instanced.js to check and time the renderer there.
// Inputs: rgba (u8), cfg {W, H, probe: [n...], runs: [{n, stepsStage, stepsFinal, stepsQat}], render: bool},
// trainer (trainer.js source), gputrain (gputrain.js source; its "./trainer.js" import is pointed at the trainer
// blob), optional instanced (instanced.js source), optional lists ({tag: {P, bg, A, n}}: drawn without training).
const cfg = ctx.inputs.cfg, dec = (b) => new TextDecoder().decode(b), blob = (s) => URL.createObjectURL(new Blob([s], { type: "text/javascript" }));
const tUrl = blob(dec(ctx.inputs.trainer));
const { encodeGPU, probeGPU } = await import(blob(dec(ctx.inputs.gputrain).replace('from "./trainer.js"', `from "${tUrl}"`)));
const inst = ctx.inputs.instanced ? await import(blob(dec(ctx.inputs.instanced).replace('from "./trainer.js"', `from "${tUrl}"`))) : null;
const dev = await ctx.gpu();
dev.addEventListener("uncapturederror", (e) => ctx.log("gpu error: " + e.error.message));
const out = { runs: [], probe: [], draws: [], limits: { maxStorageBufferBindingSize: dev.limits.maxStorageBufferBindingSize, maxBufferSize: dev.limits.maxBufferSize },
  float32Blendable: dev.features.has("float32-blendable") };
const tgt = ctx.inputs.rgba;

// Draw a list with the instanced renderer at W x H into rgba8unorm, score it against the target, and time
// frames at two larger sizes (median of 5; a frame over 2 s stops the timing).
async function drawCheck(tag, r) {
  const code = inst.emitInstanced(r), mod = dev.createShaderModule({ code }), info = await mod.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error").map((m) => m.message);
  if (errs.length) { const d = { tag, n: r.n, errors: errs.slice(0, 3) }; out.draws.push(d); ctx.log(JSON.stringify(d)); return; }
  const rd = inst.createRenderer(dev, "rgba8unorm", code), W = cfg.W, H = cfg.H;
  const tex = dev.createTexture({ size: [W, H], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  rd.draw(tex.createView(), W, H, r.n);
  const bpr = Math.ceil(W * 4 / 256) * 256, buf = dev.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]); dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange()); let s = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) { const d = src[y * bpr + x * 4 + c] - tgt[(y * W + x) * 4 + c]; s += d * d; }
  buf.unmap(); buf.destroy(); tex.destroy();
  const d = { tag, n: r.n, codeLen: code.length, format: rd.format, psnr: +(10 * Math.log10(255 * 255 / Math.max(s / (W * H * 3), 1e-12))).toFixed(3), ms: {} };
  for (const [w, h] of [[768, 512], [1920, 1280]]) {
    const t2 = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT }), times = [];
    for (let i = 0; i < 6; i++) { const t0 = performance.now(); rd.draw(t2.createView(), w, h, r.n); await dev.queue.onSubmittedWorkDone(); const ms = performance.now() - t0; if (i) times.push(ms); if (ms > 2000) break; }
    times.sort((a, b) => a - b); d.ms[`${w}x${h}`] = times.length ? +times[times.length >> 1].toFixed(1) : null; t2.destroy();
  }
  rd.destroy(); out.draws.push(d); ctx.log(JSON.stringify(d));
}

for (const n of cfg.probe || []) {   // ms per training step, as the page's menu measures it
  const ms = await probeGPU(dev, { rgba: tgt, width: cfg.W, height: cfg.H }, { n });
  out.probe.push({ n, ms }); ctx.log(`probe n=${n}: ${ms.toFixed(2)} ms/step`);
}
const lists = ctx.inputs.lists instanceof Uint8Array ? JSON.parse(dec(ctx.inputs.lists)) : ctx.inputs.lists || {};   // u8 JSON past the argv limit
for (const [tag, l] of Object.entries(lists)) await drawCheck(tag, { P: Int32Array.from(l.P), bg: Int32Array.from(l.bg), A: l.A, n: l.n });
for (const [ri, run] of (cfg.runs || []).entries()) {
  const trace = [];
  const r = await encodeGPU(dev, { rgba: tgt, width: cfg.W, height: cfg.H }, { ...run,
    onProgress: (p) => { trace.push([p.phase, p.step, +p.psnrSample.toFixed(2), +p.seconds.toFixed(2)]); ctx.progress((ri + p.step / p.totalSteps) / cfg.runs.length, `n=${run.n} ${p.phase}`); } });
  out.runs.push({ cfg: run, seconds: r.seconds, points: r.points, overflows: r.overflows, steps: r.steps, trace, P: Array.from(r.P), bg: Array.from(r.bg), A: r.A });
  ctx.log(`${run.tag || ""} n=${run.n}: ${r.seconds.toFixed(1)} s, ${r.steps} steps, overflows ${r.overflows}, ${JSON.stringify(r.points.map((p) => [p.n, +p.psnr.toFixed(2)]))}`);
  if (inst && cfg.render) await drawCheck(`${run.tag || ""}-trained`, r);
}
return out;
