// GPU-bridge job: check diffusion.js against the torch fixtures on real hardware, then time it.
// inputs: src (diffusion.js bytes), man (manifest.json bytes), w (weights.bin as u16),
//         cases (test/cases.json bytes, optional), noise (1x3x64x64 f32, optional: returns that sample)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { ImageDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const w = f16ToF32(ctx.inputs.w);
const info = ctx.describe ? await ctx.describe() : null;
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
const out = { checks: [], runs: [], errs };

if (ctx.inputs.cases) {
  const cases = JSON.parse(td.decode(ctx.inputs.cases));
  for (const act of ["f32", "f16"]) {
    const m = new ImageDiffusion(device, man, w, { act });
    for (const c of cases) {
      let got, want;
      if (c.kind === "forward") {
        got = await m.forward(new Float32Array(c.x), c.t, c.y, c.H, c.W);
        want = c.v;
      } else {
        await m.sample({ cls: c.y, H: c.H, W: c.W, steps: c.steps, cfg: c.cfg, noise: new Float32Array(c.noise) });
        got = (await m.read(m.dBuf, 3 * c.H * c.W)).map((v) => Math.max(-1, Math.min(1, v)));
        want = c.out;
      }
      let maxd = 0, maxw = 0;
      for (let i = 0; i < want.length; i++) { maxd = Math.max(maxd, Math.abs(got[i] - want[i])); maxw = Math.max(maxw, Math.abs(want[i])); }
      out.checks.push({ act, kind: c.kind, H: c.H, W: c.W, rel: maxd / maxw, nan: got.some(Number.isNaN) });
    }
    m.freePlan();
  }
  ctx.progress(0.3, "checks done");
}

const m = new ImageDiffusion(device, man, w);
const configs = [{ H: 64, cfg: 2, steps: 24 }, { H: 64, cfg: 1, steps: 24 }, { H: 64, cfg: 2, steps: 12 },
                 { H: 128, cfg: 2, steps: 24 }];
for (const c of configs) {
  for (let rep = 0; rep < 3; rep++) {          // first run warms pipelines and buffers
    const t0 = performance.now();
    await m.sample({ cls: 0, H: c.H, W: c.H, steps: c.steps, cfg: c.cfg, seed: 5 });
    await device.queue.onSubmittedWorkDone();
    const total = performance.now() - t0;
    if (rep > 0) out.runs.push({ ...c, rep, total_ms: total, step_ms: total / c.steps, bytes: m.bytes });
  }
}
// share of a guided 64x64 forward spent in attention: time 20 forwards with and without the attention ops
// (without them the output is wrong; only the time matters)
m.plan(2, 64, 64);
const all = m.ops;
const timeFwd = async (ops) => {
  m.ops = ops;
  const t0 = performance.now();
  for (let i = 0; i < 20; i++) { const enc = device.createCommandEncoder(); m.encodeForward(enc); device.queue.submit([enc.finish()]); }
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 20;
};
await timeFwd(all);
out.fwd_ms = await timeFwd(all);
out.fwd_no_attn_ms = await timeFwd(all.filter((o) => o.k !== "attn"));
out.n_dispatch = all.length;
m.ops = all;
if (ctx.inputs.noise) {
  await m.sample({ cls: 0, H: 64, W: 64, steps: 24, cfg: 2, noise: ctx.inputs.noise });
  await device.queue.onSubmittedWorkDone();
  out.img = await m.read(m.dBuf, 3 * 64 * 64);
}
ctx.progress(1, "done");
out.adapter = info && info.adapter;
return out;
