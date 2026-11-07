// GPU-bridge job: single forward passes on real hardware, compared with torch on the page side where possible.
// inputs: src, man, w (real weights), xin (2x3xHxW f32 for H in 64,128), tman, tw, cases (fixture)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
const out = {};
// 1. fixture cases (random weights), same as run/test_gpu.mjs on SwiftShader
const tm = new TextureDiffusion(device, JSON.parse(td.decode(ctx.inputs.tman)), f16ToF32(ctx.inputs.tw));
const cases = JSON.parse(td.decode(ctx.inputs.cases));
out.fixture = [];
for (const c of cases.filter((c) => c.kind === "forward")) {
  const got = await tm.forward(new Float32Array(c.x), c.t, c.y, c.H, c.W, c.pad === "circular");
  let maxd = 0, maxw = 0;
  for (let i = 0; i < c.v.length; i++) { maxd = Math.max(maxd, Math.abs(got[i] - c.v[i])); maxw = Math.max(maxw, Math.abs(c.v[i])); }
  out.fixture.push({ H: c.H, W: c.W, pad: c.pad, rel: maxd / maxw });
}
tm.freePlan();
// 2. real weights: forward at 64 and 128 on fixed inputs, returned for comparison
const m = new TextureDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const x = ctx.inputs.xin;
const x64 = x.subarray(0, 2 * 3 * 64 * 64), x128 = x.subarray(2 * 3 * 64 * 64);
out.v64 = await m.forward(x64, 0.6, [23, 47], 64, 64, true);
out.v128 = await m.forward(x128, 0.6, [23, 47], 128, 128, true);
out.v128b = await m.forward(x128, 0.6, [23, 47], 128, 128, true);   // repeat: nondeterminism?
out.errs = errs;
return out;
