// GPU-bridge job: the splat collage on real hardware. For each scene: compile time of the baked shader,
// and frame time at 1024 and 2048 px square (the collage fitted inside) for the storage-buffer shader the
// page uses and for the baked one an exported file uses. inputs: s6, s12, s24 (scene JSON from run/mac_inputs.mjs)
const td = new TextDecoder();
const device = await ctx.gpu();
const VIEW = `
struct V { x0: f32, y0: f32, scale: f32, _p: f32 }
@group(0) @binding(0) var<uniform> view: V;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(xy * 2.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(collage(vec2f(view.x0, view.y0) + pos.xy * view.scale, view.scale), 1.0); }`;
async function pipeline(code) {
  const t0 = performance.now();
  const mod = device.createShaderModule({ code });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(errs.slice(0, 3).map((m) => `${m.lineNum}: ${m.message}`).join("; "));
  const pipe = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
  return { pipe, ms: performance.now() - t0 };
}
async function frames(pipe, entries, N) {
  const tex = device.createTexture({ size: [N, N], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  const ms = [];
  for (let i = 0; i < 8; i++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    ms.push(performance.now() - t0);
  }
  tex.destroy();
  const w = ms.slice(2).sort((a, b) => a - b);
  return { median_ms: w[w.length >> 1], first_ms: ms[0] };
}
const out = { runs: [] };
let n = 0;
for (const key of ["s6", "s12", "s24"]) {
  const s = JSON.parse(td.decode(ctx.inputs[key]));
  const buf = (arr, F) => { const a = new F(arr), b = device.createBuffer({ size: Math.max(16, a.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, a); return b; };
  const st = await pipeline(s.storage);
  const bk = await pipeline(s.baked + VIEW);
  const r = { regions: s.K, bakedBytes: s.baked.length, storageCompileMs: st.ms, bakedCompileMs: bk.ms };
  const spB = buf(s.sp, Float32Array), lbB = buf(s.labels, Uint32Array), xfB = buf(s.xf, Float32Array);
  for (const N of [1024, 2048]) {
    const scale = Math.max(s.W0, s.H0) / N;
    const u = new Uint32Array(s.uni); const f = new Float32Array(u.buffer); f[9] = 0; f[10] = -(N * scale - s.H0) / 2; f[11] = scale;
    const ub = buf(u, Uint32Array);
    r[`storage_${N}`] = await frames(st.pipe, [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: spB } },
      { binding: 2, resource: { buffer: lbB } }, { binding: 3, resource: { buffer: xfB } }], N);
    const vb = buf([0, -(N * scale - s.H0) / 2, scale, 0], Float32Array);
    r[`baked_${N}`] = await frames(bk.pipe, [{ binding: 0, resource: { buffer: vb } }], N);
  }
  out.runs.push(r);
  ctx.log(JSON.stringify(r));
  ctx.progress(++n / 3, key);
}
out.adapter = (await ctx.describe()).adapter;
return out;
