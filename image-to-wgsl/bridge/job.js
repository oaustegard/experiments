// GPU Bridge job: render image-to-wgsl shaders on a real GPU.
// Input `spec` (u8, UTF-8 JSON): [{id, wgsl, w, h, sizes: [[w, h], ...], reps}]
// Output per shader: img_<id> (u8 RGBA at w x h), and in `rows` the compile messages,
// a repeat-render determinism check and median frame ms per size (wall clock to onSubmittedWorkDone).
const spec = JSON.parse(new TextDecoder().decode(ctx.inputs.spec));
const dev = await ctx.gpu();
const info = ctx.describe ? await ctx.describe() : null;
const out = { rows: [], adapter: info };
const ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

async function renderOnce(pipe, bg, w, h, read) {
  const tex = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, 0, 0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  let buf, bpr;
  if (read) {
    bpr = Math.ceil(w * 4 / 256) * 256;
    buf = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  }
  const t0 = performance.now();
  dev.queue.submit([enc.finish()]);
  await dev.queue.onSubmittedWorkDone();
  const ms = performance.now() - t0;
  let px = null;
  if (read) {
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    px = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) px.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap(); buf.destroy();
  }
  tex.destroy();
  return { ms, px };
}

let done = 0;
const total = spec.reduce((a, s) => a + 2 + s.sizes.length * (s.reps + 1), 0);
for (const s of spec) {
  const row = { id: s.id };
  dev.pushErrorScope("validation");
  const mod = dev.createShaderModule({ code: s.wgsl });
  const ci = await mod.getCompilationInfo();
  row.messages = ci.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  const pipe = dev.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  const err = await dev.popErrorScope();
  if (err) { row.error = err.message; out.rows.push(row); continue; }
  const a = await renderOnce(pipe, bg, s.w, s.h, true); ctx.progress(++done / total, s.id);
  const b = await renderOnce(pipe, bg, s.w, s.h, true); ctx.progress(++done / total, s.id);
  let ndiff = 0;
  for (let i = 0; i < a.px.length; i++) if (a.px[i] !== b.px[i]) ndiff++;
  row.repeat_bytes_differ = ndiff;
  out["img_" + s.id] = a.px;
  row.ms = {};
  // Guard: a frame over SLOW_MS ends this shader's timing (one slow frame is the measurement,
  // and repeating it can run the job past its timeout, which loses every result).
  const SLOW_MS = 2000;
  let slow = false;
  for (const [w, h] of s.sizes) {
    if (slow) { row.ms[`${w}x${h}`] = { skipped: true }; done += s.reps + 1; continue; }
    const t = [];
    for (let r = 0; r <= s.reps; r++) {
      const x = await renderOnce(pipe, bg, w, h, false);
      ctx.progress(++done / total, `${s.id} ${w}x${h}`);
      if (x.ms > SLOW_MS) { t.push(x.ms); slow = true; done += s.reps - r; break; }
      if (r > 0) t.push(x.ms);
    }
    t.sort((p, q) => p - q);
    row.ms[`${w}x${h}`] = { median: t[Math.floor(t.length / 2)], min: t[0], max: t[t.length - 1], n: t.length, slow };
  }
  ctx.log(s.id, JSON.stringify(row.ms));
  out.rows.push(row);
}
return out;
