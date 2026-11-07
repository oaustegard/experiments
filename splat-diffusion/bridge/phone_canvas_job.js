// GPU-bridge job for a phone: replay the page's sampling loop with a real WebGPU canvas (OffscreenCanvas),
// first as the old page did it (direct full-screen paint after every step), then with the cached-tile painter.
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32, Painter, WGSL_PAINT } = await import(url);
const device = await ctx.gpu();
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
let lost = null;
device.lost.then((i) => { lost = i.message; });
const m = new SplatDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const W = 780, H = 1690;                      // an iPhone screen at the page's DPR cap of 2
const fmt = navigator.gpu.getPreferredCanvasFormat();
const out = { old: null, cached: null, errs, lost: null };
// old page: direct paint of every pixel x every splat after each step
{
  const canvas = new OffscreenCanvas(W, H);
  const cx = canvas.getContext("webgpu");
  cx.configure({ device, format: fmt, alphaMode: "opaque" });
  const mod = device.createShaderModule({ code: WGSL_PAINT });
  const pipe = device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] } });
  const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const per = [];
  let t = performance.now();
  try {
    await m.sample({ cls: 23, steps: 24, cfg: 3, seed: 3, onStep: async () => {
      const ab = new ArrayBuffer(16), dv = new DataView(ab);
      dv.setUint32(0, m.K, true); dv.setFloat32(4, 768, true);
      device.queue.writeBuffer(u, 0, ab);
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: m.b.splats } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: cx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const n = performance.now(); per.push(Math.round(n - t)); t = n;
    } });
    out.old = { ok: !lost, step_ms: per, lost };
  } catch (e) { out.old = { ok: false, error: e.message, step_ms: per, lost }; }
  ctx.log("old: " + JSON.stringify(out.old));
  ctx.progress(0.5, "old loop done");
}
// new page: cached tile + blit, previews skipped while one is in flight
if (!lost) {
  const canvas = new OffscreenCanvas(W, H);
  const painter = new Painter(device, canvas, fmt);
  const per = [];
  let t = performance.now(), painting = false;
  try {
    await m.sample({ cls: 23, steps: 24, cfg: 3, seed: 3, onStep: async () => {
      if (!painting) {
        painting = true;
        painter.setSplats(m.b.splats, m.K);
        painter.draw(768, 0, 0);
        device.queue.onSubmittedWorkDone().then(() => { painting = false; });
      }
      const n = performance.now(); per.push(Math.round(n - t)); t = n;
    } });
    await device.queue.onSubmittedWorkDone();
    out.cached = { ok: !lost, step_ms: per, lost };
  } catch (e) { out.cached = { ok: false, error: e.message, step_ms: per, lost }; }
  ctx.log("cached: " + JSON.stringify(out.cached));
}
out.lost = lost;
return out;
