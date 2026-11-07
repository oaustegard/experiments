// GPU-bridge job: the exported viewer's path on real hardware. The 12-region baked shader of the parrots,
// drawn into a WebGPU canvas context (OffscreenCanvas, as a worker has no DOM) exactly as collage.html
// draws, then the canvas texture read back. inputs: s12 (scene JSON from run/mac_inputs.mjs)
const td = new TextDecoder();
const s = JSON.parse(td.decode(ctx.inputs.s12));
const device = await ctx.gpu();
let lost = null;
device.lost.then((i) => { lost = i.message; });
const VIEW = `
struct V { x0: f32, y0: f32, scale: f32, _p: f32 }
@group(0) @binding(0) var<uniform> view: V;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(xy * 2.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(collage(vec2f(view.x0, view.y0) + pos.xy * view.scale, view.scale), 1.0); }`;
const W = 768, H = 512;
const cv = new OffscreenCanvas(W, H);
const gc = cv.getContext("webgpu");
gc.configure({ device, format: "rgba8unorm", alphaMode: "opaque", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const t0 = performance.now();
const mod = device.createShaderModule({ code: s.baked + VIEW });
const errs = (await mod.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => m.message);
const pipe = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module: mod, entryPoint: "vs" }, fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
const compileMs = performance.now() - t0;
const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(u, 0, new Float32Array([0, 0, s.W0 / W, 0]));
const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u } }] });
const tex = gc.getCurrentTexture();
const bpr = W * 4, rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const e = device.createCommandEncoder();
const p = e.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
p.setPipeline(pipe); p.setBindGroup(0, bg); p.draw(3); p.end();
e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
device.queue.submit([e.finish()]);
await rb.mapAsync(GPUMapMode.READ);
const px = new Uint8Array(rb.getMappedRange().slice(0));
rb.unmap();
let sum = 0, nonblack = 0;
for (let i = 0; i < W * H; i++) { const v = px[4 * i] + px[4 * i + 1] + px[4 * i + 2]; sum += v; if (v > 30) nonblack++; }
return { errs, lost, compileMs, meanRGB: sum / (3 * W * H), nonblackFrac: nonblack / (W * H), image: px };
