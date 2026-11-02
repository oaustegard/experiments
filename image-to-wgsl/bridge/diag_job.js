// GPU Bridge job: where does the instanced renderer stall on WebKit? Logs each step with its time.
const spec = JSON.parse(new TextDecoder().decode(ctx.inputs.spec));
const ad = await navigator.gpu.requestAdapter();
const dev = await ad.requestDevice({ requiredFeatures: ad.features.has("float32-blendable") ? ["float32-blendable"] : [] });
ctx.log("float32-blendable " + dev.features.has("float32-blendable"));
function createRenderer(dev, fmt, code) {
  const mod = dev.createShaderModule({ code });
  const ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const OVER = { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                 alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" } };
  const mk = (v, f, format, blend) => dev.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: v },
    fragment: { module: mod, entryPoint: f, targets: [{ format, blend }] }, primitive: { topology: "triangle-list" } });
  const FT = dev.features.has("float32-blendable") ? "rgba32float" : "rgba16float";
  const pBg = mk("vb", "fb", FT), pSp = mk("vs", "fs", FT, OVER), pCp = mk("vc", "fc", fmt);
  const ubg = (p) => dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  const gBg = ubg(pBg), gSp = ubg(pSp);
  let tex = null, gCp = null, tw = 0, th = 0;
  return {
    format: FT,
    draw(view, w, h, n) {
      if (w !== tw || h !== th) {
        tex?.destroy();
        tex = dev.createTexture({ size: [w, h], format: FT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        gCp = dev.createBindGroup({ layout: pCp.getBindGroupLayout(0), entries: [{ binding: 1, resource: tex.createView() }] });
        tw = w; th = h;
      }
      dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, 0, 0]));
      const enc = dev.createCommandEncoder();
      let pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
      pass.setPipeline(pBg); pass.setBindGroup(0, gBg); pass.draw(3);
      if (n > 0) { pass.setPipeline(pSp); pass.setBindGroup(0, gSp); pass.draw(6, n); }
      pass.end();
      pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
      pass.setPipeline(pCp); pass.setBindGroup(0, gCp); pass.draw(3); pass.end();
      dev.queue.submit([enc.finish()]);
    },
  };
}
const T = () => performance.now(); const out = [];
for (const s of spec.shaders) {
  let t = T(); const mod = dev.createShaderModule({ code: s.code }); const info = await mod.getCompilationInfo();
  ctx.log(s.id + " compile " + (T() - t).toFixed(0) + " ms, messages " + JSON.stringify(info.messages.map((m) => m.message).slice(0, 3)));
  t = T(); const rd = createRenderer(dev, "rgba8unorm", s.code); ctx.log(s.id + " pipelines " + (T() - t).toFixed(0) + " ms");
  const tex = dev.createTexture({ size: [768, 512], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  for (const n of [16, 256, s.n]) {
    t = T(); rd.draw(tex.createView(), 768, 512, n);
    const done = await Promise.race([dev.queue.onSubmittedWorkDone().then(() => "ok"), new Promise((r) => setTimeout(() => r("timeout"), 8000))]);
    ctx.log(s.id + " draw n=" + n + " " + done + " " + (T() - t).toFixed(0) + " ms"); out.push({ id: s.id, n, done, ms: T() - t });
    if (done !== "ok") break;
  }
}
return { out };
